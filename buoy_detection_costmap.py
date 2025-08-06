#!/usr/bin/env python3
"""
USV Duba Tespiti ve Cost Map Oluşturma Sistemi
ZED 2i kamera ile YOLO kullanarak sarı/turuncu duba tespiti
"""

import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
import threading
from queue import Queue
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json
from datetime import datetime
from pymavlink import mavutil
import math

@dataclass
class GlobalPosition:
    """Global koordinat sistemi pozisyonu"""
    latitude: float  # derece
    longitude: float  # derece
    altitude: float  # metre (deniz seviyesinden)
    timestamp: float

@dataclass
class IMUData:
    """IMU sensor verisi"""
    roll: float  # radyan
    pitch: float  # radyan
    yaw: float  # radyan (kuzeyden saat yönü)
    timestamp: float

@dataclass
class Buoy:
    """Duba veri yapısı - Global koordinat destekli"""
    id: int
    color: str  # 'yellow' veya 'orange'
    position_3d_local: np.ndarray  # [x, y, z] kamera koordinatlarında
    position_global: Optional[GlobalPosition]  # Global koordinatlarda pozisyon
    position_2d: Tuple[int, int]  # Görüntüdeki piksel konumu
    confidence: float
    timestamp: float
    last_seen: float
    detection_count: int = 1  # Kaç kez tespit edildi

class GlobalCoordinateTransform:
    """Global koordinat dönüşümü sınıfı"""
    def __init__(self):
        self.origin_lat = None
        self.origin_lon = None
        self.origin_alt = None
        self.initialized = False
        
        # Earth radius in meters
        self.earth_radius = 6378137.0
        
    def set_origin(self, lat: float, lon: float, alt: float):
        """Referans noktasını ayarla (ilk GPS konumu)"""
        self.origin_lat = lat
        self.origin_lon = lon
        self.origin_alt = alt
        self.initialized = True
        print(f"Global koordinat origin ayarlandı: {lat:.6f}, {lon:.6f}, {alt:.1f}m")
    
    def local_to_global(self, local_pos: np.ndarray, current_gps: GlobalPosition, 
                       current_imu: IMUData) -> GlobalPosition:
        """
        Yerel kamera koordinatlarını global koordinatlara çevir
        local_pos: [x, y, z] kamera koordinatlarında (x:ileri, y:sol, z:yukarı)
        """
        if not self.initialized:
            # İlk GPS konumunu origin olarak ayarla
            self.set_origin(current_gps.latitude, current_gps.longitude, current_gps.altitude)
        
        # Kamera koordinatlarını gemi koordinat sistemine çevir
        # ZED kamera: x=sağ, y=aşağı, z=ileri
        # Gemi: x=ileri, y=sol, z=yukarı olacak şekilde dönüştür
        x_ship = local_pos[2]  # kamera z -> gemi x (ileri)
        y_ship = -local_pos[0]  # kamera -x -> gemi y (sol)
        z_ship = -local_pos[1]  # kamera -y -> gemi z (yukarı)
        
        # IMU yaw açısına göre döndür (gemi yönü)
        cos_yaw = math.cos(current_imu.yaw)
        sin_yaw = math.sin(current_imu.yaw)
        
        # Döndürülmüş koordinatlar (NED - North East Down)
        x_ned = x_ship * cos_yaw - y_ship * sin_yaw  # Kuzey
        y_ned = x_ship * sin_yaw + y_ship * cos_yaw  # Doğu
        z_ned = z_ship  # Aşağı (derinlik)
        
        # NED koordinatlarını GPS'e çevir
        dlat = x_ned / self.earth_radius * 180.0 / math.pi
        dlon = y_ned / (self.earth_radius * math.cos(math.radians(current_gps.latitude))) * 180.0 / math.pi
        
        global_lat = current_gps.latitude + dlat
        global_lon = current_gps.longitude + dlon
        global_alt = current_gps.altitude - z_ned  # Deniz seviyesinden yükseklik
        
        return GlobalPosition(
            latitude=global_lat,
            longitude=global_lon, 
            altitude=global_alt,
            timestamp=time.time()
        )
    
    def calculate_distance(self, pos1: GlobalPosition, pos2: GlobalPosition) -> float:
        """İki global pozisyon arasındaki mesafeyi hesapla (Haversine formula)"""
        lat1, lon1 = math.radians(pos1.latitude), math.radians(pos1.longitude)
        lat2, lon2 = math.radians(pos2.latitude), math.radians(pos2.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        distance = self.earth_radius * c
        
        # Yükseklik farkını da ekle
        if pos1.altitude is not None and pos2.altitude is not None:
            dalt = pos2.altitude - pos1.altitude
            distance = math.sqrt(distance**2 + dalt**2)
            
        return distance

class CostMapManager:
    """Cost Map yönetim sınıfı"""
    def __init__(self, map_size=(200, 200), resolution=0.5):
        """
        map_size: Grid boyutu (cells)
        resolution: Her hücrenin gerçek boyutu (metre)
        """
        self.map_size = map_size
        self.resolution = resolution
        self.cost_map = np.zeros(map_size, dtype=np.float32)
        self.origin = (map_size[0]//2, map_size[1]//2)  # Harita merkezi
        
        # Duba parametreleri
        self.buoy_radius = 1.0  # metre
        self.safety_margin = 2.0  # metre
        self.max_cost = 100.0
        
    def world_to_grid(self, x, y):
        """Dünya koordinatlarını grid koordinatlarına çevir"""
        grid_x = int(self.origin[0] + x / self.resolution)
        grid_y = int(self.origin[1] - y / self.resolution)
        return grid_x, grid_y
    
    def add_buoy_to_map(self, buoy: Buoy):
        """Dubayı cost map'e ekle"""
        # Duba pozisyonunu grid koordinatlarına çevir
        grid_x, grid_y = self.world_to_grid(buoy.position_3d[0], buoy.position_3d[1])
        
        # Grid sınırları kontrolü
        if 0 <= grid_x < self.map_size[0] and 0 <= grid_y < self.map_size[1]:
            # Duba çevresinde cost gradient oluştur
            radius_cells = int((self.buoy_radius + self.safety_margin) / self.resolution)
            
            for i in range(max(0, grid_x - radius_cells), min(self.map_size[0], grid_x + radius_cells)):
                for j in range(max(0, grid_y - radius_cells), min(self.map_size[1], grid_y + radius_cells)):
                    distance = np.sqrt((i - grid_x)**2 + (j - grid_y)**2) * self.resolution
                    
                    if distance <= self.buoy_radius:
                        # Duba merkezi - maksimum cost
                        self.cost_map[j, i] = self.max_cost
                    elif distance <= self.buoy_radius + self.safety_margin:
                        # Güvenlik marjini - azalan cost
                        cost = self.max_cost * (1 - (distance - self.buoy_radius) / self.safety_margin)
                        self.cost_map[j, i] = max(self.cost_map[j, i], cost)
    
    def decay_costs(self, decay_rate=0.995):
        """Zamanla cost değerlerini azalt (eski gözlemler için)"""
        self.cost_map *= decay_rate
        
    def get_cost_at_position(self, x, y):
        """Belirli bir pozisyondaki cost değerini al"""
        grid_x, grid_y = self.world_to_grid(x, y)
        if 0 <= grid_x < self.map_size[0] and 0 <= grid_y < self.map_size[1]:
            return self.cost_map[grid_y, grid_x]
        return 0

class BuoyDetectionSystem:
    """YOLO tabanlı duba tespit sistemi - Global koordinat destekli"""
    def __init__(self, model_path: str, mavlink_connection: str = None):
        # YOLO modeli yükle
        self.model = YOLO(model_path)
        
        # ZED kamera başlat
        self.zed = sl.Camera()
        self.init_zed_camera()
        
        # Cost map yöneticisi
        self.cost_map_manager = CostMapManager()
        
        # Global koordinat sistemi
        self.coordinate_transform = GlobalCoordinateTransform()
        
        # MAVLink bağlantısı (opsiyonel)
        self.mavlink_connection = mavlink_connection
        self.mavlink = None
        if mavlink_connection:
            self.init_mavlink(mavlink_connection)
        
        # Mevcut GPS ve IMU verileri
        self.current_gps = None
        self.current_imu = None
        self.last_gps_time = 0
        self.last_imu_time = 0
        
        # Duba takibi - Global koordinat destekli
        self.buoys = {}  # ID -> Buoy
        self.next_buoy_id = 0
        self.global_buoy_threshold = 3.0  # Global koordinatlarda minimum mesafe (metre)
        
        # Thread güvenliği için
        self.lock = threading.Lock()
        self.detection_queue = Queue()
        
        # Görselleştirme
        self.fig = None
        self.ax_img = None
        self.ax_cost = None
        
    def init_zed_camera(self):
        """ZED 2i kamera başlatma"""
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.camera_fps = 30
        init.depth_mode = sl.DEPTH_MODE.ULTRA
        init.coordinate_units = sl.UNIT.METER
        init.depth_minimum_distance = 0.3
        init.depth_maximum_distance = 40.0
        
        status = self.zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"ZED kamera açılamadı: {status}")
            exit(1)
            
        # Runtime parametreleri
        self.runtime = sl.RuntimeParameters()
        self.runtime.sensing_mode = sl.SENSING_MODE.STANDARD
        
        # Görüntü ve derinlik matrisleri
        self.image = sl.Mat()
        self.depth = sl.Mat()
        self.point_cloud = sl.Mat()
        
        print("ZED 2i kamera başarıyla başlatıldı!")
    
    def init_mavlink(self, connection_string: str):
        """MAVLink bağlantısını başlat"""
        try:
            self.mavlink = mavutil.mavlink_connection(connection_string)
            print("MAVLink bağlantısı kuruluyor...")
            self.mavlink.wait_heartbeat()
            print("✓ MAVLink bağlantısı başarılı!")
        except Exception as e:
            print(f"⚠ MAVLink bağlantısı kurulamadı: {e}")
            print("  GPS/IMU verileri olmadan devam ediliyor...")
            self.mavlink = None
    
    def update_gps_data(self) -> bool:
        """GPS verilerini güncelle"""
        if not self.mavlink:
            return False
            
        try:
            # GPS verisi al (non-blocking)
            gps_msg = self.mavlink.recv_match(type='GPS_RAW_INT', blocking=False)
            if gps_msg and gps_msg.fix_type >= 3:  # 3D Fix gerekli
                self.current_gps = GlobalPosition(
                    latitude=gps_msg.lat / 1e7,  # mikro derece -> derece
                    longitude=gps_msg.lon / 1e7,
                    altitude=gps_msg.alt / 1000.0,  # mm -> metre
                    timestamp=time.time()
                )
                self.last_gps_time = time.time()
                return True
        except Exception as e:
            print(f"GPS veri hatası: {e}")
        
        return False
    
    def update_imu_data(self) -> bool:
        """IMU verilerini güncelle"""
        if not self.mavlink:
            return False
            
        try:
            # Attitude verisi al (roll, pitch, yaw)
            attitude_msg = self.mavlink.recv_match(type='ATTITUDE', blocking=False)
            if attitude_msg:
                self.current_imu = IMUData(
                    roll=attitude_msg.roll,
                    pitch=attitude_msg.pitch,
                    yaw=attitude_msg.yaw,
                    timestamp=time.time()
                )
                self.last_imu_time = time.time()
                return True
        except Exception as e:
            print(f"IMU veri hatası: {e}")
        
        return False
    
    def has_valid_navigation_data(self) -> bool:
        """Geçerli GPS ve IMU verisi var mı kontrol et"""
        current_time = time.time()
        gps_timeout = 2.0  # saniye
        imu_timeout = 1.0  # saniye
        
        gps_valid = (self.current_gps is not None and 
                    (current_time - self.last_gps_time) < gps_timeout)
        imu_valid = (self.current_imu is not None and 
                    (current_time - self.last_imu_time) < imu_timeout)
        
        return gps_valid and imu_valid
        
    def detect_buoys(self, frame: np.ndarray) -> List[dict]:
        """YOLO ile duba tespiti"""
        results = self.model(frame)
        detections = []
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # Sınıf ismini kontrol et (sarı/turuncu duba)
                    class_id = int(box.cls)
                    class_name = self.model.names[class_id]
                    
                    if 'yellow' in class_name.lower() or 'orange' in class_name.lower():
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf)
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class': 'yellow' if 'yellow' in class_name.lower() else 'orange',
                            'center': (int((x1+x2)/2), int((y1+y2)/2))
                        })
        
        return detections
    
    def get_3d_position(self, x: int, y: int) -> Optional[np.ndarray]:
        """2D piksel konumundan 3D dünya konumunu al"""
        err, point_cloud_value = self.point_cloud.get_value(x, y)
        
        if err == sl.ERROR_CODE.SUCCESS:
            point3D = np.array([
                point_cloud_value[0],
                point_cloud_value[1], 
                point_cloud_value[2]
            ])
            
            # NaN veya geçersiz değerleri kontrol et
            if not np.isnan(point3D).any() and point3D[2] > 0:
                return point3D
                
        return None
    
    def update_buoy_tracking(self, detections: List[dict]):
        """Global koordinat destekli duba takibini güncelle"""
        current_time = time.time()
        
        # GPS ve IMU verilerini güncelle
        self.update_gps_data()
        self.update_imu_data()
        
        with self.lock:
            # Yeni tespitleri işle
            for det in detections:
                center_x, center_y = det['center']
                position_3d_local = self.get_3d_position(center_x, center_y)
                
                if position_3d_local is not None:
                    # Global koordinata çevir (eğer GPS/IMU verisi varsa)
                    global_position = None
                    if self.has_valid_navigation_data():
                        try:
                            global_position = self.coordinate_transform.local_to_global(
                                position_3d_local, self.current_gps, self.current_imu
                            )
                        except Exception as e:
                            print(f"Global koordinat dönüşüm hatası: {e}")
                    
                    # Mevcut dubalar arasında eşleştirme ara
                    matched_buoy_id = self.find_matching_buoy(position_3d_local, global_position)
                    
                    if matched_buoy_id is not None:
                        # Mevcut dubayı güncelle
                        buoy = self.buoys[matched_buoy_id]
                        buoy.position_3d_local = position_3d_local
                        buoy.position_2d = (center_x, center_y)
                        buoy.confidence = det['confidence']
                        buoy.timestamp = current_time
                        buoy.last_seen = current_time
                        buoy.detection_count += 1
                        
                        # Global pozisyonu da güncelle
                        if global_position:
                            buoy.position_global = global_position
                            
                        print(f"Duba ID:{matched_buoy_id} güncellendi (toplam {buoy.detection_count} tespit)")
                    else:
                        # Yeni duba ekle
                        new_buoy = Buoy(
                            id=self.next_buoy_id,
                            color=det['class'],
                            position_3d_local=position_3d_local,
                            position_global=global_position,
                            position_2d=(center_x, center_y),
                            confidence=det['confidence'],
                            timestamp=current_time,
                            last_seen=current_time,
                            detection_count=1
                        )
                        self.buoys[self.next_buoy_id] = new_buoy
                        
                        # Global koordinat bilgisi yazdır
                        if global_position:
                            print(f"YENİ DUBA eklenildi - ID:{self.next_buoy_id}")
                            print(f"  Lokal: [{position_3d_local[0]:.1f}, {position_3d_local[1]:.1f}, {position_3d_local[2]:.1f}] m")
                            print(f"  Global: {global_position.latitude:.6f}, {global_position.longitude:.6f}")
                        else:
                            print(f"YENİ DUBA eklenildi - ID:{self.next_buoy_id} (sadece lokal koordinat)")
                        
                        self.next_buoy_id += 1
                        
                        # Cost map'e ekle (lokal koordinatlarla)
                        self.cost_map_manager.add_buoy_to_map(new_buoy)
            
            # Uzun süredir görülmeyen dubaları temizle
            self.cleanup_old_buoys(current_time)
    
    def find_matching_buoy(self, local_pos: np.ndarray, global_pos: Optional[GlobalPosition]) -> Optional[int]:
        """
        Yeni tespitin mevcut dubalardan hangisine ait olduğunu bul
        Global koordinat varsa önce global, yoksa lokal koordinat kullan
        """
        min_distance = float('inf')
        matched_id = None
        
        for buoy_id, buoy in self.buoys.items():
            distance = float('inf')
            
            # Önce global koordinatla karşılaştır (daha güvenilir)
            if global_pos and buoy.position_global:
                distance = self.coordinate_transform.calculate_distance(global_pos, buoy.position_global)
                threshold = self.global_buoy_threshold  # 3 metre
                
                if distance < threshold and distance < min_distance:
                    min_distance = distance
                    matched_id = buoy_id
            
            # Global koordinat yoksa lokal koordinatla karşılaştır
            elif not global_pos or not buoy.position_global:
                local_distance = np.linalg.norm(local_pos - buoy.position_3d_local)
                threshold = 2.0  # 2 metre (lokal için daha dar eşik)
                
                if local_distance < threshold and local_distance < min_distance:
                    min_distance = local_distance
                    matched_id = buoy_id
        
        return matched_id
    
    def cleanup_old_buoys(self, current_time: float):
        """Uzun süredir görülmeyen dubaları temizle"""
        timeout = 30.0  # 30 saniye (global koordinat için daha uzun timeout)
        buoys_to_remove = []
        
        for buoy_id, buoy in self.buoys.items():
            if current_time - buoy.last_seen > timeout:
                buoys_to_remove.append(buoy_id)
                if buoy.position_global:
                    print(f"Eski duba kaldırıldı - ID:{buoy_id} (Global: {buoy.position_global.latitude:.6f}, {buoy.position_global.longitude:.6f})")
                else:
                    print(f"Eski duba kaldırıldı - ID:{buoy_id} (Sadece lokal)")
        
        for buoy_id in buoys_to_remove:
            del self.buoys[buoy_id]
    
    def process_frame(self):
        """Tek frame işleme"""
        if self.zed.grab(self.runtime) == sl.ERROR_CODE.SUCCESS:
            # Görüntü ve derinlik verilerini al
            self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
            self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
            self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
            
            # OpenCV formatına çevir
            frame = self.image.get_data()
            frame_cv = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            
            # Duba tespiti
            detections = self.detect_buoys(frame_cv)
            
            # Takibi güncelle
            self.update_buoy_tracking(detections)
            
            # Cost map'i zamanla azalt
            self.cost_map_manager.decay_costs()
            
            return frame_cv, detections
        
        return None, []
    
    def visualize(self, frame: np.ndarray, detections: List[dict]):
        """Görselleştirme"""
        # Tespitleri çiz
        vis_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = (0, 255, 255) if det['class'] == 'yellow' else (0, 165, 255)
            
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_frame, f"{det['class']} {det['confidence']:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 3D pozisyonları ve global koordinatları göster
        with self.lock:
            for buoy in self.buoys.values():
                x, y = buoy.position_2d
                
                # Lokal pozisyon
                local_text = f"ID:{buoy.id} [{buoy.position_3d_local[0]:.1f}m, {buoy.position_3d_local[1]:.1f}m]"
                cv2.putText(vis_frame, local_text, (x-50, y+30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Global pozisyon (varsa)
                if buoy.position_global:
                    global_text = f"GPS: {buoy.position_global.latitude:.6f}, {buoy.position_global.longitude:.6f}"
                    cv2.putText(vis_frame, global_text, (x-50, y+45), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                
                # Tespit sayısı
                count_text = f"Tespit: {buoy.detection_count}"
                cv2.putText(vis_frame, count_text, (x-50, y+60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        
        return vis_frame
    
    def save_cost_map(self, filename: str):
        """Cost map'i kaydet"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'map_size': self.cost_map_manager.map_size,
            'resolution': self.cost_map_manager.resolution,
            'cost_map': self.cost_map_manager.cost_map.tolist(),
            'buoys': [
                {
                    'id': b.id,
                    'color': b.color,
                    'position_local': b.position_3d_local.tolist(),
                    'position_global': {
                        'latitude': b.position_global.latitude,
                        'longitude': b.position_global.longitude,
                        'altitude': b.position_global.altitude
                    } if b.position_global else None,
                    'confidence': b.confidence,
                    'detection_count': b.detection_count
                }
                for b in self.buoys.values()
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_cost_map_msg(self):
        """MAVLink için cost map mesajı hazırla"""
        # Cost map'i byte dizisine çevir
        cost_bytes = (self.cost_map_manager.cost_map * 255).astype(np.uint8).tobytes()
        
        msg = {
            'type': 'COST_MAP',
            'timestamp': time.time(),
            'map_size': self.cost_map_manager.map_size,
            'resolution': self.cost_map_manager.resolution,
            'origin': self.cost_map_manager.origin,
            'data': cost_bytes
        }
        
        return msg
    
    def run(self):
        """Ana çalışma döngüsü"""
        print("Duba tespit sistemi başlatılıyor...")
        
        try:
            while True:
                frame, detections = self.process_frame()
                
                if frame is not None:
                    # Görselleştir
                    vis_frame = self.visualize(frame, detections)
                    
                    # Görüntüyü göster
                    cv2.imshow("USV Duba Tespiti", vis_frame)
                    
                    # Cost map'i göster
                    if cv2.waitKey(1) & 0xFF == ord('c'):
                        plt.figure(figsize=(10, 10))
                        plt.imshow(self.cost_map_manager.cost_map, cmap='hot', origin='lower')
                        plt.colorbar(label='Cost')
                        plt.title('Cost Map')
                        plt.xlabel('X (cells)')
                        plt.ylabel('Y (cells)')
                        plt.show()
                    
                    # Çıkış
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                    # Cost map kaydet
                    if cv2.waitKey(1) & 0xFF == ord('s'):
                        self.save_cost_map(f"cost_map_{int(time.time())}.json")
                        print("Cost map kaydedildi!")
                        
        except KeyboardInterrupt:
            print("\nSistem durduruldu.")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Temizlik işlemleri"""
        cv2.destroyAllWindows()
        self.zed.close()
        print("Sistem kapatıldı.")

def main():
    # YOLO model yolu - kendi modelinizin yolunu buraya yazın
    MODEL_PATH = "path/to/your/buoy_model.pt"  # Değiştirin!
    
    # MAVLink bağlantısı (opsiyonel - GPS/IMU için)
    # Örnek bağlantı stringleri:
    # MAVLINK_CONNECTION = "/dev/ttyUSB0"  # Serial
    # MAVLINK_CONNECTION = "udp:127.0.0.1:14550"  # UDP
    # MAVLINK_CONNECTION = "tcp:127.0.0.1:5760"  # TCP
    MAVLINK_CONNECTION = None  # None = sadece kamera verileri kullan
    
    print("=== USV Duba Tespiti - Global Koordinat Destekli ===")
    if MAVLINK_CONNECTION:
        print(f"MAVLink bağlantısı: {MAVLINK_CONNECTION}")
        print("GPS/IMU verileri ile global koordinat desteği aktif")
    else:
        print("Sadece kamera verileri kullanılıyor (lokal koordinat)")
    
    # Sistemi başlat
    detection_system = BuoyDetectionSystem(MODEL_PATH, MAVLINK_CONNECTION)
    detection_system.run()

if __name__ == "__main__":
    main()
