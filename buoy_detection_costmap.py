# /home/onedir/halildeneme/buoy_detection_costmap.py
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
import argparse
import os

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
    # Takip alanları
    xy_enu: Optional[Tuple[float, float]] = None
    kf_state: Optional[np.ndarray] = None   # [x, y, vx, vy]
    kf_P: Optional[np.ndarray] = None       # 4x4 kovaryans
    hits: int = 1
    misses: int = 0
    confirmed: bool = False

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

    def global_to_xy(self, gp: GlobalPosition) -> Tuple[float, float]:
        """Origin'e göre yaklaşık ENU XY (metre)"""
        if not self.initialized:
            self.set_origin(gp.latitude, gp.longitude, gp.altitude)
        dlat = math.radians(gp.latitude - self.origin_lat)
        dlon = math.radians(gp.longitude - self.origin_lon)
        x_north = dlat * self.earth_radius
        y_east  = dlon * self.earth_radius * math.cos(math.radians(self.origin_lat))
        return (x_north, y_east)

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
        # Duba pozisyonunu grid koordinatlarına çevir (lokal x,y)
        grid_x, grid_y = self.world_to_grid(buoy.position_3d_local[0], buoy.position_3d_local[1])
        
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
    def __init__(self, model_path: str, mavlink_connection: str = None, mavlink_baud: Optional[int] = None,
                 headless: bool = False, save_json_interval: Optional[float] = None, save_dir: Optional[str] = None):
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
        self.mavlink_baud = mavlink_baud or 57600
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

        # Takip parametreleri
        self.confirm_hits = 2
        self.max_age_s = 2.0
        self.class_base_radius = {'orange': 0.6, 'yellow': 1.2}
        self.radius_min = 0.5
        self.radius_max = 2.5
        self.mahalanobis_gate = 5.0  # ~chi2 2D
        
        # Headless ve otomatik kayıt ayarları
        self.headless = headless
        self.save_json_interval = save_json_interval
        self.save_dir = save_dir or os.getcwd()
        os.makedirs(self.save_dir, exist_ok=True)
        self._last_json_save_ts = 0.0

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
            # Seri ise baud ile bağlan, UDP/TCP ise doğrudan
            if connection_string.startswith('/dev') or connection_string.lower().startswith('com'):
                self.mavlink = mavutil.mavlink_connection(connection_string, baud=self.mavlink_baud)
            else:
                self.mavlink = mavutil.mavlink_connection(connection_string)
            print("MAVLink bağlantısı kuruluyor...")
            self.mavlink.wait_heartbeat()
            print("✓ MAVLink bağlantısı başarılı!")

            # Mesaj aralıklarını iste (10 Hz)
            try:
                self._set_message_interval(33, 100000)  # GLOBAL_POSITION_INT
                self._set_message_interval(30, 100000)  # ATTITUDE
            except Exception:
                pass
        except Exception as e:
            print(f"⚠ MAVLink bağlantısı kurulamadı: {e}")
            print("  GPS/IMU verileri olmadan devam ediliyor...")
            self.mavlink = None

    def _set_message_interval(self, msg_id: int, interval_us: int):
        """Otopilottan belirli mesajı belirli sıklıkta yayınlamasını iste."""
        if not self.mavlink:
            return
        self.mavlink.mav.command_long_send(
            self.mavlink.target_system,
            self.mavlink.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            msg_id,
            float(interval_us),
            0, 0, 0, 0, 0
        )
    
    def update_gps_data(self) -> bool:
        """GPS verilerini güncelle"""
        if not self.mavlink:
            return False
            
        try:
            # Önce GLOBAL_POSITION_INT dene (lat/lon/alt içerir)
            gpi = self.mavlink.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
            if gpi:
                self.current_gps = GlobalPosition(
                    latitude=gpi.lat / 1e7,
                    longitude=gpi.lon / 1e7,
                    altitude=gpi.alt / 1000.0,
                    timestamp=time.time()
                )
                self.last_gps_time = time.time()
                return True

            # Sonra GPS_RAW_INT dene
            gps_msg = self.mavlink.recv_match(type='GPS_RAW_INT', blocking=False)
            if gps_msg and getattr(gps_msg, 'fix_type', 0) >= 3:
                self.current_gps = GlobalPosition(
                    latitude=gps_msg.lat / 1e7,
                    longitude=gps_msg.lon / 1e7,
                    altitude=gps_msg.alt / 1000.0,
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

    def get_3d_position_from_bbox(self, bbox: List[int]) -> Optional[np.ndarray]:
        """BBox içinden medyan 3B nokta (gürültüye dayanıklı)"""
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(self.image.get_width()-1, x2)
        y2 = min(self.image.get_height()-1, y2)
        xs, ys, zs = [], [], []
        step = max(1, (max(1, (x2 - x1)) // 8))
        for yy in range(y1, y2, step):
            for xx in range(x1, x2, step):
                err, val = self.point_cloud.get_value(xx, yy)
                if err == sl.ERROR_CODE.SUCCESS:
                    X, Y, Z = float(val[0]), float(val[1]), float(val[2])
                    if np.isfinite(X) and np.isfinite(Y) and np.isfinite(Z) and (0.3 <= Z <= 60.0):
                        xs.append(X); ys.append(Y); zs.append(Z)
        if len(zs) < 5:
            return None
        mx = float(np.median(xs)); my = float(np.median(ys)); mz = float(np.median(zs))
        return np.array([mx, my, mz], dtype=np.float32)

    # KF yardımcıları
    def _init_kf(self, xy: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        x, y = xy
        state = np.array([x, y, 0.0, 0.0], dtype=np.float32)
        P = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float32)
        return state, P

    def _kf_predict(self, buoy: Buoy, dt: float):
        if buoy.kf_state is None:
            return
        F = np.array([[1,0,dt,0],
                      [0,1,0,dt],
                      [0,0,1, 0],
                      [0,0,0, 1]], dtype=np.float32)
        q = 0.5
        G = np.array([[0.5*dt*dt, 0],[0,0.5*dt*dt],[dt,0],[0,dt]], dtype=np.float32)
        Q = (q*q) * (G @ G.T)
        buoy.kf_state = F @ buoy.kf_state
        buoy.kf_P = F @ buoy.kf_P @ F.T + Q

    def _kf_update(self, buoy: Buoy, z_xy: Tuple[float, float]):
        if buoy.kf_state is None:
            return
        H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        r = 0.5  # metre
        R = np.eye(2, dtype=np.float32) * (r*r)
        z = np.array(z_xy, dtype=np.float32)
        y = z - (H @ buoy.kf_state)
        S = H @ buoy.kf_P @ H.T + R
        K = buoy.kf_P @ H.T @ np.linalg.inv(S)
        buoy.kf_state = buoy.kf_state + K @ y
        I = np.eye(4, dtype=np.float32)
        buoy.kf_P = (I - K @ H) @ buoy.kf_P

    def _predict_xy(self, buoy: Buoy, now_t: float) -> Tuple[float, float]:
        dt = max(0.0, now_t - buoy.last_seen)
        if buoy.kf_state is None or buoy.kf_P is None:
            return buoy.xy_enu if buoy.xy_enu is not None else (0.0, 0.0)
        F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        pred = F @ buoy.kf_state
        return float(pred[0]), float(pred[1])

    def _dynamic_radius(self, color: str, range_m: float) -> float:
        base = self.class_base_radius.get(color, 1.0)
        rad = base + 0.01 * max(0.0, range_m)
        return max(self.radius_min, min(self.radius_max, rad))
    
    def update_buoy_tracking(self, detections: List[dict]):
        """Global koordinat destekli duba takibini güncelle"""
        current_time = time.time()
        
        # GPS ve IMU verilerini güncelle
        self.update_gps_data()
        self.update_imu_data()

        measurements = []
        for det in detections:
            bbox = det['bbox']
            pos3d = self.get_3d_position_from_bbox(bbox)
            if pos3d is None:
                continue
            gp = None
            xy = None
            if self.has_valid_navigation_data():
                try:
                    gp = self.coordinate_transform.local_to_global(pos3d, self.current_gps, self.current_imu)
                    xy = self.coordinate_transform.global_to_xy(gp)
                except Exception:
                    gp = None
                    xy = None
            measurements.append({
                'class': det['class'],
                'conf': det['confidence'],
                'pos3d': pos3d,
                'gp': gp,
                'xy': xy,
                'center': det['center'],
                'bbox': bbox,
                'range_m': float(np.linalg.norm(pos3d))
            })

        with self.lock:
            matched_ids = set()
            used_buoys = set()

            # Eşleştirme: yüksek güven sırasıyla
            for m in sorted(measurements, key=lambda x: x['conf'], reverse=True):
                if m['xy'] is None:
                    continue
                mx, my = m['xy']
                radius = self._dynamic_radius(m['class'], m['range_m'])

                best_id = None
                best_d = 1e9

                for bid, b in self.buoys.items():
                    if b.color != m['class'] or bid in used_buoys or b.xy_enu is None:
                        continue
                    px, py = self._predict_xy(b, current_time)
                    d = math.hypot(mx - px, my - py)
                    if d > radius:
                        continue
                    # Mahalanobis kapısı
                    ok_maha = True
                    if b.kf_P is not None and b.kf_state is not None:
                        H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
                        r = 0.5
                        R = np.eye(2, dtype=np.float32) * (r*r)
                        S = H @ b.kf_P @ H.T + R
                        dz = np.array([mx, my], dtype=np.float32) - (H @ b.kf_state)
                        try:
                            maha = float(np.sqrt(dz.T @ np.linalg.inv(S) @ dz))
                        except np.linalg.LinAlgError:
                            maha = 0.0
                        ok_maha = (maha <= self.mahalanobis_gate)
                    if ok_maha and d < best_d:
                        best_d = d
                        best_id = bid

                if best_id is not None:
                    b = self.buoys[best_id]
                    # KF predict + update
                    self._kf_predict(b, max(0.0, current_time - b.last_seen))
                    self._kf_update(b, (mx, my))
                    # Alanları güncelle
                    b.position_3d_local = m['pos3d']
                    b.position_2d = m['center']
                    b.confidence = m['conf']
                    b.timestamp = current_time
                    b.last_seen = current_time
                    b.detection_count += 1
                    b.position_global = m['gp'] if m['gp'] else b.position_global
                    b.xy_enu = (mx, my)
                    b.hits += 1
                    b.misses = 0
                    if not b.confirmed and b.hits >= self.confirm_hits:
                        b.confirmed = True
                    # Cost map'i güncelle (lokal x,y ile)
                    try:
                        self.cost_map_manager.add_buoy_to_map(b)
                    except Exception:
                        pass
                    used_buoys.add(best_id)
                    matched_ids.add(best_id)
                else:
                    # Yeni track aç
                    if m['conf'] >= 0.5 and m['xy'] is not None:
                        nb = Buoy(
                            id=self.next_buoy_id,
                            color=m['class'],
                            position_3d_local=m['pos3d'],
                            position_global=m['gp'],
                            position_2d=m['center'],
                            confidence=m['conf'],
                            timestamp=current_time,
                            last_seen=current_time,
                            detection_count=1,
                            xy_enu=m['xy']
                        )
                        state, P = self._init_kf(nb.xy_enu)
                        nb.kf_state, nb.kf_P = state, P
                        self.buoys[self.next_buoy_id] = nb
                        # Cost map'e ekle
                        try:
                            self.cost_map_manager.add_buoy_to_map(nb)
                        except Exception:
                            pass
                        used_buoys.add(self.next_buoy_id)
                        self.next_buoy_id += 1

            # Görülmeyenleri yaşlandır/temizle
            for bid, b in list(self.buoys.items()):
                if bid not in matched_ids:
                    b.misses += 1

            # Uzun süredir hiç görülmeyen ve doğrulanmamışları sil
            for bid, b in list(self.buoys.items()):
                if (current_time - b.last_seen) > self.max_age_s and not b.confirmed:
                    del self.buoys[bid]

            # Çok uzun süredir hiç görülmeyenleri (confirmed bile olsa) ayıkla
            self.cleanup_old_buoys(current_time)
    
    def find_matching_buoy(self, local_pos: np.ndarray, global_pos: Optional[GlobalPosition]) -> Optional[int]:
        """(Artık kullanılmıyor)"""
        return None
    
    def cleanup_old_buoys(self, current_time: float):
        """Uzun süredir görülmeyen dubaları temizle"""
        timeout = 30.0  # 30 saniye
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
                    if not self.headless:
                        # Görselleştir ve ekrana bas
                        vis_frame = self.visualize(frame, detections)
                        cv2.imshow("USV Duba Tespiti", vis_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('c'):
                            plt.figure(figsize=(10, 10))
                            plt.imshow(self.cost_map_manager.cost_map, cmap='hot', origin='lower')
                            plt.colorbar(label='Cost')
                            plt.title('Cost Map')
                            plt.xlabel('X (cells)')
                            plt.ylabel('Y (cells)')
                            plt.show()
                        elif key == ord('s'):
                            out = os.path.join(self.save_dir, f"cost_map_{int(time.time())}.json")
                            self.save_cost_map(out)
                            print(f"Cost map kaydedildi: {out}")
                        elif key == ord('q'):
                            break
                    else:
                        # Headless: otomatik JSON kayıt (varsa)
                        if self.save_json_interval and (time.time() - self._last_json_save_ts) >= self.save_json_interval:
                            out = os.path.join(self.save_dir, f"cost_map_{int(time.time())}.json")
                            self.save_cost_map(out)
                            self._last_json_save_ts = time.time()
                            print(f"[HEADLESS] Cost map kaydedildi: {out}")
                        
        except KeyboardInterrupt:
            print("\nSistem durduruldu.")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Temizlik işlemleri"""
        if not self.headless:
            cv2.destroyAllWindows()
        self.zed.close()
        print("Sistem kapatıldı.")

def main():
    parser = argparse.ArgumentParser(description="USV Duba Tespiti - ZED2i + YOLO + MAVLink")
    parser.add_argument("--model", type=str, default="path/to/your/buoy_model.pt", help="YOLO .pt model yolu")
    parser.add_argument("--mavlink", type=str, default=None, help="MAVLink bağlantı stringi (örn. /dev/ttyUSB0, udp:127.0.0.1:14550)")
    parser.add_argument("--baud", type=int, default=57600, help="Seri bağlantı baud hızı")
    parser.add_argument("--headless", action="store_true", help="Görüntü penceresiz (SSH/headless) çalışma modu")
    parser.add_argument("--save-json-interval", type=float, default=None, help="Headless modda periyodik JSON kayıt süresi (saniye)")
    parser.add_argument("--save-dir", type=str, default=None, help="Kayıtların yazılacağı dizin")
    args = parser.parse_args()

    MODEL_PATH = args.model
    MAVLINK_CONNECTION = args.mavlink
    MAVLINK_BAUD = args.baud
    HEADLESS = args.headless
    SAVE_JSON_INTERVAL = args.save_json_interval
    SAVE_DIR = args.save_dir

    print("=== USV Duba Tespiti - Global Koordinat Destekli ===")
    if MAVLINK_CONNECTION:
        print(f"MAVLink bağlantısı: {MAVLINK_CONNECTION} (baud={MAVLINK_BAUD})")
        print("GPS/IMU verileri ile global koordinat desteği aktif")
    else:
        print("Sadece kamera verileri kullanılıyor (lokal koordinat)")

    detection_system = BuoyDetectionSystem(
        model_path=MODEL_PATH,
        mavlink_connection=MAVLINK_CONNECTION,
        mavlink_baud=MAVLINK_BAUD,
        headless=HEADLESS,
        save_json_interval=SAVE_JSON_INTERVAL,
        save_dir=SAVE_DIR,
    )
    detection_system.run()

if __name__ == "__main__":
    main()