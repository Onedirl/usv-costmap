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

@dataclass
class Buoy:
    """Duba veri yapısı"""
    id: int
    color: str  # 'yellow' veya 'orange'
    position_3d: np.ndarray  # [x, y, z] metre cinsinden
    position_2d: Tuple[int, int]  # Görüntüdeki piksel konumu
    confidence: float
    timestamp: float
    last_seen: float

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
    """YOLO tabanlı duba tespit sistemi"""
    def __init__(self, model_path: str):
        # YOLO modeli yükle
        self.model = YOLO(model_path)
        
        # ZED kamera başlat
        self.zed = sl.Camera()
        self.init_zed_camera()
        
        # Cost map yöneticisi
        self.cost_map_manager = CostMapManager()
        
        # Duba takibi
        self.buoys = {}  # ID -> Buoy
        self.next_buoy_id = 0
        
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
        """Duba takibini güncelle"""
        current_time = time.time()
        
        with self.lock:
            # Mevcut dubaları işaretle
            for buoy_id in self.buoys:
                self.buoys[buoy_id].last_seen = current_time
            
            # Yeni tespitleri işle
            for det in detections:
                center_x, center_y = det['center']
                position_3d = self.get_3d_position(center_x, center_y)
                
                if position_3d is not None:
                    # En yakın mevcut dubayı bul
                    min_distance = float('inf')
                    matched_id = None
                    
                    for buoy_id, buoy in self.buoys.items():
                        distance = np.linalg.norm(position_3d - buoy.position_3d)
                        if distance < 2.0 and distance < min_distance:  # 2 metre eşik
                            min_distance = distance
                            matched_id = buoy_id
                    
                    if matched_id is not None:
                        # Mevcut dubayı güncelle
                        self.buoys[matched_id].position_3d = position_3d
                        self.buoys[matched_id].position_2d = (center_x, center_y)
                        self.buoys[matched_id].confidence = det['confidence']
                        self.buoys[matched_id].timestamp = current_time
                        self.buoys[matched_id].last_seen = current_time
                    else:
                        # Yeni duba ekle
                        new_buoy = Buoy(
                            id=self.next_buoy_id,
                            color=det['class'],
                            position_3d=position_3d,
                            position_2d=(center_x, center_y),
                            confidence=det['confidence'],
                            timestamp=current_time,
                            last_seen=current_time
                        )
                        self.buoys[self.next_buoy_id] = new_buoy
                        self.next_buoy_id += 1
                        
                        # Cost map'e ekle
                        self.cost_map_manager.add_buoy_to_map(new_buoy)
            
            # Uzun süredir görülmeyen dubaları temizle
            buoys_to_remove = []
            for buoy_id, buoy in self.buoys.items():
                if current_time - buoy.last_seen > 10.0:  # 10 saniye
                    buoys_to_remove.append(buoy_id)
            
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
        
        # 3D pozisyonları göster
        with self.lock:
            for buoy in self.buoys.values():
                x, y = buoy.position_2d
                text = f"ID:{buoy.id} [{buoy.position_3d[0]:.1f}m, {buoy.position_3d[1]:.1f}m]"
                cv2.putText(vis_frame, text, (x-50, y+30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
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
                    'position': b.position_3d.tolist(),
                    'confidence': b.confidence
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
    
    # Sistemi başlat
    detection_system = BuoyDetectionSystem(MODEL_PATH)
    detection_system.run()

if __name__ == "__main__":
    main()
