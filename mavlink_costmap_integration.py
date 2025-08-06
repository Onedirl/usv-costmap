#!/usr/bin/env python3
"""
MAVLink ve Cost Map Entegrasyonu
USV kontrolü ile duba tespit sistemini birleştiren modül
"""

from pymavlink import mavutil
import numpy as np
import threading
import time
import cv2
from buoy_detection_costmap import BuoyDetectionSystem, CostMapManager
import struct
import json

class USVCostMapNavigation:
    """Cost map destekli USV navigasyon sistemi"""
    
    def __init__(self, connection_string: str, model_path: str):
        # MAVLink bağlantısı
        self.master = mavutil.mavlink_connection(connection_string)
        self.target_system = 1
        self.target_component = 1
        self.wait_heartbeat()
        
        # Duba tespit sistemi
        self.detection_system = BuoyDetectionSystem(model_path)
        
        # USV durumu
        self.current_position = np.array([0.0, 0.0])  # [x, y] metre
        self.current_heading = 0.0  # derece
        self.current_speed = 0.0  # m/s
        
        # Thread kontrolü
        self.running = False
        self.detection_thread = None
        self.mavlink_thread = None
        
        # Cost map paylaşımı
        self.cost_map_update_interval = 1.0  # saniye
        self.last_cost_map_update = 0
        
    def wait_heartbeat(self):
        """MAVLink heartbeat bekle"""
        print("MAVLink heartbeat bekleniyor...")
        self.master.wait_heartbeat()
        print("Heartbeat alındı!")
    
    def start(self):
        """Sistemi başlat"""
        self.running = True
        
        # Thread'leri başlat
        self.detection_thread = threading.Thread(target=self.detection_loop)
        self.mavlink_thread = threading.Thread(target=self.mavlink_loop)
        
        self.detection_thread.start()
        self.mavlink_thread.start()
        
        print("USV Cost Map Navigasyon sistemi başlatıldı!")
    
    def stop(self):
        """Sistemi durdur"""
        self.running = False
        
        if self.detection_thread:
            self.detection_thread.join()
        if self.mavlink_thread:
            self.mavlink_thread.join()
            
        self.detection_system.cleanup()
        print("Sistem durduruldu.")
    
    def detection_loop(self):
        """Duba tespit döngüsü"""
        while self.running:
            try:
                frame, detections = self.detection_system.process_frame()
                
                if frame is not None:
                    # Görselleştirme
                    vis_frame = self.detection_system.visualize(frame, detections)
                    cv2.imshow("USV Duba Tespiti", vis_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break
                        
                    # Cost map güncellemesi gerekiyor mu?
                    current_time = time.time()
                    if current_time - self.last_cost_map_update > self.cost_map_update_interval:
                        self.send_cost_map_to_mavlink()
                        self.last_cost_map_update = current_time
                        
            except Exception as e:
                print(f"Tespit döngüsü hatası: {e}")
                
        cv2.destroyAllWindows()
    
    def mavlink_loop(self):
        """MAVLink veri alım döngüsü"""
        while self.running:
            try:
                # GPS pozisyonu
                gps_msg = self.master.recv_match(type='GPS_RAW_INT', blocking=False)
                if gps_msg:
                    # Basit lokal koordinat dönüşümü (gerçek uygulamada daha karmaşık)
                    # Bu örnek için başlangıç noktasından olan mesafeyi metre olarak alıyoruz
                    self.current_position[0] = gps_msg.lat / 1e7 * 111320  # yaklaşık
                    self.current_position[1] = gps_msg.lon / 1e7 * 111320
                
                # Yön bilgisi
                attitude_msg = self.master.recv_match(type='ATTITUDE', blocking=False)
                if attitude_msg:
                    self.current_heading = attitude_msg.yaw * 57.2958  # rad to deg
                
                # Hız bilgisi
                vfr_msg = self.master.recv_match(type='VFR_HUD', blocking=False)
                if vfr_msg:
                    self.current_speed = vfr_msg.groundspeed
                
                # Cost map tabanlı güvenlik kontrolü
                self.check_collision_risk()
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"MAVLink döngüsü hatası: {e}")
    
    def send_cost_map_to_mavlink(self):
        """Cost map'i MAVLink üzerinden gönder"""
        try:
            # Cost map'i küçült ve sıkıştır
            cost_map = self.detection_system.cost_map_manager.cost_map
            
            # 50x50'ye yeniden boyutlandır (veri miktarını azaltmak için)
            small_map = cv2.resize(cost_map, (50, 50))
            
            # 0-255 aralığına normalize et
            normalized_map = (small_map * 255 / 100).astype(np.uint8)
            
            # TERRAIN_DATA mesajı olarak gönder (custom olarak kullanıyoruz)
            # Gerçek uygulamada custom MAVLink mesajı tanımlanmalı
            
            # Şimdilik debug mesajı olarak gönder
            debug_text = f"COSTMAP:{len(self.detection_system.buoys)} buoys detected"
            self.master.mav.statustext_send(
                mavutil.mavlink.MAV_SEVERITY_INFO,
                debug_text.encode('utf-8')
            )
            
            print(f"Cost map güncellendi: {len(self.detection_system.buoys)} duba")
            
        except Exception as e:
            print(f"Cost map gönderme hatası: {e}")
    
    def check_collision_risk(self):
        """Çarpışma riski kontrolü"""
        # Aracın önündeki bölgeyi kontrol et
        lookahead_distance = 5.0  # metre
        
        # Aracın yönündeki noktayı hesapla
        heading_rad = np.radians(self.current_heading)
        future_x = self.current_position[0] + lookahead_distance * np.cos(heading_rad)
        future_y = self.current_position[1] + lookahead_distance * np.sin(heading_rad)
        
        # Cost değerini kontrol et
        cost = self.detection_system.cost_map_manager.get_cost_at_position(future_x, future_y)
        
        if cost > 80:  # Yüksek risk
            self.send_warning("YÜKSEK ÇARPIŞMA RİSKİ!")
            self.emergency_maneuver()
        elif cost > 50:  # Orta risk
            self.send_warning("Dikkat: Yakında engel var!")
    
    def send_warning(self, message: str):
        """Uyarı mesajı gönder"""
        print(f"⚠️  {message}")
        self.master.mav.statustext_send(
            mavutil.mavlink.MAV_SEVERITY_WARNING,
            message.encode('utf-8')
        )
    
    def emergency_maneuver(self):
        """Acil manevra"""
        print("🚨 ACİL MANEVRA BAŞLATILIYOR!")
        
        # Hızı düşür
        self.send_rc_override(throttle=1300)  # Yavaşla
        
        # En güvenli yönü bul
        best_heading = self.find_safe_heading()
        
        if best_heading is not None:
            # Güvenli yöne dön
            self.turn_to_heading(best_heading)
    
    def find_safe_heading(self):
        """En güvenli yönü bul"""
        min_cost = float('inf')
        best_heading = None
        
        # -90 ile +90 derece arasında tara
        for angle_offset in range(-90, 91, 10):
            test_heading = self.current_heading + angle_offset
            heading_rad = np.radians(test_heading)
            
            # Test noktası
            test_distance = 5.0
            test_x = self.current_position[0] + test_distance * np.cos(heading_rad)
            test_y = self.current_position[1] + test_distance * np.sin(heading_rad)
            
            cost = self.detection_system.cost_map_manager.get_cost_at_position(test_x, test_y)
            
            if cost < min_cost:
                min_cost = cost
                best_heading = test_heading
        
        return best_heading
    
    def turn_to_heading(self, target_heading: float):
        """Belirli bir yöne dön"""
        heading_error = target_heading - self.current_heading
        
        # -180 ile 180 arasına normalize et
        while heading_error > 180:
            heading_error -= 360
        while heading_error < -180:
            heading_error += 360
        
        # Dönüş yönünü belirle
        if heading_error > 0:
            steering = 1700  # Sağa dön
        else:
            steering = 1300  # Sola dön
        
        # Dönüş süresi (basit hesaplama)
        turn_duration = abs(heading_error) / 30.0  # 30 derece/saniye varsayımı
        
        # Dönüşü gerçekleştir
        start_time = time.time()
        while time.time() - start_time < turn_duration:
            self.send_rc_override(throttle=1550, steering=steering)
            time.sleep(0.1)
        
        # Düzelt
        self.send_rc_override(throttle=1500, steering=1500)
    
    def send_rc_override(self, throttle=1500, steering=1500):
        """RC override gönder"""
        self.master.mav.rc_channels_override_send(
            self.target_system, self.target_component,
            throttle, steering, 1500, 1500, 1500, 1500, 1500, 1500
        )
    
    def save_mission_data(self, filename: str):
        """Görev verilerini kaydet"""
        data = {
            'timestamp': time.time(),
            'usv_position': self.current_position.tolist(),
            'usv_heading': self.current_heading,
            'detected_buoys': [
                {
                    'id': b.id,
                    'color': b.color,
                    'position': b.position_3d.tolist(),
                    'first_seen': b.timestamp,
                    'last_seen': b.last_seen
                }
                for b in self.detection_system.buoys.values()
            ],
            'cost_map': {
                'size': self.detection_system.cost_map_manager.map_size,
                'resolution': self.detection_system.cost_map_manager.resolution,
                'data': self.detection_system.cost_map_manager.cost_map.tolist()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Görev verileri kaydedildi: {filename}")

def main():
    # Parametreler
    MAVLINK_CONNECTION = "/dev/ttyUSB0"  # MAVLink bağlantısı
    YOLO_MODEL = "path/to/your/buoy_model.pt"  # YOLO model dosyası
    
    try:
        # Sistemi başlat
        nav_system = USVCostMapNavigation(MAVLINK_CONNECTION, YOLO_MODEL)
        nav_system.start()
        
        print("\nUSV Cost Map Navigasyon Sistemi Çalışıyor!")
        print("Komutlar:")
        print("  q: Çıkış")
        print("  s: Verileri kaydet")
        print("  c: Cost map göster")
        
        # Ana döngü
        while nav_system.running:
            command = input()
            
            if command == 'q':
                nav_system.running = False
            elif command == 's':
                nav_system.save_mission_data(f"mission_data_{int(time.time())}.json")
            elif command == 'c':
                # Cost map görselleştirme
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 10))
                plt.imshow(nav_system.detection_system.cost_map_manager.cost_map, 
                          cmap='hot', origin='lower')
                plt.colorbar(label='Cost')
                plt.title('Current Cost Map')
                plt.show()
        
        # Sistemi kapat
        nav_system.stop()
        
    except KeyboardInterrupt:
        print("\nSistem kullanıcı tarafından durduruldu.")
        nav_system.stop()
    except Exception as e:
        print(f"Sistem hatası: {e}")
        nav_system.stop()

if __name__ == "__main__":
    main()
