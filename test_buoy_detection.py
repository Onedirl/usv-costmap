#!/usr/bin/env python3
"""
Duba tespit sistemini test etme scripti
Kamera veya video dosyası ile test yapabilirsiniz
Global koordinat simülasyonu destekli
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import argparse
import math
import json
from datetime import datetime

class SimpleBuoyDetectionTest:
    """Basit duba tespit test sınıfı - Global koordinat simülasyonu destekli"""
    
    def __init__(self, model_path: str, video_source=0, simulate_gps=False):
        """
        model_path: YOLO model dosyası
        video_source: 0 (webcam) veya video dosya yolu
        simulate_gps: GPS/IMU simülasyonu yap
        """
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_source)
        
        # Test cost map (basit 2D grid)
        self.simple_cost_map = np.zeros((100, 100), dtype=np.float32)
        self.map_scale = 10  # piksel/metre
        
        # Global koordinat simülasyonu
        self.simulate_gps = simulate_gps
        self.simulated_gps = {
            'lat': 41.0082,  # İstanbul koordinatları (örnek)
            'lon': 28.9784,
            'alt': 10.0,
            'yaw': 0.0  # radyan
        }
        
        # Tespit edilen dubaların global koordinatları
        self.global_buoys = {}  # ID -> {"lat": ..., "lon": ..., "count": ...}
        self.buoy_id_counter = 0
        self.global_threshold = 5.0  # metre
        
    def detect_buoys(self, frame):
        """YOLO ile duba tespiti"""
        results = self.model(frame)
        detections = []
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls)
                    class_name = self.model.names[class_id]
                    
                    # Sarı veya turuncu duba kontrolü
                    if any(color in class_name.lower() for color in ['yellow', 'orange', 'buoy']):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf)
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class': 'yellow' if 'yellow' in class_name.lower() else 'orange',
                            'center': (int((x1+x2)/2), int((y1+y2)/2))
                        })
        
        return detections
    
    def simulate_vehicle_movement(self, frame_count):
        """Araç hareketini simüle et (GPS değişimi)"""
        if not self.simulate_gps:
            return
            
        # Basit dairesel hareket simülasyonu
        t = frame_count * 0.1  # zaman
        radius = 0.0005  # ~50 metre
        
        # Merkez etrafında dönüş
        self.simulated_gps['lat'] = 41.0082 + radius * math.cos(t)
        self.simulated_gps['lon'] = 28.9784 + radius * math.sin(t)
        self.simulated_gps['yaw'] = t * 0.5  # Yavaş dönüş
        
    def local_to_global_simple(self, pixel_x, pixel_y, frame_shape):
        """Basit piksel koordinatını global koordinata çevir"""
        if not self.simulate_gps:
            return None
            
        # Kamera görüş alanını simüle et (örnek: 60 derece)
        frame_center_x = frame_shape[1] / 2
        frame_center_y = frame_shape[0] / 2
        
        # Pikseli metre cinsine çevir (yaklaşık)
        meters_per_pixel = 0.1  # Her piksel ~10cm temsil ediyor
        x_meters = (pixel_x - frame_center_x) * meters_per_pixel
        y_meters = (pixel_y - frame_center_y) * meters_per_pixel
        
        # Araç yönüne göre döndür
        cos_yaw = math.cos(self.simulated_gps['yaw'])
        sin_yaw = math.sin(self.simulated_gps['yaw'])
        
        x_rotated = x_meters * cos_yaw - y_meters * sin_yaw
        y_rotated = x_meters * sin_yaw + y_meters * cos_yaw
        
        # Global koordinata çevir (basit)
        earth_radius = 6378137.0
        dlat = x_rotated / earth_radius * 180.0 / math.pi
        dlon = y_rotated / (earth_radius * math.cos(math.radians(self.simulated_gps['lat']))) * 180.0 / math.pi
        
        global_lat = self.simulated_gps['lat'] + dlat
        global_lon = self.simulated_gps['lon'] + dlon
        
        return {'lat': global_lat, 'lon': global_lon}
    
    def calculate_distance(self, pos1, pos2):
        """İki global pozisyon arasındaki mesafe (Haversine)"""
        lat1, lon1 = math.radians(pos1['lat']), math.radians(pos1['lon'])
        lat2, lon2 = math.radians(pos2['lat']), math.radians(pos2['lon'])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return 6378137.0 * c  # metre
    
    def find_matching_global_buoy(self, global_pos):
        """Global koordinatta eşleşen dubayı bul"""
        if not global_pos:
            return None
            
        min_distance = float('inf')
        matched_id = None
        
        for buoy_id, buoy_data in self.global_buoys.items():
            distance = self.calculate_distance(global_pos, buoy_data)
            if distance < self.global_threshold and distance < min_distance:
                min_distance = distance
                matched_id = buoy_id
                
        return matched_id
    
    def update_global_buoys(self, detections, frame_shape):
        """Global duba takibini güncelle"""
        if not self.simulate_gps:
            return
            
        for det in detections:
            center_x, center_y = det['center']
            
            # Global koordinata çevir
            global_pos = self.local_to_global_simple(center_x, center_y, frame_shape)
            if global_pos:
                # Mevcut dubada eşleştirme ara
                matched_id = self.find_matching_global_buoy(global_pos)
                
                if matched_id is not None:
                    # Mevcut dubayı güncelle
                    self.global_buoys[matched_id]['count'] += 1
                    # Pozisyonu güncelle (ortalaması)
                    old_count = self.global_buoys[matched_id]['count'] - 1
                    self.global_buoys[matched_id]['lat'] = (
                        self.global_buoys[matched_id]['lat'] * old_count + global_pos['lat']
                    ) / self.global_buoys[matched_id]['count']
                    self.global_buoys[matched_id]['lon'] = (
                        self.global_buoys[matched_id]['lon'] * old_count + global_pos['lon']
                    ) / self.global_buoys[matched_id]['count']
                else:
                    # Yeni duba ekle
                    self.global_buoys[self.buoy_id_counter] = {
                        'lat': global_pos['lat'],
                        'lon': global_pos['lon'],
                        'color': det['class'],
                        'count': 1,
                        'first_seen': time.time()
                    }
                    print(f"YENİ GLOBAL DUBA - ID:{self.buoy_id_counter} "
                          f"({global_pos['lat']:.6f}, {global_pos['lon']:.6f})")
                    self.buoy_id_counter += 1
        
        return detections
    
    def update_simple_cost_map(self, detections, frame_shape):
        """Basit cost map güncelleme"""
        # Cost map'i zamanla azalt
        self.simple_cost_map *= 0.98
        
        # Tespit edilen dubaları ekle
        for det in detections:
            center_x, center_y = det['center']
            
            # Görüntü koordinatlarını harita koordinatlarına çevir
            map_x = int(center_x * 100 / frame_shape[1])
            map_y = int(center_y * 100 / frame_shape[0])
            
            # Cost ekle (Gaussian benzeri)
            for i in range(max(0, map_x-10), min(100, map_x+10)):
                for j in range(max(0, map_y-10), min(100, map_y+10)):
                    distance = np.sqrt((i-map_x)**2 + (j-map_y)**2)
                    if distance < 10:
                        cost = 100 * np.exp(-distance/5)
                        self.simple_cost_map[j, i] = max(self.simple_cost_map[j, i], cost)
    
    def visualize_frame(self, frame, detections):
        """Tespit sonuçlarını görselleştir"""
        vis_frame = frame.copy()
        
        # Tespit kutularını çiz
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = (0, 255, 255) if det['class'] == 'yellow' else (0, 165, 255)
            
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Etiket
            label = f"{det['class']} {det['confidence']:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(vis_frame, (x1, y1-20), (x1+label_size[0], y1), color, -1)
            cv2.putText(vis_frame, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Merkez noktası
            cx, cy = det['center']
            cv2.circle(vis_frame, (cx, cy), 5, color, -1)
        
        # Tespit sayısı
        info_text = f"Lokal tespit: {len(detections)}"
        cv2.putText(vis_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Global duba sayısı
        if self.simulate_gps:
            global_text = f"Global dubalar: {len(self.global_buoys)}"
            cv2.putText(vis_frame, global_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # GPS simülasyon bilgisi
            gps_text = f"GPS: {self.simulated_gps['lat']:.6f}, {self.simulated_gps['lon']:.6f}"
            cv2.putText(vis_frame, gps_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            yaw_text = f"Yaw: {math.degrees(self.simulated_gps['yaw']):.1f}°"
            cv2.putText(vis_frame, yaw_text, (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return vis_frame
    
    def visualize_cost_map(self):
        """Cost map'i görselleştir"""
        # Normalize et ve renk haritasına çevir
        normalized = (self.simple_cost_map * 255 / 100).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        
        # Büyüt
        display_map = cv2.resize(colored, (400, 400))
        
        # Başlık ekle
        cv2.putText(display_map, "Cost Map", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return display_map
    
    def run(self):
        """Test döngüsü"""
        print("Duba tespit testi başlatılıyor...")
        print("Komutlar:")
        print("  q: Çıkış")
        print("  s: Ekran görüntüsü al")
        print("  c: Cost map göster/gizle")
        
        show_cost_map = True
        frame_count = 0
        fps = 0
        start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Video sonu veya kamera hatası!")
                break
            
            # FPS hesapla
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - start_time)
                start_time = time.time()
            
            # Araç hareketini simüle et
            self.simulate_vehicle_movement(frame_count)
            
            # Duba tespiti
            detections = self.detect_buoys(frame)
            
            # Global koordinat güncellemesi
            self.update_global_buoys(detections, frame.shape)
            
            # Cost map güncelle
            self.update_simple_cost_map(detections, frame.shape)
            
            # Görselleştir
            vis_frame = self.visualize_frame(frame, detections)
            
            # FPS göster
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Ana pencere
            cv2.imshow("Duba Tespiti", vis_frame)
            
            # Cost map penceresi
            if show_cost_map:
                cost_display = self.visualize_cost_map()
                cv2.imshow("Cost Map", cost_display)
            
            # Klavye kontrolü
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = int(time.time())
                cv2.imwrite(f"buoy_detection_{timestamp}.jpg", vis_frame)
                cv2.imwrite(f"cost_map_{timestamp}.jpg", self.visualize_cost_map())
                
                # Global duba verilerini kaydet
                if self.simulate_gps and self.global_buoys:
                    global_data = {
                        'timestamp': datetime.now().isoformat(),
                        'vehicle_position': self.simulated_gps,
                        'global_buoys': self.global_buoys
                    }
                    with open(f"global_buoys_{timestamp}.json", 'w') as f:
                        json.dump(global_data, f, indent=2)
                    print(f"Görüntüler ve global veriler kaydedildi!")
                else:
                    print(f"Görüntüler kaydedildi!")
            elif key == ord('c'):
                show_cost_map = not show_cost_map
                if not show_cost_map:
                    cv2.destroyWindow("Cost Map")
        
        # Temizlik
        self.cap.release()
        cv2.destroyAllWindows()

def create_synthetic_buoy_video(output_path="synthetic_buoys.mp4", duration=30):
    """Test için sentetik duba videosu oluştur"""
    print("Sentetik duba videosu oluşturuluyor...")
    
    # Video yazıcı
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (1280, 720))
    
    # Duba parametreleri
    buoys = [
        {'x': 200, 'y': 300, 'vx': 2, 'vy': 0.5, 'color': 'yellow'},
        {'x': 800, 'y': 400, 'vx': -1.5, 'vy': -0.3, 'color': 'orange'},
        {'x': 500, 'y': 200, 'vx': 0.8, 'vy': 1.2, 'color': 'yellow'},
        {'x': 1000, 'y': 500, 'vx': -2, 'vy': -0.8, 'color': 'orange'},
    ]
    
    frames = duration * 30  # 30 FPS
    
    for frame_idx in range(frames):
        # Deniz arka planı
        img = np.ones((720, 1280, 3), dtype=np.uint8) * 50
        img[:,:,0] = 100  # Mavi ton
        
        # Dalgalar ekle (basit sinüs dalgası)
        for y in range(0, 720, 20):
            wave_offset = int(10 * np.sin(frame_idx * 0.1 + y * 0.02))
            cv2.line(img, (0, y + wave_offset), (1280, y + wave_offset), 
                    (120, 80, 40), 2)
        
        # Dubaları çiz
        for buoy in buoys:
            # Pozisyonu güncelle
            buoy['x'] += buoy['vx']
            buoy['y'] += buoy['vy']
            
            # Sınırları kontrol et
            if buoy['x'] < 50 or buoy['x'] > 1230:
                buoy['vx'] *= -1
            if buoy['y'] < 50 or buoy['y'] > 670:
                buoy['vy'] *= -1
            
            # Duba rengi
            if buoy['color'] == 'yellow':
                color = (0, 255, 255)
            else:
                color = (0, 165, 255)
            
            # Duba çiz (daire)
            center = (int(buoy['x']), int(buoy['y']))
            cv2.circle(img, center, 30, color, -1)
            cv2.circle(img, center, 30, (0, 0, 0), 2)
            
            # Üst kısım (konik)
            pts = np.array([
                [center[0] - 20, center[1] - 10],
                [center[0] + 20, center[1] - 10],
                [center[0], center[1] - 40]
            ], np.int32)
            cv2.fillPoly(img, [pts], color)
            
            # Yansıma
            cv2.ellipse(img, (center[0], center[1] + 5), (25, 10), 
                       0, 0, 180, (100, 100, 100), -1)
        
        # Frame numarası
        cv2.putText(img, f"Frame: {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(img)
        
        if frame_idx % 30 == 0:
            print(f"İşleniyor: {frame_idx/frames*100:.1f}%")
    
    out.release()
    print(f"Sentetik video oluşturuldu: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Duba Tespit Sistemi Test")
    parser.add_argument('--model', type=str, required=True, 
                       help='YOLO model dosyası yolu')
    parser.add_argument('--source', type=str, default=0,
                       help='Video kaynağı (0=webcam, veya video dosyası)')
    parser.add_argument('--create-synthetic', action='store_true',
                       help='Test için sentetik video oluştur')
    parser.add_argument('--simulate-gps', action='store_true',
                       help='GPS/IMU simülasyonu ile global koordinat testi')
    
    args = parser.parse_args()
    
    if args.create_synthetic:
        create_synthetic_buoy_video()
        print("Sentetik video ile test için:")
        print(f"python {__file__} --model {args.model} --source synthetic_buoys.mp4")
        return
    
    # Test sistemini başlat
    try:
        if args.source.isdigit():
            args.source = int(args.source)
        
        print("=== Duba Tespit Sistemi - Test Modu ===")
        if args.simulate_gps:
            print("GPS/IMU simülasyonu aktif - Global koordinat testi")
            print("Araç hareketi simüle edilecek, aynı dubalar tekrar tespit edilmeyecek")
        else:
            print("Sadece lokal tespit - Global koordinat yok")
        
        tester = SimpleBuoyDetectionTest(args.model, args.source, args.simulate_gps)
        tester.run()
        
    except Exception as e:
        print(f"Hata: {e}")

if __name__ == "__main__":
    main()
