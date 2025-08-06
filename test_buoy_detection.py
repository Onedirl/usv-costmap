#!/usr/bin/env python3
"""
Duba tespit sistemini test etme scripti
Kamera veya video dosyası ile test yapabilirsiniz
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import argparse

class SimpleBuoyDetectionTest:
    """Basit duba tespit test sınıfı (ZED kamera olmadan)"""
    
    def __init__(self, model_path: str, video_source=0):
        """
        model_path: YOLO model dosyası
        video_source: 0 (webcam) veya video dosya yolu
        """
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_source)
        
        # Test cost map (basit 2D grid)
        self.simple_cost_map = np.zeros((100, 100), dtype=np.float32)
        self.map_scale = 10  # piksel/metre
        
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
        info_text = f"Tespit: {len(detections)} duba"
        cv2.putText(vis_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
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
            
            # Duba tespiti
            detections = self.detect_buoys(frame)
            
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
        
        tester = SimpleBuoyDetectionTest(args.model, args.source)
        tester.run()
        
    except Exception as e:
        print(f"Hata: {e}")

if __name__ == "__main__":
    main()
