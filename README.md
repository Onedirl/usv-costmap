# USV Duba Tespiti ve Cost Map Sistemi

Bu proje, insansız yüzey araçları (USV) için YOLO tabanlı duba tespiti ve cost map oluşturma sistemidir.

## 🚢 Sistem Özellikleri

- **YOLO v8** ile gerçek zamanlı sarı/turuncu duba tespiti
- **ZED 2i** stereo kamera ile 3D konum belirleme
- Dinamik **cost map** oluşturma ve güncelleme
- **MAVLink** protokolü ile USV entegrasyonu
- Otomatik çarpışma önleme sistemi
- Gerçek zamanlı görselleştirme

## 📋 Gereksinimler

### Donanım
- ZED 2i stereo kamera
- MAVLink uyumlu otopilot (Pixhawk, ArduPilot vb.)
- NVIDIA GPU (YOLO için önerilir)

### Yazılım
```bash
# Python gereksinimleri
pip install -r requirements.txt

# ZED SDK kurulumu (ayrıca yapılmalı)
# https://www.stereolabs.com/developers/release/
```

## 🚀 Kurulum

1. **ZED SDK Kurulumu**
   ```bash
   # Ubuntu için
   wget https://download.stereolabs.com/zedsdk/3.8/cu117/ubuntu20
   chmod +x ubuntu20
   ./ubuntu20
   ```

2. **Python bağımlılıkları**
   ```bash
   pip install -r requirements.txt
   ```

3. **YOLO Modeli**
   - Kendi duba modelinizi eğitin veya hazır model kullanın
   - Model dosyasını `.pt` formatında kaydedin

## 💻 Kullanım

### 1. Test Scripti (Kamera/Video ile)
```bash
# Webcam ile test
python test_buoy_detection.py --model path/to/model.pt --source 0

# Video dosyası ile test
python test_buoy_detection.py --model path/to/model.pt --source video.mp4

# Sentetik test videosu oluştur
python test_buoy_detection.py --create-synthetic
```

### 2. USV Entegrasyonu
```bash
# MAVLink ve duba tespiti birlikte
python mavlink_costmap_integration.py
```

### 3. Sadece Duba Tespiti
```bash
python buoy_detection_costmap.py
```

## 🗂️ Dosya Yapısı

- `pyhalilmav.py` - Orijinal USV kontrol kodu
- `buoy_detection_costmap.py` - Duba tespiti ve cost map oluşturma
- `mavlink_costmap_integration.py` - MAVLink entegrasyonu
- `test_buoy_detection.py` - Test ve simülasyon scripti
- `requirements.txt` - Python bağımlılıkları

## 🎮 Kontroller

### Test Modu
- **Q**: Çıkış
- **S**: Ekran görüntüsü kaydet
- **C**: Cost map göster/gizle

### USV Modu
- **Enter**: Sonraki test aşamasına geç
- **Ctrl+C**: Acil durdurma

## 📊 Cost Map Özellikleri

- **Dinamik güncelleme**: Yeni dubalar otomatik eklenir
- **Zaman bazlı azalma**: Eski gözlemler zamanla silinir
- **Gradient cost**: Duba merkezinden uzaklaştıkça azalan tehlike
- **Güvenlik marjini**: Her duba etrafında 2 metre güvenlik alanı

## 🔧 Yapılandırma

### Cost Map Parametreleri
```python
map_size = (200, 200)      # Grid boyutu
resolution = 0.5           # metre/hücre
buoy_radius = 1.0         # metre
safety_margin = 2.0       # metre
max_cost = 100.0          # maksimum tehlike değeri
```

### Kamera Parametreleri
```python
camera_resolution = HD720  # 1280x720
camera_fps = 30
depth_mode = ULTRA
depth_max_distance = 40.0  # metre
```

## 📈 Performans İpuçları

1. **GPU Kullanımı**: YOLO için CUDA destekli GPU kullanın
2. **Model Optimizasyonu**: YOLOv8n (nano) hızlı başlangıç için
3. **Cost Map Boyutu**: Geniş alanlar için düşük çözünürlük
4. **FPS**: Gerçek zamanlı için minimum 15 FPS hedefleyin

## 🚨 Güvenlik Notları

- Test sırasında her zaman manuel kontrol hazır bulundurun
- Gerçek su testlerinden önce simülasyonda test edin
- Acil durdurma mekanizmasını test edin
- GPS ve batarya durumunu sürekli kontrol edin

## 📝 Lisans

Bu proje eğitim amaçlıdır. Ticari kullanım için uygun lisansları kontrol edin.

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun
3. Değişikliklerinizi commit edin
4. Pull request gönderin

## 📞 Destek

Sorularınız için issue açabilir veya dokümantasyonu inceleyebilirsiniz.
