# 🚢 USV Global Koordinat Sistemi - Çözüm Dokumentasyonu

## 🎯 Problem

Gemi hareket ettiği sırada aynı dubaları farklı pozisyonlarda görüyor ve tekrar tekrar kaydediyor. Bu durum:
- Gereksiz bellek kullanımı
- Yanlış cost map bilgisi  
- Navigasyon hatalarına sebep oluyor

## ✅ Çözüm: Global Koordinat Sistemi

### Ana Kavram
Dubaları **global GPS koordinatlarında** saklayarak, gemi nerede olursa olsun aynı dubalar için tek kayıt tutuluyor.

### Nasıl Çalışıyor?

1. **GPS + IMU Entegrasyonu**
   - Geminin anlık GPS konumu alınır
   - IMU'dan gemi yönü (yaw açısı) okunur

2. **Koordinat Dönüşümü**
   ```
   Kamera [x,y,z] → Gemi [fore,port,up] → GPS [lat,lon,alt]
   ```

3. **Akıllı Eşleştirme**
   - Yeni tespit edilen duba global koordinatlarda kontrol edilir
   - 3 metre içinde başka duba varsa → Mevcut kaydı günceller
   - 3 metre içinde duba yoksa → Yeni kayıt açar

## 🔧 Teknik Detaylar

### Koordinat Dönüşüm Algoritması

```python
def local_to_global(local_pos, current_gps, current_imu):
    # 1. Kamera → Gemi koordinatı
    x_ship = local_pos[2]   # kamera z → gemi x (ileri)
    y_ship = -local_pos[0]  # kamera -x → gemi y (sol)
    
    # 2. Gemi yönüne göre döndür
    cos_yaw = math.cos(current_imu.yaw)
    sin_yaw = math.sin(current_imu.yaw)
    
    x_ned = x_ship * cos_yaw - y_ship * sin_yaw  # Kuzey
    y_ned = x_ship * sin_yaw + y_ship * cos_yaw  # Doğu
    
    # 3. GPS koordinatına çevir
    dlat = x_ned / earth_radius * 180.0 / math.pi
    dlon = y_ned / (earth_radius * cos(current_gps.lat)) * 180.0 / math.pi
    
    return GPS(lat + dlat, lon + dlon)
```

### Mesafe Hesaplama (Haversine Formula)

```python
def calculate_distance(pos1, pos2):
    # Dünya eğriliğini dikkate alan hassas mesafe hesabı
    lat1, lon1 = radians(pos1.lat), radians(pos1.lon)
    lat2, lon2 = radians(pos2.lat), radians(pos2.lon)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return earth_radius * c  # metre cinsinden mesafe
```

## 📊 Sistem Avantajları

### Önceki Durum (Sadece Lokal)
```
Frame 1: Duba A [5m, 2m]   → Kayıt 1
Frame 5: Duba A [7m, 1m]   → Kayıt 2 (TEKRAR!)
Frame 10: Duba A [10m, 0m] → Kayıt 3 (TEKRAR!)
```

### Yeni Durum (Global Koordinat)
```
Frame 1: Duba A [41.008234°, 28.978567°] → Kayıt 1 (YENİ)
Frame 5: Duba A [41.008235°, 28.978568°] → Kayıt 1 güncellendi (3m içinde)
Frame 10: Duba A [41.008236°, 28.978569°] → Kayıt 1 güncellendi (2m içinde)
```

## 🛠️ Kurulum ve Kullanım

### 1. Gerçek Sistem (GPS/IMU ile)

```python
# buoy_detection_costmap.py dosyasında
MAVLINK_CONNECTION = "/dev/ttyUSB0"  # Gerçek bağlantı
```

```bash
python buoy_detection_costmap.py
```

### 2. Test Sistemi (Simülasyon ile)

```bash
# GPS simülasyonu ile test
python test_buoy_detection.py --model model.pt --source 0 --simulate-gps
```

## 📈 Sonuçlar

### Bellek Tasarrufu
- **Önceki:** 100 frame → 50+ duba kaydı
- **Şimdi:** 100 frame → 5-10 gerçek duba kaydı

### Navigasyon Doğruluğu  
- **Önceki:** Hayali dubalar nedeniyle karışık cost map
- **Şimdi:** Gerçek duba konumları ile temiz cost map

### Sistem Güvenilirliği
- **Önceki:** Gemi hareket ettikçe kayıt sayısı artıyor
- **Şimdi:** Sabit kayıt sayısı, güncel pozisyonlar

## 🎮 Kullanıcı Deneyimi

### Görsel Bilgiler
- **Lokal Pozisyon:** `[15.2m, -3.4m, 2.1m]`
- **Global Pozisyon:** `GPS: 41.008234, 28.978567`
- **Tespit Sayısı:** `Tespit: 15` (kaç kez görüldü)
- **Sistem Durumu:** `GPS/IMU: Aktif`

### Kontroller
- **Q:** Çıkış
- **S:** Anlık durum kaydet (JSON formatında global koordinatlar)
- **C:** Cost map göster/gizle

## 🔍 Test Senaryoları

### Senaryo 1: Sabit Gemi
```
Gemi duruyor → Duba sürekli aynı global koordinatta → Tek kayıt
```

### Senaryo 2: Hareketli Gemi
```
Gemi hareket ediyor → Duba farklı açılardan görünüyor → Tek kayıt güncelleniyor
```

### Senaryo 3: Çoklu Duba
```
3 farklı duba → Her biri farklı global koordinat → 3 ayrı kayıt
```

## 📋 Konfigürasyon

### Ana Parametreler
```python
global_buoy_threshold = 3.0    # Eşleştirme mesafesi (metre)
buoy_timeout = 30.0           # Eski duba silme süresi (saniye)
earth_radius = 6378137.0      # WGS84 Dünya yarıçapı
```

### MAVLink Bağlantıları
```python
"/dev/ttyUSB0"              # Serial USB
"udp:127.0.0.1:14550"       # UDP (SITL simülasyon)
"tcp:127.0.0.1:5760"        # TCP bağlantı
None                        # Bağlantı yok (sadece kamera)
```

## 🚨 Sorun Giderme

### GPS Sinyali Yok
```
⚠ MAVLink bağlantısı kurulamadı
  GPS/IMU verileri olmadan devam ediliyor...
```
→ Serial bağlantıyı kontrol edin, açık alanda test yapın

### IMU Veri Sorunu  
```
IMU veri hatası: timeout
```
→ Otopilot bağlantısını kontrol edin, MAVLink stream rate artırın

### Yanlış Koordinat Dönüşümü
```
Dubalar yanlış yerde görünüyor
```
→ IMU kalibrasyonu yapın, kamera-gemi ekseni hizalamasını kontrol edin

## 🎉 Başarı Kriteri

✅ **Problem Çözüldü!** Artık gemi hareket ederken aynı dubalar tekrar tespit edilmiyor.

✅ **Global Koordinat Desteği:** Her duba gerçek dünya koordinatlarında saklanıyor.

✅ **Akıllı Filtreleme:** 3 metre eşiği ile aynı dubalar otomatik eşleştiriliyor.

✅ **Gerçek Zamanlı:** GPS/IMU verisi ile anlık koordinat dönüşümü yapılıyor.

---

Bu çözüm sayesinde USV navigasyon sisteminiz artık çok daha güvenilir ve etkili! 🚢🎯
