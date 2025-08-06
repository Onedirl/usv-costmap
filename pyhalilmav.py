#!/usr/bin/env python3
"""
USV Güvenli Test Prosedürü - Aşamalı Testing
"""

from pymavlink import mavutil
import time

class USVSafeTesting:
    def __init__(self, connection_string):
        self.master = mavutil.mavlink_connection(connection_string)
        self.target_system = 1
        self.target_component = 1
        self.wait_heartbeat()
        print("USV Test Moduna Hazır!")
    
    def wait_heartbeat(self):
        print("Heartbeat bekleniyor...")
        self.master.wait_heartbeat()
        
    def pre_flight_checks(self):
        """Test öncesi kontroller"""
        print("\n=== TEST ÖNCESİ KONTROLLER ===")
        
        # GPS durumu
        gps_msg = self.master.recv_match(type='GPS_RAW_INT', blocking=True, timeout=10)
        if gps_msg:
            if gps_msg.fix_type >= 3:
                print("✓ GPS Fix: İyi")
                print(f"  Satelit sayısı: {gps_msg.satellites_visible}")
            else:
                print("✗ GPS Fix: Yetersiz!")
                return False
        else:
            print("✗ GPS verisi alınamadı!")
            return False
            
        # Batarya durumu
        battery_msg = self.master.recv_match(type='BATTERY_STATUS', blocking=True, timeout=5)
        if battery_msg:
            voltage = battery_msg.voltages[0] / 1000.0 if battery_msg.voltages[0] != -1 else 0
            print(f"✓ Batarya: {voltage:.1f}V")
            if voltage < 11.0:  # 3S LiPo için minimum
                print("⚠ Batarya seviyesi düşük!")
        
        # Sistem durumu
        heartbeat = self.master.recv_match(type='HEARTBEAT', blocking=True, timeout=5)
        if heartbeat:
            if heartbeat.system_status == mavutil.mavlink.MAV_STATE_STANDBY:
                print("✓ Sistem durumu: Hazır")
            else:
                print(f"⚠ Sistem durumu: {heartbeat.system_status}")
        
        print("Test öncesi kontroller tamamlandı.\n")
        return True
    
    def stage_1_stationary_test(self):
        """Aşama 1: Sabit test (hareket yok)"""
        print("=== AŞAMA 1: SABİT TEST ===")
        print("Araç suya yerleştirildi, hareket yok")
        
        if not self.arm_vehicle():
            return False
            
        print("10 saniye bekleniliyor - sistem stabilitesi kontrol ediliyor...")
        for i in range(10):
            # Sadece durma sinyali gönder
            self.send_rc_override(throttle=1500, steering=1500)
            
            # Durum bilgisi al
            if i % 3 == 0:  # Her 3 saniyede bir durum yazdır
                self.print_attitude()
            
            time.sleep(1)
        
        self.disarm_vehicle()
        print("Aşama 1 tamamlandı. Devam etmek için onay bekliyor...\n")
        return True
    
    def stage_2_minimal_movement(self):
        """Aşama 2: Minimal hareket testi"""
        print("=== AŞAMA 2: MİNİMAL HAREKET ===")
        print("Çok düşük hızla kısa mesafe test")
        
        if not self.arm_vehicle():
            return False
        
        # Çok düşük hız - %8 güç, 2 saniye
        print("Minimal ileri hareket başlatılıyor...")
        throttle_pwm = 1500 + int(8 * 4)  # %8 hız
        
        for i in range(20):  # 2 saniye (20 * 0.1s)
            self.send_rc_override(throttle=throttle_pwm, steering=1500)
            time.sleep(0.1)
        
        # Dur
        for i in range(10):  # 1 saniye dur
            self.send_rc_override(throttle=1500, steering=1500)
            time.sleep(0.1)
        
        self.disarm_vehicle()
        print("Aşama 2 tamamlandı.\n")
        return True
    
    def stage_3_controlled_movement(self):
        """Aşama 3: Kontrollü hareket testi"""
        print("=== AŞAMA 3: KONTROLLÜ HAREKET ===")
        print("Orta hızda kontrollü hareket")
        
        if not self.arm_vehicle():
            return False
        
        # %15 hız, 5 saniye ileri
        print("Kontrollü ileri hareket...")
        self.move_forward_safe(speed_percent=15, duration=5)
        
        time.sleep(2)
        
        # Geri hareket testi
        print("Geri hareket testi...")
        self.move_backward_safe(speed_percent=10, duration=3)
        
        time.sleep(1)
        
        self.disarm_vehicle()
        print("Aşama 3 tamamlandı.\n")
        return True
    
    def stage_4_steering_test(self):
        """Aşama 4: Yönlendirme testi"""
        print("=== AŞAMA 4: YÖNLENDİRME TESTİ ===")
        
        if not self.arm_vehicle():
            return False
        
        # Sola dönüş
        print("Sola dönüş testi...")
        self.turn_test(direction="left", duration=3)
        
        time.sleep(2)
        
        # Sağa dönüş  
        print("Sağa dönüş testi...")
        self.turn_test(direction="right", duration=3)
        
        time.sleep(1)
        
        self.disarm_vehicle()
        print("Aşama 4 tamamlandı.\n")
        return True
    
    def move_forward_safe(self, speed_percent=10, duration=3):
        """Güvenli ileri hareket"""
        throttle_pwm = 1500 + int(speed_percent * 4)
        throttle_pwm = min(throttle_pwm, 1700)  # Güvenlik sınırı
        
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_rc_override(throttle=throttle_pwm, steering=1500)
            time.sleep(0.1)
        
        # Yumuşak durma
        for i in range(10):
            self.send_rc_override(throttle=1500, steering=1500)
            time.sleep(0.1)
    
    def move_backward_safe(self, speed_percent=10, duration=3):
        """Güvenli geri hareket"""
        throttle_pwm = 1500 - int(speed_percent * 4)
        throttle_pwm = max(throttle_pwm, 1300)  # Güvenlik sınırı
        
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_rc_override(throttle=throttle_pwm, steering=1500)
            time.sleep(0.1)
        
        # Dur
        for i in range(10):
            self.send_rc_override(throttle=1500, steering=1500)
            time.sleep(0.1)
    
    def turn_test(self, direction="left", duration=3):
        """Dönüş testi"""
        if direction == "left":
            steering_pwm = 1300  # Sol
        else:
            steering_pwm = 1700  # Sağ
        
        # Düşük hızla dönüş
        throttle_pwm = 1550  # %12.5 hız
        
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_rc_override(throttle=throttle_pwm, steering=steering_pwm)
            time.sleep(0.1)
        
        # Dur
        for i in range(10):
            self.send_rc_override(throttle=1500, steering=1500)
            time.sleep(0.1)
    
    def emergency_stop(self):
        """Acil durdurma"""
        print("!!! ACİL DURDURMA !!!")
        for i in range(20):  # 2 saniye boyunca dur sinyali
            self.send_rc_override(throttle=1500, steering=1500)
            time.sleep(0.1)
        self.disarm_vehicle()
    
    def arm_vehicle(self):
        """Aracı arm et"""
        print("Araç arm ediliyor...")
        self.master.mav.command_long_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )
        
        msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
        if msg and msg.result == 0:
            print("✓ Araç arm edildi!")
            return True
        else:
            print("✗ Araç arm edilemedi!")
            return False
    
    def disarm_vehicle(self):
        """Aracı disarm et"""
        self.master.mav.command_long_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        print("Araç disarm edildi.")
    
    def send_rc_override(self, throttle=1500, steering=1500):
        """RC override gönder"""
        self.master.mav.rc_channels_override_send(
            self.target_system, self.target_component,
            throttle, steering, 1500, 1500, 1500, 1500, 1500, 1500
        )
    
    def print_attitude(self):
        """Araç duruş bilgisi"""
        attitude = self.master.recv_match(type='ATTITUDE', blocking=False)
        if attitude:
            roll = attitude.roll * 57.2958  # radyan to derece
            pitch = attitude.pitch * 57.2958
            yaw = attitude.yaw * 57.2958
            print(f"Duruş - Roll: {roll:.1f}°, Pitch: {pitch:.1f}°, Yaw: {yaw:.1f}°")

def main():
    connection_string = "/dev/ttyUSB0"  # Kendi bağlantınızı ayarlayın
    
    try:
        usv_test = USVSafeTesting(connection_string)
        
        # Test öncesi kontroller
        if not usv_test.pre_flight_checks():
            print("Test öncesi kontroller başarısız! Test durduruldu.")
            return
        
        print("Test başlatılıyor... Her aşama için onay beklenecek.")
        print("Acil durdurma için Ctrl+C kullanın.\n")
        
        # Aşama 1: Sabit test
        input("AŞAMA 1 için Enter'a basın (Araç suya yerleştirildi mi?): ")
        if not usv_test.stage_1_stationary_test():
            print("Aşama 1 başarısız!")
            return
        
        # Aşama 2: Minimal hareket
        input("AŞAMA 2 için Enter'a basın (Minimal hareket testi): ")
        if not usv_test.stage_2_minimal_movement():
            print("Aşama 2 başarısız!")
            return
        
        # Aşama 3: Kontrollü hareket
        input("AŞAMA 3 için Enter'a basın (Kontrollü hareket testi): ")
        if not usv_test.stage_3_controlled_movement():
            print("Aşama 3 başarısız!")
            return
        
        # Aşama 4: Yönlendirme
        input("AŞAMA 4 için Enter'a basın (Yönlendirme testi): ")
        if not usv_test.stage_4_steering_test():
            print("Aşama 4 başarısız!")
            return
        
        print("\n🎉 TÜM TESTLER BAŞARILI!")
        print("USV temel hareketler için hazır.")
        
    except KeyboardInterrupt:
        print("\nTest kullanıcı tarafından durduruldu!")
        usv_test.emergency_stop()
    except Exception as e:
        print(f"Hata: {e}")
        usv_test.emergency_stop()

if __name__ == "__main__":
    main()