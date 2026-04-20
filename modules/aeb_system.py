import carla
from configs import aeb_config

class AEBSystem:
    def __init__(self):
        # Thông số từ config
        self.danger_zone_x_min = aeb_config.DANGER_ZONE_X_MIN
        self.danger_zone_x_max = aeb_config.DANGER_ZONE_X_MAX
        self.brake_threshold = aeb_config.BRAKE_HEIGHT_THRESHOLD
        
    def analyze_risk(self, boxes):
        """
        Đầu vào: Danh sách boxes từ Perception
        Đầu ra: (should_brake, risk_level, target_box)
        """
        should_brake = False
        risk_level = 0.0 # 0.0 -> 1.0
        target_box = None
        
        for box in boxes:
            # 1. Kiểm tra vật thể có nằm trong "Luồng đường" nguy hiểm (Trục X)
            center_x = (box['xmin'] + box['xmax']) / 2
            
            if self.danger_zone_x_min <= center_x <= self.danger_zone_x_max:
                # 2. Kiểm tra độ gần (dựa vào chiều cao Box)
                height = box['ymax'] - box['ymin']
                
                if height >= self.brake_threshold:
                    # Kích hoạt phanh nếu vật đủ gần
                    should_brake = True
                    # Tính độ nguy hiểm dựa trên độ lớn của box so với ngưỡng
                    risk_level = min(1.0, height / (self.brake_threshold * 2))
                    target_box = box
                    # Nếu có nhiều vật, ưu tiên vật gần nhất (box cao nhất)
                    break 
        
        return should_brake, risk_level, target_box
