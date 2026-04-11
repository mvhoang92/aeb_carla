import numpy as np

class AdaptiveSpeedController:
    """Điều chỉnh tốc độ dựa trên độ cong của đường"""
    
    def __init__(self, max_speed=1.0, min_speed=0.3):
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.prev_steering_angles = []
        self.window_size = 5  # Số frame để tính độ cong
    
    def calculate_curvature(self, steering_angle):
        """
        Tính độ cong của đường dựa trên steering angle
        Steering angle lớn = đường cong nhiều = giảm tốc độ
        """
        self.prev_steering_angles.append(abs(steering_angle))
        
        # Giữ chỉ window_size frame gần nhất
        if len(self.prev_steering_angles) > self.window_size:
            self.prev_steering_angles.pop(0)
        
        # Tính trung bình steering angle
        avg_steering = np.mean(self.prev_steering_angles)
        
        return avg_steering
    
    def calculate_adaptive_speed(self, steering_angle):
        """
        Tính tốc độ thích ứng
        
        Args:
            steering_angle: Góc lái hiện tại (-1.0 đến 1.0)
        
        Returns:
            throttle: Tốc độ (0.0 đến 1.0)
        """
        curvature = self.calculate_curvature(steering_angle)
        
        # Ánh xạ curvature (0 đến 1) sang tốc độ (max_speed đến min_speed)
        # Curvature cao = tốc độ thấp
        throttle = self.max_speed - (curvature * (self.max_speed - self.min_speed))
        
        # Giới hạn throttle
        throttle = np.clip(throttle, self.min_speed, self.max_speed)
        
        return throttle
    
    def reset(self):
        """Reset controller"""
        self.prev_steering_angles = []
