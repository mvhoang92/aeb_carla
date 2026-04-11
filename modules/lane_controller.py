import numpy as np

class LaneController:
    """Điều khiển vô lăng để bám làn đường"""
    
    def __init__(self, image_width=800, max_steering_angle=1.0):
        self.image_width = image_width
        self.max_steering_angle = max_steering_angle
        self.center_x = image_width / 2
        
        # PID Controller parameters
        self.kp = 0.005  # Proportional gain
        self.ki = 0.0001  # Integral gain
        self.kd = 0.01   # Derivative gain
        
        self.integral_error = 0
        self.prev_error = 0
    
    def calculate_steering_angle(self, lane_center):
        """
        Tính steering angle dựa trên vị trí tâm làn
        
        Args:
            lane_center: Vị trí x của tâm làn (pixel)
        
        Returns:
            steering_angle: Góc lái (-1.0 đến 1.0)
        """
        if lane_center is None:
            return 0.0
        
        # Tính sai số: khoảng cách từ tâm ảnh đến tâm làn
        error = lane_center - self.center_x
        
        # PID Control
        self.integral_error += error
        derivative_error = error - self.prev_error
        
        # Tính steering angle
        steering_angle = (
            self.kp * error + 
            self.ki * self.integral_error + 
            self.kd * derivative_error
        )
        
        # Giới hạn steering angle
        steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)
        
        self.prev_error = error
        
        return steering_angle
    
    def reset(self):
        """Reset PID controller"""
        self.integral_error = 0
        self.prev_error = 0
