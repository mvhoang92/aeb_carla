class LaneDepartureWarning:
    """Cảnh báo khi xe sắp rời khỏi làn đường"""
    
    def __init__(self, image_width=800, warning_threshold=100):
        """
        Args:
            image_width: Chiều rộng ảnh
            warning_threshold: Khoảng cách từ tâm ảnh để cảnh báo (pixel)
        """
        self.image_width = image_width
        self.center_x = image_width / 2
        self.warning_threshold = warning_threshold
        self.warning_active = False
        self.warning_side = None  # 'left' hoặc 'right'
    
    def check_departure(self, lane_center, left_lines, right_lines):
        """
        Kiểm tra xem xe có sắp rời khỏi làn không
        
        Returns:
            (warning_active, warning_side, distance_to_edge)
        """
        if lane_center is None:
            return False, None, None
        
        # Tính khoảng cách từ tâm ảnh đến tâm làn
        error = lane_center - self.center_x
        
        # Kiểm tra cảnh báo
        self.warning_active = abs(error) > self.warning_threshold
        
        if self.warning_active:
            self.warning_side = 'left' if error < 0 else 'right'
        else:
            self.warning_side = None
        
        return self.warning_active, self.warning_side, abs(error)
    
    def get_warning_message(self):
        """Lấy thông báo cảnh báo"""
        if not self.warning_active:
            return "Lane OK"
        
        if self.warning_side == 'left':
            return "WARNING: Drifting LEFT!"
        else:
            return "WARNING: Drifting RIGHT!"
