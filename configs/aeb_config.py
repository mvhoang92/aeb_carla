# ==========================================
# CẤU HÌNH THÔNG SỐ AEB PROJECT
# ==========================================

# 1. Thông số Camera (Cho MỖI chiếc Camera)
CAM_WIDTH = 800
CAM_HEIGHT = 600
CAM_FOV = 90
CAM_FPS = 20.0

# Vị trí gắn Camera kính lái (Dashcam)
CAM_X = 1.5 
CAM_Z = 1.4 

# 2. Thông số Môi trường CARLA
HOST = '127.0.0.1'
PORT = 2000
NUM_TRAFFIC_VEHICLES = 100  # Bác có thể tăng/giảm tùy cấu hình máy
NUM_PEDESTRIANS = 100      # Thêm người đi bộ

# 3. Thông số Nhận diện (YOLO)
YOLO_WEIGHTS = "weights/best.onnx"
CONFIDENCE_THRESHOLD = 0.5

# 4. Thông số Phanh (AEB)
DANGER_ZONE_X_MIN = CAM_WIDTH * 0.33
DANGER_ZONE_X_MAX = CAM_WIDTH * 0.66
BRAKE_HEIGHT_THRESHOLD = CAM_HEIGHT * 0.35

# 5. Thông số Màn hình Pygame (Dành cho Dual View)
# Bắt buộc nhân đôi chiều ngang để chứa 1 Cam Trái + 1 Cam Phải
WINDOW_WIDTH = CAM_WIDTH * 2  # Tương đương 1600
WINDOW_HEIGHT = CAM_HEIGHT    # Giữ nguyên 600