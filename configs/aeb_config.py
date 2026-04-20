# ==========================================
# CẤU HÌNH THÔNG SỐ AEB PROJECT (Tesla Model 3 Setup)
# ==========================================

# 1. Thông số Camera (Cho MỖI chiếc Camera)
# Thiết lập chuẩn 16:9 hoặc gần giống tỷ lệ tự nhiên để hiển thị chân thực
CAM_WIDTH = 1280
CAM_HEIGHT = 720

# Dựa theo thực tế Tesla Model 3 sử dụng cụm camera:
# - Camera góc hẹp (Narrow): ~25 độ (Cao tốc)
# - Camera chính (Main): ~50-60 độ (Khoảng cách vừa)
# - Camera góc rộng (Wide - Fisheye): ~120 độ (Phát hiện vật cắt ngang)
# Lựa chọn 90 độ cho ảnh 720p giúp cân bằng tốt nhất giữa góc nhìn và tỉ lệ hiển thị Box cho AI.
CAM_FOV = 90
CAM_FPS = 30.0

# Vị trí gắn Camera AEB (Tesla Model 3 - Sau gương chiếu hậu, trong cabin)
# x = 0.5m: ngay sau kính chắn gió
# z = 1.3m: cao ngang với gương chiếu hậu giữa xe
CAM_X = 0.5
CAM_Z = 1.3

# 2. Thông số Môi trường CARLA
HOST = '127.0.0.1'
PORT = 2000
NUM_TRAFFIC_VEHICLES = 100
NUM_PEDESTRIANS = 100

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 3. Thông số Nhận diện (YOLO)
YOLO_WEIGHTS = os.path.join(BASE_DIR, "weights/yolov8n.onnx")
CONFIDENCE_THRESHOLD = 0.4

# 4. Thông số Phanh (AEB)
# Vùng nguy hiểm theo trục X: Lấy phần trung tâm khoảng 40% chiều ngang (30% -> 70%)
DANGER_ZONE_X_MIN = int(CAM_WIDTH * 0.3)
DANGER_ZONE_X_MAX = int(CAM_WIDTH * 0.7)

# Chiều cao box tối thiểu (pixel) để kích hoạt phanh (Tỷ lệ với chiều cao ảnh)
BRAKE_HEIGHT_THRESHOLD = int(CAM_HEIGHT * (120/640))

# 5. Thông số Màn hình Pygame (Dành cho Dual View)
# Bắt buộc nhân đôi chiều ngang để chứa 1 Cam Trái + 1 Cam Phải
WINDOW_WIDTH = CAM_WIDTH * 2
WINDOW_HEIGHT = CAM_HEIGHT