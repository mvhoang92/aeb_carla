import cv2
import numpy as np

class KalmanFilterLane:
    """Kalman Filter để làm mịn vị trí làn"""
    def __init__(self):
        self.kf = cv2.KalmanFilter(1, 1)
        self.kf.measurementMatrix = np.array([[1.0]], dtype=np.float32)
        self.kf.transitionMatrix = np.array([[1.0]], dtype=np.float32)
        self.kf.processNoiseCov = np.array([[0.01]], dtype=np.float32)
        self.kf.measurementNoiseCov = np.array([[4.0]], dtype=np.float32)
        self.kf.statePost = np.array([[400.0]], dtype=np.float32)

    def update(self, measurement):
        """Cập nhật Kalman Filter"""
        self.kf.correct(np.array([[measurement]], dtype=np.float32))
        prediction = self.kf.predict()
        return prediction[0][0]

class LaneDetector:
    """Phát hiện vạch làn đường bằng OpenCV với Kalman Filter"""

    def __init__(self, height=600, width=800):
        self.height = height
        self.width = width

        # ROI (Region of Interest) - chỉ xử lý phần dưới của ảnh
        self.roi_top = int(height * 0.5)  # Bắt đầu từ 50% chiều cao
        self.roi_bottom = height

        # Kalman Filter để làm mịn vị trí làn
        self.kalman = KalmanFilterLane()
        self.prev_lane_center = width / 2
        
    def preprocess(self, image):
        """Tiền xử lý ảnh: Grayscale, Blur, Threshold"""
        # Chuyển sang grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Blur để giảm nhiễu
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold để tách vạch trắng (vạch trắng có giá trị cao)
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        
        return thresh
    
    def detect_edges(self, image):
        """Phát hiện cạnh bằng Canny"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return edges
    
    def detect_lanes(self, image):
        """Phát hiện vạch làn bằng Hough Line Transform"""
        # Tiền xử lý
        edges = self.detect_edges(image)
        
        # Chỉ xử lý ROI (phần dưới của ảnh)
        roi = edges[self.roi_top:self.roi_bottom, :]
        
        # Hough Line Transform
        lines = cv2.HoughLinesP(
            roi, 
            rho=1, 
            theta=np.pi/180, 
            threshold=50,
            minLineLength=50,
            maxLineGap=10
        )
        
        return lines, roi
    
    def filter_lanes(self, lines):
        """Lọc các đường thẳng để tìm vạch làn (dọc hoặc gần dọc)"""
        if lines is None:
            return None, None
        
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Tính độ dốc
            if x2 - x1 == 0:
                slope = float('inf')
            else:
                slope = (y2 - y1) / (x2 - x1)
            
            # Vạch trái: độ dốc âm (từ trên phải xuống dưới trái)
            if slope < -0.5:
                left_lines.append(line)
            # Vạch phải: độ dốc dương (từ trên trái xuống dưới phải)
            elif slope > 0.5:
                right_lines.append(line)
        
        return left_lines, right_lines
    
    def get_lane_center(self, image):
        """Tính tâm làn đường với Kalman Filter"""
        lines, roi = self.detect_lanes(image)

        if lines is None:
            return self.prev_lane_center, None, None

        left_lines, right_lines = self.filter_lanes(lines)

        # Tính trung bình vị trí vạch trái và phải
        left_x = None
        right_x = None

        if left_lines:
            left_x = np.mean([line[0][0] for line in left_lines])

        if right_lines:
            right_x = np.mean([line[0][0] for line in right_lines])

        # Tính tâm làn
        lane_center = None
        if left_x is not None and right_x is not None:
            lane_center = (left_x + right_x) / 2
        elif left_x is not None:
            lane_center = left_x + 100  # Giả định độ rộng làn
        elif right_x is not None:
            lane_center = right_x - 100

        # Áp dụng Kalman Filter để làm mịn
        if lane_center is not None:
            lane_center = self.kalman.update(lane_center)
            self.prev_lane_center = lane_center
        else:
            lane_center = self.prev_lane_center

        return lane_center, left_lines, right_lines
    
    def draw_lanes(self, image, left_lines, right_lines):
        """Vẽ vạch làn lên ảnh"""
        output = image.copy()
        
        if left_lines is not None:
            for line in left_lines:
                x1, y1, x2, y2 = line[0]
                y1 += self.roi_top
                y2 += self.roi_top
                cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if right_lines is not None:
            for line in right_lines:
                x1, y1, x2, y2 = line[0]
                y1 += self.roi_top
                y2 += self.roi_top
                cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return output
