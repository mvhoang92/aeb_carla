from ultralytics import YOLO

class YoloDetector:
    def __init__(self, weights_path, conf_thresh=0.5):
        print(f"[*] Đang tải mô hình AI từ: {weights_path}...")
        self.model = YOLO(weights_path)
        self.conf_thresh = conf_thresh
        
        # Ánh xạ ID Class theo đúng model của bác
        self.classes = {0: 'person', 1: 'bike_motorbike', 2: 'vehicle'}
        
        # Bảng màu cho Pygame (R, G, B)
        self.colors = {
            0: (255, 50, 50),   # Đỏ cho Người
            1: (50, 255, 50),   # Xanh lá cho Xe máy/Đạp
            2: (50, 150, 255)   # Xanh dương cho Ô tô
        }
        print("[*] Tải AI hoàn tất!")

    def detect(self, image_array):
        # image_array là mảng RGB lấy từ Camera CARLA
        # Tắt verbose=False để terminal không bị spam text liên tục
        results = self.model.predict(image_array, conf=self.conf_thresh, verbose=False)
        
        boxes = []
        for r in results:
            for box in r.boxes:
                # Trích xuất tọa độ
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                boxes.append({
                    'xmin': int(x1), 'ymin': int(y1),
                    'xmax': int(x2), 'ymax': int(y2),
                    'conf': conf, 
                    'class_id': cls_id,
                    'name': self.classes.get(cls_id, 'unknown')
                })
        return boxes