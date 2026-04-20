import cv2
import numpy as np
import onnxruntime as ort
import os

# Mapping YOLO COCO class names -> 4 class của hệ thống AEB
# YOLOv8 COCO indices: 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
COCO_TO_AEB = {
    0: (0, 'person'),
    1: (1, 'bike_motorbike'),
    2: (2, 'car'),
    3: (1, 'bike_motorbike'),
    5: (3, 'truck'),
    7: (3, 'truck'),
}

class YoloDetector:
    """
    Inference YOLOv8 (TensorRT) sử dụng ONNX Runtime.
    Tối ưu hóa phần cứng ở mức cao nhất (FP16 + TensorRT).
    """
    INPUT_SIZE = 640

    def __init__(self, weights_path, conf_thresh=0.4):
        # Nếu truyền vào file .pt, ta đổi sang .onnx
        if weights_path.endswith('.pt'):
            weights_path = weights_path.replace('.pt', '.onnx')
            
        print(f"[*] Đang khởi tạo TensorRT Session cho: {weights_path}...")
        
        # Cấu hình Providers: Ưu tiên TensorRT, sau đó là CUDA, cuối cùng là CPU
        providers = [
            ('TensorrtExecutionProvider', {
                'device_id': 0,
                'trt_fp16_enable': True,       # Kích hoạt FP16 siêu tốc
                'trt_engine_cache_enable': True, # Lưu cache engine để khởi động nhanh lần sau
                'trt_engine_cache_path': os.path.dirname(weights_path),
            }),
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ]
        
        try:
            self.session = ort.InferenceSession(weights_path, providers=providers)
            active_providers = self.session.get_providers()
            print(f"[*] AEB Perception: Đã kích hoạt {active_providers[0]}")
        except Exception as e:
            print(f"[!] Lỗi khởi tạo TensorRT: {e}. Đang thử fallback...")
            self.session = ort.InferenceSession(weights_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        # Kiểm tra kiểu dữ liệu đầu vào mà model yêu cầu (float32 hay float16)
        self.input_info = self.session.get_inputs()[0]
        self.input_name = self.input_info.name
        self.input_type = self.input_info.type # 'tensor(float16)' hoặc 'tensor(float)'
        
        print(f"[*] Model yêu cầu kiểu dữ liệu: {self.input_type}")

        self.conf_thresh = conf_thresh
        self.colors = {
            0: (255,  50,  50),   # Đỏ
            1: ( 50, 255,  50),   # Xanh lá
            2: ( 50, 150, 255),   # Xanh dương
            3: (255, 165,   0),   # Cam
        }

    def detect(self, bgr_image):
        """
        bgr_image : numpy BGR 1280x720 (hoặc bất kỳ resolution nào)
        Trả về    : list[dict] chuẩn AEB
        """
        img_h, img_w = bgr_image.shape[:2]

        # 1. Pre-processing: Resize -> RGB -> CHW -> Normalize -> Batch
        img_resized = cv2.resize(bgr_image, (self.INPUT_SIZE, self.INPUT_SIZE))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Chuyển đổi kiểu dữ liệu dựa trên yêu cầu của Model (Quan trọng để tránh crash)
        if 'float16' in self.input_type:
            img_input = img_rgb.transpose(2, 0, 1).astype(np.float16) / 255.0
        else:
            img_input = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
            
        img_input = np.expand_dims(img_input, axis=0)

        # 2. Inference
        outputs = self.session.run(None, {self.input_name: img_input})
        preds = outputs[0][0].T  # (8400, 84)

        # 3. Post-processing (Vectorized NumPy)
        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]
        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # Filter by confidence
        mask = max_scores > self.conf_thresh
        if not np.any(mask):
            return []

        filtered_boxes = boxes_xywh[mask]
        filtered_scores = max_scores[mask]
        filtered_class_ids = class_ids[mask]

        scale_x = img_w / self.INPUT_SIZE
        scale_y = img_h / self.INPUT_SIZE

        results = []
        rects = []
        confs = []

        for i in range(len(filtered_boxes)):
            coco_cls = filtered_class_ids[i]
            if coco_cls not in COCO_TO_AEB:
                continue

            cx, cy, w, h = filtered_boxes[i]
            x1 = int((cx - w / 2) * scale_x)
            y1 = int((cy - h / 2) * scale_y)
            x2 = int((cx + w / 2) * scale_x)
            y2 = int((cy + h / 2) * scale_y)

            aeb_cls, aeb_name = COCO_TO_AEB[coco_cls]
            
            results.append({
                'xmin': max(0, x1), 'ymin': max(0, y1),
                'xmax': min(img_w, x2), 'ymax': min(img_h, y2),
                'conf': float(filtered_scores[i]),
                'class_id': aeb_cls, 'name': aeb_name
            })
            # For NMS
            rects.append([max(0, x1), max(0, y1), x2-x1, y2-y1])
            confs.append(float(filtered_scores[i]))

        # 4. NMS
        if results:
            indices = cv2.dnn.NMSBoxes(rects, confs, self.conf_thresh, 0.45)
            if len(indices) > 0:
                final_results = [results[i] for i in indices.flatten()]
                return final_results

        return []