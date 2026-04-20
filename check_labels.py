#!/usr/bin/env python3
"""
Ve YOLO annotation len anh va luu vao thu muc check_labels/ de kiem tra nhanh.
Mac dinh doc dataset tu aeb/data.
"""
import os
import glob
import random
import argparse
from pathlib import Path

import cv2

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = CURRENT_DIR / 'data'
DEFAULT_OUTPUT_DIR = CURRENT_DIR / 'check_labels'

CLASS_INFO = {
    0: ('person',    (50,  50,  255)),   # Đỏ
    1: ('bike_motorbike', (50, 255, 50)), # Xanh lá
    2: ('car',       (255, 150, 50)),    # Xanh dương
    3: ('truck',     (0,   165, 255)),   # Cam
}


def draw_and_save(img_path, lbl_path, out_path):
    img = cv2.imread(img_path)
    if img is None:
        return

    h, w = img.shape[:2]
    n_boxes = 0

    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f.read().strip().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)

                name, color = CLASS_INFO.get(cls_id, (f'cls{cls_id}', (255, 255, 255)))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                label = f"{name}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - lh - 4), (x1 + lw + 4, y1), color, -1)
                cv2.putText(img, label, (x1 + 2, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                n_boxes += 1

    # Ghi số box lên góc trái
    cv2.putText(img, f"{n_boxes} objects", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main(args):
    img_dir = os.path.join(args.dataset, 'images')
    lbl_dir = os.path.join(args.dataset, 'labels')
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    all_imgs = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    if not all_imgs:
        print(f"[!] Không tìm thấy ảnh trong {img_dir}")
        return

    if args.random:
        random.shuffle(all_imgs)
    all_imgs = all_imgs[:args.num]

    print(f"[*] Đang render {len(all_imgs)} ảnh → {out_dir}/")
    for i, img_path in enumerate(all_imgs):
        stem     = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, stem + '.txt')
        out_path = os.path.join(out_dir, stem + '.jpg')
        draw_and_save(img_path, lbl_path, out_path)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(all_imgs)}")

    print(f"[*] Xong! Mở thư mục {out_dir}/ để kiểm tra.")
    print(f"    Màu: ĐỎ=person  XANH LÁ=bike_motorbike  XANH DƯƠNG=car  CAM=truck")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=str(DEFAULT_DATASET_DIR))
    parser.add_argument('--output',  default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--num',     default=500, type=int, help='Số ảnh muốn render')
    parser.add_argument('--random',  action='store_true',   help='Chọn ngẫu nhiên')
    args = parser.parse_args()
    main(args)
