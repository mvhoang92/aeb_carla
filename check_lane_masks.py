import argparse
import glob
import os
import random
from pathlib import Path

import cv2
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = CURRENT_DIR / 'lane_data'
DEFAULT_OUTPUT_DIR = DEFAULT_DATASET_DIR / 'overlays'


def overlay_lane_mask(image, mask):
    overlay = image.copy()
    lane_region = mask > 0
    if np.any(lane_region):
        lane_color = np.zeros_like(image)
        lane_color[:, :] = (0, 220, 170)
        blended = cv2.addWeighted(image, 0.6, lane_color, 0.4, 0.0)
        overlay[lane_region] = blended[lane_region]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 220, 0), 2)
    return overlay


def mask_preview(mask):
    preview = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    lane_region = mask > 0
    if np.any(lane_region):
        preview[lane_region] = (0, 220, 170)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(preview, contours, -1, (255, 220, 0), 2)
    return preview


def compose_panel(image, overlay, mask):
    return np.hstack([image, overlay, mask_preview(mask)])


def main(args):
    images_dir = Path(args.dataset) / 'images'
    masks_dir = Path(args.dataset) / 'masks'
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(glob.glob(str(images_dir / '*.jpg')))
    if not image_paths:
        print(f'[!] Khong tim thay anh trong {images_dir}')
        return

    if args.random:
        random.shuffle(image_paths)
    if args.num > 0:
        image_paths = image_paths[:args.num]

    print(f'[*] Dang render {len(image_paths)} anh lane -> {output_dir}')
    for index, image_path in enumerate(image_paths, start=1):
        stem = Path(image_path).stem
        mask_path = masks_dir / f'{stem}.png'
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue

        overlay = overlay_lane_mask(image, mask)
        panel = compose_panel(image, overlay, mask)
        cv2.putText(
            panel,
            f'lane pixels: {int(np.count_nonzero(mask))}',
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(output_dir / f'{stem}.jpg'), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])

        if index % 50 == 0:
            print(f'  {index}/{len(image_paths)}')

    print(f'[*] Xong. Mo {output_dir} de kiem tra mask lane.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render visible lane labels from saved lane masks')
    parser.add_argument('--dataset', default=str(DEFAULT_DATASET_DIR))
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--num', default=0, type=int, help='0 = render all images')
    parser.add_argument('--random', action='store_true')
    main(parser.parse_args())
