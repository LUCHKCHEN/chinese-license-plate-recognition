#!/usr/bin/env python
"""车牌识别交互模式：循环输入图片路径，输出检测框和车牌字符。"""

import argparse
import os

from common import (
    PROJECT_ROOT,
    bounded_float,
    non_negative_float,
    configure_stdout,
    positive_int,
    resolve_project_path,
)

os.environ.setdefault('YOLO_CONFIG_DIR', str(PROJECT_ROOT / '.ultralytics'))

import cv2
from ultralytics import YOLO

from ocr_model import load_ocr_checkpoint, recognize_plate_image
from plate_utils import crop_with_padding, draw_text_with_pillow, read_image, save_image_jpg

EXIT_COMMANDS = {'q', 'exit', 'quit'}


def parse_args():
    parser = argparse.ArgumentParser(description='车牌识别交互模式')
    parser.add_argument('--weights', default='runs/detect/license_plate/weights/best.pt', help='YOLO 检测权重')
    parser.add_argument('--ocr-weights', default='runs/ocr/best.pth', help='OCR 权重')
    parser.add_argument('--conf', type=bounded_float(0.0, 1.0), default=0.25, help='检测置信度阈值')
    parser.add_argument('--imgsz', type=positive_int, default=640, help='检测图片尺寸')
    parser.add_argument('--device', default='0', help='推理设备')
    parser.add_argument('--pad', type=non_negative_float, default=0.05, help='OCR 裁剪边缘比例')
    parser.add_argument('--output', default='outputs/predictions', help='结果保存目录')
    return parser.parse_args()


def load_models(args):
    """加载 YOLO 检测模型和 CRNN 识别模型，两个模型只加载一次。"""
    weights_path = resolve_project_path(args.weights)
    ocr_weights_path = resolve_project_path(args.ocr_weights)

    if not weights_path.is_file():
        raise SystemExit(f'检测权重不存在: {weights_path}')
    if not ocr_weights_path.is_file():
        raise SystemExit(f'OCR 权重不存在: {ocr_weights_path}')

    try:
        detection_model = YOLO(str(weights_path))
        ocr_model, char_map, ocr_width, ocr_height = load_ocr_checkpoint(ocr_weights_path, args.device)
    except Exception as exc:
        raise SystemExit(f'模型加载失败: {exc}') from exc

    return detection_model, ocr_model, char_map, ocr_width, ocr_height


def predict_image(args, source_path, detection_model, ocr_model, char_map, ocr_width, ocr_height, output_dir):
    """识别单张图片：检测车牌位置，再裁剪车牌并识别字符。"""
    if not source_path.is_file():
        raise FileNotFoundError(f'输入图片不存在: {source_path}')

    image = read_image(source_path)
    if image is None:
        raise ValueError(f'图片读取失败: {source_path}')

    # 第一步：用 YOLO 检测车牌位置
    results = detection_model.predict(
        source=image,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )
    boxes = results[0].boxes
    annotated = image.copy()
    detection_count = 0 if boxes is None else len(boxes)
    print(f'检测到 {detection_count} 个车牌')

    if boxes is not None and detection_count > 0:
        for index, box in enumerate(boxes):
            # 第二步：裁剪车牌区域，再用 CRNN 识别字符
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0]]
            confidence = float(box.conf[0])
            crop = crop_with_padding(image, x1, y1, x2, y2, pad_ratio=args.pad)
            text = ''
            if crop is not None:
                try:
                    text = recognize_plate_image(
                        ocr_model, char_map, crop, width=ocr_width, height=ocr_height, device=args.device
                    )
                except Exception as exc:
                    print(f'  检测 {index + 1} 的字符识别失败，已忽略该字符: {exc}')
                    text = ''

            left, top = int(x1), int(y1)
            right, bottom = int(x2), int(y2)
            cv2.rectangle(annotated, (left, top), (right, bottom), (0, 255, 0), 2)
            label = f'license_plate {confidence:.2f} {text}'
            annotated = draw_text_with_pillow(annotated, label, left, max(0, top - 32), font_size=20)
            print(f'  检测 {index + 1}: 坐标=({left},{top},{right},{bottom}), 置信度={confidence:.4f}, 车牌={text}')

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{source_path.stem}_result.jpg'
    save_image_jpg(output_path, annotated)
    print(f'结果已保存: {output_path}')


def run_interactive(args, output_dir, detection_model, ocr_model, char_map, ocr_width, ocr_height):
    """交互模式：循环等待用户输入图片路径，输入 q 或 exit 退出。"""
    print('交互模式：输入图片路径开始识别，输入 q 或 exit 退出')
    while True:
        try:
            raw_input = input('图片路径: ').strip().strip('"').strip("'")
        except (EOFError, KeyboardInterrupt):
            print('\n已退出交互模式')
            break

        if raw_input.lower() in EXIT_COMMANDS:
            print('已退出交互模式')
            break
        if not raw_input:
            continue

        source_path = resolve_project_path(raw_input)
        try:
            predict_image(
                args,
                source_path,
                detection_model,
                ocr_model,
                char_map,
                ocr_width,
                ocr_height,
                output_dir,
            )
        except Exception as exc:
            print(f'识别失败: {exc}')


def main():
    configure_stdout()
    args = parse_args()
    output_dir = resolve_project_path(args.output)
    detection_model, ocr_model, char_map, ocr_width, ocr_height = load_models(args)
    run_interactive(args, output_dir, detection_model, ocr_model, char_map, ocr_width, ocr_height)


if __name__ == '__main__':
    main()
