#!/usr/bin/env python
"""读取 CLPD.csv，生成 YOLO 检测数据集和 CRNN 字符识别数据集。"""

import argparse
import csv
import math
import random
import shutil
from pathlib import Path

from common import PROJECT_ROOT, configure_stdout, non_negative_float, positive_int, resolve_path
from plate_utils import crop_with_padding, read_image, resize_to_fixed, save_image_jpg


def parse_args():
    parser = argparse.ArgumentParser(description='把 CLPD.csv 转换成 YOLO 检测数据和 OCR 数据')
    parser.add_argument('--csv', default='CLPD.csv', help='CSV 文件路径，默认项目根目录下的 CLPD.csv')
    parser.add_argument('--image-root', default='.', help='图片相对 CSV 路径的根目录，默认项目根目录')
    parser.add_argument('--output', default='datasets', help='输出目录，默认项目根目录下的 datasets')
    parser.add_argument('--seed', type=int, default=42, help='随机划分种子')
    parser.add_argument('--ocr-width', type=positive_int, default=320, help='OCR 图片宽度')
    parser.add_argument('--ocr-height', type=positive_int, default=96, help='OCR 图片高度')
    parser.add_argument('--ocr-pad', type=non_negative_float, default=0.05, help='OCR 裁剪时额外扩大的边缘比例')
    return parser.parse_args()


def read_csv_rows(csv_path):
    """自动尝试常见编码，返回字段名和全部行。"""
    for encoding in ('utf-8-sig', 'gbk', 'utf-8'):
        try:
            with open(csv_path, 'r', encoding=encoding, newline='') as file:
                reader = csv.DictReader(file)
                fieldnames = reader.fieldnames
                rows = list(reader)
            return fieldnames, rows
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise SystemExit(f'无法读取 CSV 文件: {csv_path}')


def parse_box(row, width, height):
    """把四个角点转成水平外接矩形；无效时返回 None 和原因。"""
    try:
        points = [float(row[key]) for key in ('x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4')]
    except (KeyError, TypeError, ValueError):
        return None, '坐标缺失或不是数字'
    if not all(math.isfinite(value) for value in points):
        return None, '坐标包含无效数字'

    xs = [points[index] for index in range(0, 8, 2)]
    ys = [points[index] for index in range(1, 8, 2)]
    left = max(0.0, min(xs))
    top = max(0.0, min(ys))
    right = min(float(width), max(xs))
    bottom = min(float(height), max(ys))
    if right - left <= 0 or bottom - top <= 0:
        return None, '外接矩形为空或完全在图片外'
    return (left, top, right, bottom), None


def write_data_yaml(detection_root):
    data_yaml = detection_root / 'data.yaml'
    root_posix = detection_root.as_posix()
    data_yaml.write_text(
        f'path: "{root_posix}"\n'
        'train: images/train\n'
        'val: images/val\n'
        'test: images/test\n'
        'nc: 1\n'
        'names: [license_plate]\n',
        encoding='utf-8',
    )
    return data_yaml


def reset_directory(path):
    """删除上次生成的数据，避免旧图片混入新的训练集。"""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main():
    configure_stdout()
    args = parse_args()

    csv_path = resolve_path(PROJECT_ROOT, args.csv)
    image_root = resolve_path(PROJECT_ROOT, args.image_root)
    output_root = resolve_path(PROJECT_ROOT, args.output)
    detection_root = output_root / 'license_plate'
    ocr_root = output_root / 'ocr'

    if not csv_path.is_file():
        raise SystemExit(f'CSV 文件不存在: {csv_path}')

    fieldnames, rows = read_csv_rows(csv_path)
    required_columns = {'path', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'label'}
    missing_columns = required_columns - set(fieldnames or [])
    if missing_columns:
        raise SystemExit(f'CSV 缺少必要列: {sorted(missing_columns)}')

    records_by_image = {}
    skipped_rows = []
    for row_number, row in enumerate(rows, start=2):
        raw_path = (row.get('path') or '').strip()
        if not raw_path:
            skipped_rows.append((row_number, 'path 为空'))
            continue

        image_path = Path(raw_path)
        if not image_path.is_absolute():
            image_path = image_root / image_path
        image = read_image(image_path)
        if image is None:
            skipped_rows.append((row_number, f'图片读取失败: {raw_path}'))
            continue

        height, width = image.shape[:2]
        box, reason = parse_box(row, width, height)
        if box is None:
            skipped_rows.append((row_number, f'{reason}: {raw_path}'))
            continue

        label = (row.get('label') or '').replace('\t', '').replace('\n', '')
        image_key = str(image_path.resolve())
        image_info = records_by_image.setdefault(image_key, {'path': image_path.resolve(), 'records': []})
        image_info['records'].append({'box': box, 'label': label, 'width': width, 'height': height})

    if not records_by_image:
        raise SystemExit('没有解析到任何有效标注，请检查 CSV 和图片路径')

    image_paths = list(records_by_image.keys())
    random.Random(args.seed).shuffle(image_paths)
    train_count = int(len(image_paths) * 0.7)
    val_count = int(len(image_paths) * 0.2)
    split_paths = {
        'train': image_paths[:train_count],
        'val': image_paths[train_count:train_count + val_count],
        'test': image_paths[train_count + val_count:],
    }

    reset_directory(detection_root / 'images')
    reset_directory(detection_root / 'labels')
    for split in ('train', 'val', 'test'):
        (detection_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (detection_root / 'labels' / split).mkdir(parents=True, exist_ok=True)

    split_image_counts = {split: 0 for split in ('train', 'val', 'test')}
    for split, image_keys in split_paths.items():
        for image_key in image_keys:
            info = records_by_image[image_key]
            image_path = info['path']
            target_image = detection_root / 'images' / split / image_path.name
            shutil.copy2(image_path, target_image)

            label_lines = []
            for record in info['records']:
                left, top, right, bottom = record['box']
                box_width = right - left
                box_height = bottom - top
                center_x = left + box_width / 2.0
                center_y = top + box_height / 2.0
                image_width = record['width']
                image_height = record['height']
                label_lines.append(
                    f'0 {center_x / image_width:.6f} {center_y / image_height:.6f} '
                    f'{box_width / image_width:.6f} {box_height / image_height:.6f}'
                )

            if label_lines:
                # YOLO 标签使用归一化的中心点坐标和宽高
                label_file = detection_root / 'labels' / split / f'{image_path.stem}.txt'
                label_file.write_text('\n'.join(label_lines) + '\n', encoding='utf-8')
                split_image_counts[split] += 1


    data_yaml = write_data_yaml(detection_root)

    all_labels = [
        record['label']
        for info in records_by_image.values()
        for record in info['records']
        if record['label']
    ]
    if not all_labels:
        raise SystemExit('CSV 中没有可用于 OCR 的车牌字符串')

    characters = sorted(set(''.join(all_labels)))
    char_map_path = ocr_root / 'characters.txt'
    char_map_path.parent.mkdir(parents=True, exist_ok=True)
    char_map_path.write_text('\n'.join(characters) + '\n', encoding='utf-8')

    reset_directory(ocr_root / 'images')
    for split in ('train', 'val', 'test'):
        (ocr_root / 'images' / split).mkdir(parents=True, exist_ok=True)

    ocr_manifest_lines = {split: [] for split in ('train', 'val', 'test')}
    ocr_counts = {split: 0 for split in ('train', 'val', 'test')}
    for split, image_keys in split_paths.items():
        for image_key in image_keys:
            info = records_by_image[image_key]
            image = read_image(info['path'])
            if image is None:
                continue
            for record_index, record in enumerate(info['records']):
                if not record['label']:
                    continue
                crop = crop_with_padding(image, *record['box'], pad_ratio=args.ocr_pad)
                if crop is None:
                    continue
                crop = resize_to_fixed(crop, args.ocr_width, args.ocr_height)
                file_name = f'{info["path"].stem}_{record_index}.jpg'
                save_image_jpg(ocr_root / 'images' / split / file_name, crop)
                relative_image = f'images/{split}/{file_name}'
                ocr_manifest_lines[split].append(f'{relative_image}\t{record["label"]}')
                ocr_counts[split] += 1

    for split in ('train', 'val', 'test'):
        manifest_path = ocr_root / f'{split}.txt'
        manifest_path.write_text('\n'.join(ocr_manifest_lines[split]) + '\n', encoding='utf-8')

    skipped_path = PROJECT_ROOT / 'skipped_rows.log'
    with open(skipped_path, 'w', encoding='utf-8') as file:
        for row_number, reason in skipped_rows:
            file.write(f'{row_number}\t{reason}\n')

    print('检测数据划分:')
    for split in ('train', 'val', 'test'):
        print(f'  {split}: {split_image_counts[split]} 张图片')
    print('OCR 数据划分:')
    for split in ('train', 'val', 'test'):
        print(f'  {split}: {ocr_counts[split]} 个车牌样本')
    print(f'字符表 {len(characters)} 个字符 -> {char_map_path}')
    print(f'YOLO 配置: {data_yaml}')
    print(f'跳过行 {len(skipped_rows)} 条 -> {skipped_path}')


if __name__ == '__main__':
    main()
