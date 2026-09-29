#!/usr/bin/env python
"""训练 YOLOv8 单类车牌检测模型，并评估验证集和测试集。"""

import argparse
import os

from common import PROJECT_ROOT, configure_stdout, non_negative_int, positive_float, positive_int, resolve_path

os.environ.setdefault('YOLO_CONFIG_DIR', str(PROJECT_ROOT / '.ultralytics'))

import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='训练 YOLOv8 车牌检测模型')
    parser.add_argument('--model', default='yolov8n.pt', help='预训练模型，可切换 yolov8s.pt')
    parser.add_argument('--data', default='datasets/license_plate/data.yaml', help='data.yaml 路径')
    parser.add_argument('--epochs', type=positive_int, default=100, help='训练轮数')
    parser.add_argument('--batch', type=positive_int, default=16, help='批次大小')
    parser.add_argument('--imgsz', type=positive_int, default=640, help='训练图片尺寸')
    parser.add_argument('--lr0', type=positive_float, default=0.01, help='初始学习率')
    parser.add_argument('--device', default='0', help='训练设备：0 表示第一块 GPU，cpu 表示 CPU')
    parser.add_argument('--project', default='runs/detect', help='训练结果根目录')
    parser.add_argument('--name', default='license_plate', help='本次训练名称')
    parser.add_argument('--optimizer', default='SGD', help='优化器，默认 SGD，可切换 AdamW 或 auto')
    parser.add_argument('--workers', type=non_negative_int, default=4, help='数据加载进程数')
    parser.add_argument('--resume', action='store_true', help='从同名训练目录的 last.pt 继续训练')
    return parser.parse_args()


def check_device(device):
    if str(device).lower() != 'cpu' and not torch.cuda.is_available():
        raise SystemExit('当前 PyTorch 无法使用 GPU，请先安装 CUDA 版 PyTorch，或改用 --device cpu')


def print_metrics(metrics, split_name):
    print(f'{split_name} mAP50: {metrics.box.map50:.4f}, mAP50-95: {metrics.box.map:.4f}')


def main():
    configure_stdout()
    args = parse_args()
    check_device(args.device)

    data_yaml = resolve_path(PROJECT_ROOT, args.data)
    if not data_yaml.is_file():
        raise SystemExit(f'data.yaml 不存在: {data_yaml}')

    project_root = resolve_path(PROJECT_ROOT, args.project)
    if args.resume:
        last_path = project_root / args.name / 'weights' / 'last.pt'
        if not last_path.is_file():
            raise SystemExit(f'找不到可续训的 last.pt: {last_path}')
        model = YOLO(str(last_path))
        model.train(resume=True, exist_ok=True)
    else:
        model = YOLO(args.model)
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            lr0=args.lr0,
            optimizer=args.optimizer,
            device=args.device,
            workers=args.workers,
            project=str(project_root),
            name=args.name,
            exist_ok=True,
        )

    best_path = project_root / args.name / 'weights' / 'best.pt'
    if not best_path.is_file():
        raise SystemExit(f'训练完成但未找到 best.pt: {best_path}')

    best_model = YOLO(str(best_path))
    val_metrics = best_model.val(data=str(data_yaml), split='val', device=args.device, batch=args.batch, imgsz=args.imgsz)
    print_metrics(val_metrics, '验证集')
    test_metrics = best_model.val(data=str(data_yaml), split='test', device=args.device, batch=args.batch, imgsz=args.imgsz)
    print_metrics(test_metrics, '测试集')
    print(f'最佳检测权重: {best_path}')


if __name__ == '__main__':
    main()
