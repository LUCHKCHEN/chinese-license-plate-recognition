#!/usr/bin/env python
"""训练 CRNN 车牌字符识别模型，并输出验证集和测试集整串准确率。"""

import argparse
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common import PROJECT_ROOT, configure_stdout, non_negative_int, positive_float, positive_int, resolve_path
from ocr_model import CRNN, CharMap, greedy_decode, load_ocr_checkpoint, save_ocr_checkpoint, to_torch_device
from plate_utils import read_image


def parse_args():
    parser = argparse.ArgumentParser(description='训练 CRNN 车牌字符识别模型')
    parser.add_argument('--data', default='datasets/ocr', help='OCR 数据目录')
    parser.add_argument('--characters', default='', help='字符表路径，默认取 OCR 数据目录下的 characters.txt')
    parser.add_argument('--width', type=positive_int, default=320, help='输入图片宽度')
    parser.add_argument('--height', type=positive_int, default=96, help='输入图片高度')
    parser.add_argument('--epochs', type=positive_int, default=100, help='训练轮数')
    parser.add_argument('--batch', type=positive_int, default=64, help='批次大小')
    parser.add_argument('--lr', type=positive_float, default=0.001, help='初始学习率')
    parser.add_argument('--workers', type=non_negative_int, default=4, help='数据加载进程数')
    parser.add_argument('--device', default='0', help='训练设备：0 表示第一块 GPU，cpu 表示 CPU')
    parser.add_argument('--project', default='runs/ocr', help='训练结果根目录')
    parser.add_argument('--name', default='best', help='本次训练名称')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    return parser.parse_args()


def check_device(device):
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('当前 PyTorch 无法使用 GPU，请先安装 CUDA 版 PyTorch，或改用 --device cpu')


def augment_image(image):
    """训练时对车牌图做轻微扰动，验证和测试不使用。"""
    image = image.astype(np.float32)
    image = np.clip(image * random.uniform(0.9, 1.1), 0.0, 255.0)
    mean = image.mean()
    image = (image - mean) * random.uniform(0.9, 1.1) + mean
    if random.random() < 0.5:
        noise = np.random.normal(0.0, random.uniform(0.0, 5.0), image.shape).astype(np.float32)
        image = image + noise
    image = np.clip(image, 0.0, 255.0).astype(np.uint8)

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, random.uniform(-2.0, 2.0), random.uniform(0.95, 1.05))
    return cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)


class PlateOcrDataset(Dataset):
    def __init__(self, manifest_path, image_root, char_map, width, height, augment=False):
        if not manifest_path.is_file():
            raise FileNotFoundError(f'OCR 清单不存在: {manifest_path}')
        self.char_map = char_map
        self.width = width
        self.height = height
        self.augment = augment
        self.records = []
        for line in manifest_path.read_text(encoding='utf-8').splitlines():
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            relative_image, label = parts[0], parts[1].strip()
            if not label:
                continue
            try:
                encoded = char_map.encode(label)
            except KeyError:
                continue
            image_path = image_root / relative_image
            if not image_path.is_file():
                continue
            self.records.append((image_path, encoded))
        if not self.records:
            raise SystemExit(f'清单中没有可用 OCR 样本: {manifest_path}')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_path, encoded = self.records[index]
        image = read_image(image_path)
        if image is None:
            raise FileNotFoundError(f'OCR 图片读取失败: {image_path}')
        if self.augment:
            image = augment_image(image)
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32) / 255.0)
        target = torch.tensor(encoded, dtype=torch.long)
        return tensor, target, target.shape[0]


def collate_ocr(batch):
    images, targets, lengths = zip(*batch)
    images = torch.stack(images)
    targets = torch.cat(targets)
    target_lengths = torch.tensor(lengths, dtype=torch.long)
    return images, targets, target_lengths


def evaluate(model, loader, char_map, device):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for images, targets, target_lengths in loader:
            images = images.to(device)
            targets = targets.to(device)
            log_probs = model(images)
            predictions.extend(greedy_decode(log_probs, char_map))
            start = 0
            for length in target_lengths.tolist():
                labels.append(char_map.decode(targets[start:start + length].tolist()))
                start += length
    if not labels:
        return 0.0
    correct = sum(pred == label for pred, label in zip(predictions, labels))
    return correct / len(labels)


def main():
    configure_stdout()
    args = parse_args()
    device = to_torch_device(args.device)
    check_device(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    data_root = resolve_path(PROJECT_ROOT, args.data)
    characters_path = data_root / 'characters.txt' if not args.characters else resolve_path(PROJECT_ROOT, args.characters)
    if not characters_path.is_file():
        raise SystemExit(f'字符表不存在: {characters_path}')

    char_map = CharMap.load(characters_path)
    train_dataset = PlateOcrDataset(
        data_root / 'train.txt', data_root, char_map, args.width, args.height, augment=True
    )
    val_dataset = PlateOcrDataset(
        data_root / 'val.txt', data_root, char_map, args.width, args.height, augment=False
    )
    test_dataset = PlateOcrDataset(
        data_root / 'test.txt', data_root, char_map, args.width, args.height, augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_ocr,
        persistent_workers=args.workers > 0,
        pin_memory=device.type == 'cuda',
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_ocr,
        persistent_workers=args.workers > 0,
        pin_memory=device.type == 'cuda',
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_ocr,
        persistent_workers=args.workers > 0,
        pin_memory=device.type == 'cuda',
    )

    model = CRNN(num_classes=char_map.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CTCLoss(blank=char_map.blank_index, zero_infinity=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.name == 'best':
        best_path = PROJECT_ROOT / args.project / 'best.pth'
    else:
        best_path = PROJECT_ROOT / args.project / args.name / 'best.pth'
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0

    print(f'训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}, 测试样本: {len(test_dataset)}')
    print(f'字符数量: {char_map.num_classes - 1}, 输出类别: {char_map.num_classes}, 设备: {device}')
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batch_count = 0
        for images, targets, target_lengths in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            log_probs = model(images)
            input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long, device=device)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())
            batch_count += 1
        average_loss = total_loss / max(batch_count, 1)
        val_accuracy = evaluate(model, val_loader, char_map, device)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_ocr_checkpoint(best_path, model, char_map, args.width, args.height)
        print(f'轮次 {epoch}/{args.epochs}, 损失 {average_loss:.4f}, 验证整串准确率 {val_accuracy:.4f}')
        scheduler.step()

    model, char_map, width, height = load_ocr_checkpoint(best_path, device)
    val_accuracy = evaluate(model, val_loader, char_map, device)
    test_accuracy = evaluate(model, test_loader, char_map, device)
    print(f'最佳验证整串准确率: {val_accuracy:.4f}')
    print(f'测试集整串准确率: {test_accuracy:.4f}')
    print(f'最佳 OCR 权重: {best_path}')


if __name__ == '__main__':
    main()
