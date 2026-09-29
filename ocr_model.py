#!/usr/bin/env python
"""轻量 CRNN 车牌字符识别模型：卷积提取特征，双向 LSTM 建模序列，CTC 解码。"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def to_torch_device(value):
    """把命令行里的 0/cpu 等写法统一成 PyTorch 设备对象。"""
    if isinstance(value, int):
        return torch.device(f'cuda:{value}')
    text = str(value).strip().lower()
    if text.isdigit():
        return torch.device(f'cuda:{text}')
    return torch.device(text)


class CharMap:
    """字符映射表：索引 0 固定为 CTC 空白位，字符从 1 开始。"""

    def __init__(self, characters):
        self.chars = list(dict.fromkeys(characters))
        if not self.chars:
            raise ValueError('字符表不能为空')
        self.blank_index = 0
        self.char_to_index = {char: i + 1 for i, char in enumerate(self.chars)}
        self.index_to_char = {i + 1: char for i, char in enumerate(self.chars)}

    @property
    def num_classes(self):
        return len(self.chars) + 1

    def encode(self, label):
        return [self.char_to_index[char] for char in label]

    def decode(self, indices):
        result = []
        for index in indices:
            index = int(index)
            if index != self.blank_index:
                result.append(self.index_to_char.get(index, ''))
        return ''.join(result)

    def save(self, path):
        Path(path).write_text('\n'.join(self.chars) + '\n', encoding='utf-8')

    @classmethod
    def load(cls, path):
        lines = Path(path).read_text(encoding='utf-8').splitlines()
        chars = [line for line in lines if line.strip() != '']
        return cls(chars)


class CRNN(nn.Module):
    """输入三通道固定尺寸图片，输出 CTC 对数概率。"""

    def __init__(self, num_classes, hidden_size=256, sequence_length=40):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, sequence_length)),
        )
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=False,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, images):
        features = self.cnn(images)
        features = features.squeeze(2)
        features = features.permute(2, 0, 1)
        sequence_out, _ = self.lstm(features)
        sequence_out = self.dropout(sequence_out)
        log_probs = self.fc(sequence_out)
        return F.log_softmax(log_probs, dim=-1)


def greedy_decode(log_probs, char_map):
    """CTC 贪心解码，输入形状为 (序列长度, 批量, 类别数)。"""
    preds = log_probs.argmax(dim=-1)
    if preds.dim() == 1:
        preds = preds.unsqueeze(1)
    preds = preds.cpu().numpy()

    texts = []
    sequence_length, batch_size = preds.shape
    for batch_index in range(batch_size):
        chars = []
        previous = char_map.blank_index
        for time_index in range(sequence_length):
            index = int(preds[time_index, batch_index])
            if index != char_map.blank_index and index != previous:
                chars.append(char_map.index_to_char.get(index, ''))
            previous = index
        texts.append(''.join(chars))
    return texts


@torch.no_grad()
def recognize_plate_image(model, char_map, image, width=320, height=96, device='cpu'):
    """识别单张裁剪后的车牌图片，返回车牌字符串。"""
    if image is None:
        raise ValueError('要识别的车牌图片为空')
    if int(width) <= 0 or int(height) <= 0:
        raise ValueError(f'识别尺寸必须大于 0: {width}x{height}')

    device = to_torch_device(device)
    resized = cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1).astype(np.float32) / 255.0)
    tensor = tensor.unsqueeze(0).to(device)
    log_probs = model(tensor)
    return greedy_decode(log_probs, char_map)[0]


def save_ocr_checkpoint(path, model, char_map, width=320, height=96):
    """保存 OCR 模型权重、字符表和输入尺寸。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'state_dict': model.state_dict(),
            'chars': char_map.chars,
            'width': width,
            'height': height,
        },
        str(path),
    )


def load_ocr_checkpoint(path, device='cpu'):
    """加载 OCR 权重，返回模型、字符表和输入尺寸。"""
    device = to_torch_device(device)
    path = Path(path)
    try:
        checkpoint = torch.load(str(path), map_location=device, weights_only=True)
        char_map = CharMap(checkpoint['chars'])
        model = CRNN(num_classes=char_map.num_classes)
        model.load_state_dict(checkpoint['state_dict'])
    except Exception as exc:
        raise ValueError(f'OCR 权重文件无法加载: {path}') from exc

    model.to(device)
    model.eval()
    width = int(checkpoint.get('width', 320))
    height = int(checkpoint.get('height', 96))
    return model, char_map, width, height
