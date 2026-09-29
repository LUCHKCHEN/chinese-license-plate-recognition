#!/usr/bin/env python
"""图片读写和车牌裁剪公共工具，统一处理中文路径。"""

import math
from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def read_image(path):
    """读取图片；失败时返回 None，不抛异常。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def save_image_jpg(path, image):
    """保存 JPG 图片，兼容中文路径。"""
    if image is None:
        raise ValueError('要保存的图片为空')
    ok, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError(f'图片编码失败: {path}')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer.tofile(str(path))


def crop_with_padding(image, x1, y1, x2, y2, pad_ratio=0.05):
    """按水平框裁剪，默认在框外扩 5% 边缘，并限制在图片范围内。"""
    if image is None:
        return None
    if pad_ratio < 0:
        raise ValueError(f'padding 比例不能为负数: {pad_ratio}')

    height, width = image.shape[:2]
    box_width = float(x2) - float(x1)
    box_height = float(y2) - float(y1)
    if box_width <= 0 or box_height <= 0:
        return None

    pad_x = box_width * pad_ratio
    pad_y = box_height * pad_ratio
    left = max(0, math.floor(float(x1) - pad_x))
    top = max(0, math.floor(float(y1) - pad_y))
    right = min(width, math.ceil(float(x2) + pad_x))
    bottom = min(height, math.ceil(float(y2) + pad_y))
    if right <= left or bottom <= top:
        return None
    return image[top:bottom, left:right]


def resize_to_fixed(image, width, height):
    """把图片统一缩放到固定尺寸。"""
    if image is None:
        raise ValueError('要缩放的图片为空')
    target_width = int(width)
    target_height = int(height)
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f'缩放尺寸必须大于 0: {width}x{height}')
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def draw_text_with_pillow(image, text, x, y, color=(0, 255, 0), font_size=20):
    """在图片上绘制中文文字，兼容 OpenCV 不支持的中文字符。"""
    if PIL_AVAILABLE:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        font = None
        for font_path in ('C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf'):
            try:
                font = ImageFont.truetype(font_path, int(font_size))
                break
            except OSError:
                continue
        if font is None:
            font = ImageFont.load_default()

        text_box = draw.textbbox((0, 0), text, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        draw.rectangle([x - 4, y - 4, x + text_width + 4, y + text_height + 4], fill=(0, 0, 0))
        draw.text((x, y), text, fill=tuple(reversed(color)), font=font)
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    safe_text = text.encode('ascii', 'ignore').decode('ascii')
    cv2.putText(image, safe_text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return image
