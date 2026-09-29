#!/usr/bin/env python
"""项目内公共路径、控制台输出和命令行参数校验工具。"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def configure_stdout():
    """把控制台输出统一为 UTF-8，避免 Windows 中文乱码。"""
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass


def resolve_path(base_dir, value):
    """把相对路径或绝对路径统一解析为绝对路径。"""
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def resolve_project_path(value):
    """按项目根目录解析命令行传入的路径。"""
    return resolve_path(PROJECT_ROOT, value)


def positive_int(value):
    """命令行参数校验：必须是大于 0 的整数。"""
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f'必须是整数: {value}') from exc
    if number <= 0:
        raise argparse.ArgumentTypeError(f'必须大于 0: {value}')
    return number


def non_negative_int(value):
    """命令行参数校验：必须是大于等于 0 的整数。"""
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f'必须是整数: {value}') from exc
    if number < 0:
        raise argparse.ArgumentTypeError(f'必须大于等于 0: {value}')
    return number


def positive_float(value):
    """命令行参数校验：必须是大于 0 的小数。"""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f'必须是小数: {value}') from exc
    if number <= 0:
        raise argparse.ArgumentTypeError(f'必须大于 0: {value}')
    return number


def non_negative_float(value):
    """命令行参数校验：必须是大于等于 0 的小数。"""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f'必须是小数: {value}') from exc
    if number < 0:
        raise argparse.ArgumentTypeError(f'必须大于等于 0: {value}')
    return number


def bounded_float(lower, upper):
    """生成闭区间小数校验器，用于置信度、裁剪比例等参数。"""
    def parser(value):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(f'必须是小数: {value}') from exc
        if number < lower or number > upper:
            raise argparse.ArgumentTypeError(f'必须在 {lower} 到 {upper} 之间: {value}')
        return number

    return parser
