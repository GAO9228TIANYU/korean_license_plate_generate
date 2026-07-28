#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Korean License Plate Generator — KR_LPR format output.
/home/ubuntu/Desktop/GTY/projects/kakao_project/korean-license-plate-generator
/home/ubuntu/mydata/GTY/korean_license_generator_10w
Output structure:
  result/
    labels.names
    chunk_00/
      12가1234_000000.jpg
      12가1234_000000.txt   # 每行一个字符 bbox: x1 y1 x2 y2
    chunk_01/
      ...

Usage:
  python generate_all.py --total 300000 --chunk-size 10000 --workers 64
  python generate_all.py --total 10 --workers 1          # 快速测试
"""

import os
import sys
import random
import argparse
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
#  Romanized filename → Korean character mapping
# ═══════════════════════════════════════════════════════════════════
ROMAN_TO_KR = {
    # 가나다라마
    "ga": "가",
    "na": "나",
    "da": "다",
    "ra": "라",
    "ma": "마",
    # 바사아자하
    "ba": "바",
    "sa": "사",
    "a": "아",
    "ja": "자",
    "ha": "하",
    # 거너더러머버서어저허
    "geo": "거",
    "neo": "너",
    "deo": "더",
    "reo": "러",
    "meo": "머",
    "beo": "버",
    "seo": "서",
    "eo": "어",
    "jeo": "저",
    "heo": "허",
    # 고노도로모보소오조호
    "go": "고",
    "no": "노",
    "do": "도",
    "ro": "로",
    "mo": "모",
    "bo": "보",
    "so": "소",
    "o": "오",
    "jo": "조",
    "ho": "호",
    # 구누두루무부수우주
    "gu": "구",
    "nu": "누",
    "du": "두",
    "ru": "루",
    "mu": "무",
    "bu": "부",
    "su": "수",
    "u": "우",
    "ju": "주",
    # 배
    "bae": "배",
    "yuk": "육",
}

# region1[i] + region2[i] → Korean city name (按文件编号 001-017 排列)
REGION_KR = {
    1: "부산",
    2: "충북",
    3: "충남",
    4: "대구",
    5: "대전",
    6: "강원",
    7: "경북",
    8: "경기",
    9: "경남",
    10: "인천",
    11: "제주",
    12: "전북",
    13: "전남",
    14: "광주",
    15: "세종",
    16: "서울",
    17: "울산",
}

# 数字文件名就是 "0"-"9"，直接用
NUM_TO_KR = {str(i): str(i) for i in range(10)}


# ═══════════════════════════════════════════════════════════════════
#  Asset loader
# ═══════════════════════════════════════════════════════════════════
class Assets:
    """加载一次所有素材图片和对应的韩文名称。"""

    def __init__(self, base_dir="./assets"):
        self.nums = self._load_dir(os.path.join(base_dir, "nums"), NUM_TO_KR)
        self.chars = self._load_dir(os.path.join(base_dir, "chars"), ROMAN_TO_KR)
        self.chars_truck = self._load_dir(
            os.path.join(base_dir, "chars_truck"), ROMAN_TO_KR
        )
        self.region1, self.region2, self.region_names = self._load_regions(base_dir)
        self.plates = {}
        for t in ("type_a", "type_b", "type_c", "type_d", "type_e", "type_f"):
            d = os.path.join(base_dir, "plates", t)
            self.plates = {}
            for t in (
                "type_a",
                "type_b",
                "type_c",
                "type_d",
                "type_e",
                "type_f",
                "type_g",
                "type_h",
            ):
                d = os.path.join(base_dir, "plates", t)
                self.plates[t] = []
                for f in sorted(os.listdir(d)):
                    if not f.lower().endswith(
                        (".jpg", ".jpeg", ".png", ".bmp", ".webp")
                    ):
                        continue
                    img = cv2.imread(os.path.join(d, f))
                    if img is None:
                        raise FileNotFoundError(
                            f"Failed to read plate template: {os.path.join(d, f)}"
                        )
                    self.plates[t].append(img)

                if not self.plates[t]:
                    raise FileNotFoundError(f"No plate templates found in: {d}")

    @staticmethod
    def _load_dir(path, name_map):
        """返回 [(image, korean_name), ...] 按文件名排序。"""
        result = []
        for fname in sorted(os.listdir(path)):
            stem = os.path.splitext(fname)[0]
            img = cv2.imread(os.path.join(path, fname))
            kr = name_map.get(stem, stem)
            result.append((img, kr))
        return result

    @staticmethod
    def _load_regions(base_dir):
        """加载 region1 和 region2，按编号排序并构建 (img1, img2, kr_name) 三元组。"""
        r1_dir = os.path.join(base_dir, "region1")
        r2_dir = os.path.join(base_dir, "region2")
        r1_files = sorted(os.listdir(r1_dir))
        r2_files = sorted(os.listdir(r2_dir))
        r1_imgs, r2_imgs, names = [], [], []
        for f1, f2 in zip(r1_files, r2_files):
            idx = int(f1[:3])
            r1_imgs.append(cv2.imread(os.path.join(r1_dir, f1)))
            r2_imgs.append(cv2.imread(os.path.join(r2_dir, f2)))
            names.append(REGION_KR[idx])
        return r1_imgs, r2_imgs, names


# ═══════════════════════════════════════════════════════════════════
#  noise functions
# ═══════════════════════════════════════════════════════════════════


def random_crop_margin(img, bboxes):
    """Simulate imperfect detector ROI.

    Some samples lose a small border; some samples include extra surrounding
    margin. Bboxes are transformed to stay consistent with the output image.
    """
    if random.random() > 0.45:
        return img, bboxes

    h, w = img.shape[:2]

    # Negative means crop inside the plate; positive means add surrounding margin.
    left = random.randint(-int(w * 0.035), int(w * 0.04))
    right = random.randint(-int(w * 0.035), int(w * 0.04))
    top = random.randint(-int(h * 0.06), int(h * 0.06))
    bottom = random.randint(-int(h * 0.06), int(h * 0.06))

    x1 = max(0, -left)
    y1 = max(0, -top)
    x2 = min(w, w + right)
    y2 = min(h, h + bottom)

    if x2 <= x1 + 8 or y2 <= y1 + 8:
        return img, bboxes

    cropped = img[y1:y2, x1:x2]

    pad_left = max(0, left)
    pad_top = max(0, top)
    pad_right = max(0, right)
    pad_bottom = max(0, bottom)

    if pad_left or pad_top or pad_right or pad_bottom:
        # Border replicate is closer to a detector ROI containing nearby plate/car pixels
        # than a constant artificial color.
        cropped = cv2.copyMakeBorder(
            cropped,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REPLICATE,
        )

    out = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    crop_w = cropped.shape[1]
    crop_h = cropped.shape[0]
    sx = w / crop_w
    sy = h / crop_h

    new_boxes = []
    for bx1, by1, bx2, by2 in bboxes:
        nx1 = (bx1 - x1 + pad_left) * sx
        ny1 = (by1 - y1 + pad_top) * sy
        nx2 = (bx2 - x1 + pad_left) * sx
        ny2 = (by2 - y1 + pad_top) * sy
        new_boxes.append(clip_bbox(nx1, ny1, nx2, ny2, w, h))

    return out, new_boxes


def random_resolution(img):
    """Simulate low-resolution plate ROI resized back to model input size."""
    if random.random() > 0.55:
        return img

    h, w = img.shape[:2]

    scale = random.choice(
        [
            random.uniform(0.35, 0.50),
            random.uniform(0.50, 0.75),
            random.uniform(0.75, 0.90),
        ]
    )

    small_w = max(32, int(w * scale))
    small_h = max(16, int(h * scale))

    down_interp = random.choice([cv2.INTER_AREA, cv2.INTER_LINEAR])
    up_interp = random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST])

    small = cv2.resize(img, (small_w, small_h), interpolation=down_interp)
    out = cv2.resize(small, (w, h), interpolation=up_interp)

    return out


def random_sharpen_or_sensor(img):
    """Optional camera-like sharpening after low-resolution upsampling."""
    if random.random() > 0.25:
        return img

    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=random.uniform(0.6, 1.2))
    amount = random.uniform(0.4, 1.0)
    out = cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def clip_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(0, min(w - 1, int(round(x2))))
    y2 = max(0, min(h - 1, int(round(y2))))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def crop_pad_resize(
    img,
    bboxes,
    crop_left=0,
    crop_right=0,
    crop_top=0,
    crop_bottom=0,
    pad_left=0,
    pad_right=0,
    pad_top=0,
    pad_bottom=0,
):
    """Crop/pad ROI then resize back, while keeping char bboxes aligned."""
    h, w = img.shape[:2]

    x1 = max(0, int(crop_left))
    y1 = max(0, int(crop_top))
    x2 = min(w, w - int(crop_right))
    y2 = min(h, h - int(crop_bottom))

    if x2 <= x1 + 8 or y2 <= y1 + 8:
        return img, bboxes

    cropped = img[y1:y2, x1:x2]

    if pad_left or pad_right or pad_top or pad_bottom:
        cropped = cv2.copyMakeBorder(
            cropped,
            int(pad_top),
            int(pad_bottom),
            int(pad_left),
            int(pad_right),
            borderType=cv2.BORDER_REPLICATE,
        )

    crop_h, crop_w = cropped.shape[:2]
    out = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    sx = w / crop_w
    sy = h / crop_h

    new_boxes = []
    for bx1, by1, bx2, by2 in bboxes:
        nx1 = (bx1 - x1 + pad_left) * sx
        ny1 = (by1 - y1 + pad_top) * sy
        nx2 = (bx2 - x1 + pad_left) * sx
        ny2 = (by2 - y1 + pad_top) * sy
        new_boxes.append(clip_bbox(nx1, ny1, nx2, ny2, w, h))

    return out, new_boxes


def random_8digit_left_edge_aug(img, bboxes):
    """For 8-char wide plates: expose model to tight/shifted left edge crops."""
    if random.random() > 0.75:
        return img, bboxes

    h, w = img.shape[:2]

    if random.random() < 0.80:
        return crop_pad_resize(
            img,
            bboxes,
            crop_left=random.randint(int(w * 0.035), int(w * 0.105)),
            crop_right=random.randint(0, int(w * 0.025)),
            crop_top=random.randint(0, int(h * 0.015)),
            crop_bottom=random.randint(0, int(h * 0.015)),
        )

    return crop_pad_resize(
        img,
        bboxes,
        pad_left=random.randint(int(w * 0.015), int(w * 0.055)),
        crop_right=random.randint(0, int(w * 0.035)),
    )


def random_special_plate_appearance(img, plate_type):
    """Extra realism for colored/region plates without changing labels or bboxes."""
    if plate_type not in {"C", "D", "G", "H", "I"}:
        return img

    out = img.astype(np.float32)
    h, w = out.shape[:2]

    if plate_type in {"G", "H", "I"} and random.random() < 0.85:
        # Subtle uneven paint/reflective surface.
        gx = np.linspace(
            random.uniform(-12, 8), random.uniform(8, 18), w, dtype=np.float32
        )
        gy = np.linspace(
            random.uniform(-8, 6), random.uniform(6, 14), h, dtype=np.float32
        )
        grad = gx.reshape(1, w, 1) + gy.reshape(h, 1, 1)
        out += grad

        # Fine sensor/paint speckles.
        if random.random() < 0.55:
            speckle = np.random.normal(0, random.uniform(1.5, 4.5), out.shape).astype(
                np.float32
            )
            out += speckle

    if plate_type in {"C", "D", "G", "H"} and random.random() < 0.45:
        # Region text is often the hard part; simulate slight blur/low contrast.
        out = cv2.GaussianBlur(
            np.clip(out, 0, 255).astype(np.uint8), (3, 3), random.uniform(0.2, 0.6)
        ).astype(np.float32)

    return np.clip(out, 0, 255).astype(np.uint8)


def transform_bboxes(bboxes, M, out_w, out_h):
    new_boxes = []
    for x1, y1, x2, y2 in bboxes:
        pts = np.array(
            [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ],
            dtype=np.float32,
        )

        if M.shape == (2, 3):
            pts_h = np.concatenate([pts, np.ones((4, 1), dtype=np.float32)], axis=1)
            warped = pts_h @ M.T
        else:
            pts_h = np.concatenate([pts, np.ones((4, 1), dtype=np.float32)], axis=1)
            warped_h = pts_h @ M.T
            warped = warped_h[:, :2] / np.maximum(warped_h[:, 2:3], 1e-6)

        nx1, ny1 = warped.min(axis=0)
        nx2, ny2 = warped.max(axis=0)
        new_boxes.append(clip_bbox(nx1, ny1, nx2, ny2, out_w, out_h))

    return new_boxes


def random_photometric(img):
    img = img.astype(np.float32)

    # brightness / contrast
    alpha = random.uniform(0.65, 1.35)
    beta = random.uniform(-35, 35)
    img = img * alpha + beta

    # color temperature / channel shift
    if random.random() < 0.6:
        gains = np.array(
            [
                random.uniform(0.85, 1.15),
                random.uniform(0.85, 1.15),
                random.uniform(0.85, 1.15),
            ],
            dtype=np.float32,
        )
        img = img * gains.reshape(1, 1, 3)

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def random_noise(img):
    if random.random() < 0.6:
        sigma = random.uniform(2, 12)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if random.random() < 0.15:
        amount = random.uniform(0.001, 0.006)
        mask = np.random.rand(*img.shape[:2])
        img[mask < amount] = 0
        img[mask > 1 - amount] = 255

    return img


def random_motion_blur(img):
    if random.random() > 0.25:
        return img

    k = random.choice([3, 5, 7])
    kernel = np.zeros((k, k), dtype=np.float32)

    if random.random() < 0.5:
        kernel[k // 2, :] = 1.0
    else:
        kernel[:, k // 2] = 1.0

    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def random_jpeg(img):
    if random.random() > 0.45:
        return img

    quality = random.randint(45, 92)
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec if dec is not None else img


def random_glare(img):
    if random.random() > 0.35:
        return img

    h, w = img.shape[:2]
    overlay = img.copy().astype(np.float32)

    cx = random.randint(0, w - 1)
    cy = random.randint(0, h - 1)
    rx = random.randint(max(10, w // 8), max(20, w // 3))
    ry = random.randint(max(8, h // 8), max(16, h // 2))

    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (cx, cy), (rx, ry), random.uniform(-20, 20), 0, 360, 1, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=random.uniform(8, 20))

    strength = random.uniform(35, 95)
    overlay += mask[..., None] * strength

    return np.clip(overlay, 0, 255).astype(np.uint8)


def random_affine_or_perspective(img, bboxes):
    h, w = img.shape[:2]

    # Mostly keep geometry normal; only some samples are tilted.
    if random.random() > 0.45:
        return img, bboxes

    if random.random() < 0.7:
        angle = random.uniform(-5.0, 5.0)
        scale = random.uniform(0.96, 1.04)
        tx = random.uniform(-w * 0.015, w * 0.015)
        ty = random.uniform(-h * 0.025, h * 0.025)

        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty

        out = cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return out, transform_bboxes(bboxes, M, w, h)

    margin_x = w * random.uniform(0.01, 0.05)
    margin_y = h * random.uniform(0.02, 0.08)

    src = np.float32(
        [
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ]
    )
    dst = np.float32(
        [
            [random.uniform(0, margin_x), random.uniform(0, margin_y)],
            [w - 1 - random.uniform(0, margin_x), random.uniform(0, margin_y)],
            [w - 1 - random.uniform(0, margin_x), h - 1 - random.uniform(0, margin_y)],
            [random.uniform(0, margin_x), h - 1 - random.uniform(0, margin_y)],
        ]
    )

    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return out, transform_bboxes(bboxes, M, w, h)


def random_print_artifact(img):
    # Slight edge roughness / print variation.
    if random.random() < 0.35:
        if random.random() < 0.5:
            img = cv2.GaussianBlur(img, (3, 3), random.uniform(0.2, 0.7))
        else:
            kernel = np.ones((2, 2), np.uint8)
            if random.random() < 0.5:
                img = cv2.erode(img, kernel, iterations=1)
            else:
                img = cv2.dilate(img, kernel, iterations=1)
    return img


def augment_plate(img, bboxes, plate_type=None):
    if plate_type == "E":
        img, bboxes = random_8digit_left_edge_aug(img, bboxes)

    if plate_type in {"G", "H", "I", "C", "D"} and random.random() < 0.35:
        img, bboxes = random_crop_margin(img, bboxes)

    img, bboxes = random_affine_or_perspective(img, bboxes)
    img, bboxes = random_crop_margin(img, bboxes)

    img = random_resolution(img)
    img = random_sharpen_or_sensor(img)

    img = random_special_plate_appearance(img, plate_type)

    img = random_print_artifact(img)
    img = random_photometric(img)
    img = random_glare(img)
    img = random_noise(img)
    img = random_motion_blur(img)
    img = random_jpeg(img)
    return img, bboxes


# ═══════════════════════════════════════════════════════════════════
#  Helper functions
# ═══════════════════════════════════════════════════════════════════
def make_plate_base(w, h, color_bgr, border_bgr=(80, 80, 80), radius=10):
    """Create a blank rounded license plate base.

    This is used for synthetic yellow/green plate types where we do not have
    static blank background images. The old generator uses plate background
    images; this function creates an equivalent simple base programmatically.
    """
    plate = np.full((h, w, 3), 180, dtype=np.uint8)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (radius, 0), (w - radius, h), 255, -1)
    cv2.rectangle(mask, (0, radius), (w, h - radius), 255, -1)

    for cx in (radius, w - radius):
        for cy in (radius, h - radius):
            cv2.circle(mask, (cx, cy), radius, 255, -1)

    plate[mask > 0] = color_bgr

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(plate, contours, -1, border_bgr, thickness=3)

    # Bolt holes, similar to real plates.
    hole_r = max(3, int(min(w, h) * 0.025))
    hole_x = max(18, int(w * 0.06))
    for cx in (hole_x, w - hole_x):
        cv2.circle(plate, (cx, h // 2), hole_r, (70, 70, 70), -1)
        cv2.circle(plate, (cx, h // 2), hole_r, (35, 35, 35), 1)

    return plate


def composite(plate, char_img, x, y, w, h, text_color=None):
    """Resize char_img to (w,h), paste it on plate, return bbox.

    text_color is BGR. When set, the non-background glyph area is recolored.
    This is useful for green plates, whose real glyphs are white.
    """
    char_resized = cv2.resize(char_img, (w, h))
    char_bright = random_bright(char_resized)

    patch = plate[y : y + h, x : x + w]

    gray = cv2.cvtColor(char_bright, cv2.COLOR_BGR2GRAY)

    # The source glyph images are usually dark text on bright background.
    # Existing code uses mask_inv as glyph foreground, so keep that behavior.
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    if text_color is not None:
        fg_color = np.zeros_like(char_bright)
        fg_color[:, :] = np.array(text_color, dtype=np.uint8)
        fg = cv2.bitwise_and(fg_color, fg_color, mask=mask_inv)
    else:
        fg = cv2.bitwise_and(char_bright, char_bright, mask=mask_inv)

    bg = cv2.bitwise_and(patch, patch, mask=mask)
    plate[y : y + h, x : x + w] = cv2.add(bg, fg)

    return (x, y, x + w, y + h)


def random_bright(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = np.array(img, dtype=np.float64)
    factor = 0.5 + np.random.uniform()
    img[:, :, 2] = np.clip(img[:, :, 2] * factor, 0, 255)
    img = np.array(img, dtype=np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


# ═══════════════════════════════════════════════════════════════════
#  Per-type generators: 返回 (image, plate_text, [(x1,y1,x2,y2), ...])
# ═══════════════════════════════════════════════════════════════════


def gen_type_a(assets):
    """Type A — 乘用车 1行牌 520×110: NN韩NNNN"""
    plate_bg = random.choice(assets.plates["type_a"]).copy()
    plate = cv2.resize(plate_bg, (520, 110))

    seq = []  # [(korean_str, (x1,y1,x2,y2)), ...]

    # digit 1
    idx = random.randint(0, 9)
    bbox = composite(plate, assets.nums[idx][0], 35, 13, 56, 83)
    seq.append((assets.nums[idx][1], bbox))

    # digit 2
    idx = random.randint(0, 9)
    bbox = composite(plate, assets.nums[idx][0], 91, 13, 56, 83)
    seq.append((assets.nums[idx][1], bbox))

    # Korean char
    idx = random.randint(0, len(assets.chars) - 1)
    bbox = composite(plate, assets.chars[idx][0], 147, 13, 60, 83)
    seq.append((assets.chars[idx][1], bbox))

    # digits 3-6 (4-digit number)
    col = 243
    for _ in range(4):
        idx = random.randint(0, 9)
        bbox = composite(plate, assets.nums[idx][0], col, 13, 56, 83)
        seq.append((assets.nums[idx][1], bbox))
        col += 56

    text = "".join(s for s, _ in seq)
    bboxes = [b for _, b in seq]
    return plate, text, bboxes


def gen_type_b(assets):
    """Type B — 乘用车 2行牌 355×155: NN韩NNNN"""
    plate_bg = random.choice(assets.plates["type_b"]).copy()
    plate = cv2.resize(plate_bg, (355, 155))

    seq = []

    # digit 1
    idx = random.randint(0, 9)
    bbox = composite(plate, assets.nums[idx][0], 15, 45, 45, 83)
    seq.append((assets.nums[idx][1], bbox))

    # digit 2
    idx = random.randint(0, 9)
    bbox = composite(plate, assets.nums[idx][0], 60, 45, 45, 83)
    seq.append((assets.nums[idx][1], bbox))

    # Korean char (原代码偏移了 +2, +12)
    idx = random.randint(0, len(assets.chars) - 1)
    bbox = composite(plate, assets.chars[idx][0], 107, 57, 49, 70)
    seq.append((assets.chars[idx][1], bbox))

    # digits 3-6
    col = 158
    for _ in range(4):
        idx = random.randint(0, 9)
        bbox = composite(plate, assets.nums[idx][0], col, 45, 45, 83)
        seq.append((assets.nums[idx][1], bbox))
        col += 45

    text = "".join(s for s, _ in seq)
    bboxes = [b for _, b in seq]
    return plate, text, bboxes


def gen_type_c(assets):
    """Type C — 货车 2行牌 335×170: 地区NN韩NNNN"""
    plate_bg = random.choice(assets.plates["type_c"]).copy()
    plate = cv2.resize(plate_bg, (335, 170))

    seq = []

    # ── 上排：地区 + 2位数字 ──
    ri = random.randint(0, len(assets.region_names) - 1)

    # region1 贴图
    r1_img = cv2.resize(assets.region1[ri], (44, 60))
    r1_bright = random_bright(r1_img)
    gray = cv2.cvtColor(r1_bright, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    patch = plate[8:68, 76:120]
    bg = cv2.bitwise_and(patch, patch, mask=mask)
    fg = cv2.bitwise_and(r1_bright, r1_bright, mask=mask_inv)
    plate[8:68, 76:120] = cv2.add(bg, fg)

    # region2 贴图
    r2_img = cv2.resize(assets.region2[ri], (44, 60))
    r2_bright = random_bright(r2_img)
    gray = cv2.cvtColor(r2_bright, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    patch = plate[8:68, 120:164]
    bg = cv2.bitwise_and(patch, patch, mask=mask)
    fg = cv2.bitwise_and(r2_bright, r2_bright, mask=mask_inv)
    plate[8:68, 120:164] = cv2.add(bg, fg)

    # 合并地区 bbox
    seq.append((assets.region_names[ri], (76, 8, 164, 68)))

    # 上排 digit 1 (只取 8 或 9)
    idx = random.randint(8, 9)
    bbox = composite(plate, assets.nums[idx][0], 172, 8, 44, 60)
    seq.append((assets.nums[idx][1], bbox))

    # 上排 digit 2
    idx = random.randint(0, 9)
    bbox = composite(plate, assets.nums[idx][0], 216, 8, 44, 60)
    seq.append((assets.nums[idx][1], bbox))

    # ── 下排：韩文 + 4位数字 ──
    ci = random.randint(0, len(assets.chars_truck) - 1)
    bbox = composite(plate, assets.chars_truck[ci][0], 8, 72, 64, 62)
    seq.append((assets.chars_truck[ci][1], bbox))

    col = 72
    for _ in range(4):
        idx = random.randint(0, 9)
        bbox = composite(plate, assets.nums[idx][0], col, 72, 64, 90)
        seq.append((assets.nums[idx][1], bbox))
        col += 64

    text = "".join(s for s, _ in seq)
    bboxes = [b for _, b in seq]
    return plate, text, bboxes


def gen_type_d(assets):
    """Type D — 货车 1行牌 520×110: 地区NN韩NNNN"""
    plate_bg = random.choice(assets.plates["type_d"]).copy()
    plate = cv2.resize(plate_bg, (520, 110))

    seq = []

    # ── 地区 (上下两行叠放在左侧) ──
    ri = random.randint(0, len(assets.region_names) - 1)

    # region1 上半
    r1_img = cv2.resize(assets.region1[ri], (60, 42))
    r1_bright = random_bright(r1_img)
    gray = cv2.cvtColor(r1_bright, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    patch = plate[13:55, 25:85]
    bg = cv2.bitwise_and(patch, patch, mask=mask)
    fg = cv2.bitwise_and(r1_bright, r1_bright, mask=mask_inv)
    plate[13:55, 25:85] = cv2.add(bg, fg)

    # region2 下半
    r2_img = cv2.resize(assets.region2[ri], (60, 42))
    r2_bright = random_bright(r2_img)
    gray = cv2.cvtColor(r2_bright, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    patch = plate[55:97, 25:85]
    bg = cv2.bitwise_and(patch, patch, mask=mask)
    fg = cv2.bitwise_and(r2_bright, r2_bright, mask=mask_inv)
    plate[55:97, 25:85] = cv2.add(bg, fg)

    # 合并地区 bbox
    seq.append((assets.region_names[ri], (25, 13, 85, 97)))

    # digit 1 (只取 8 或 9)
    idx = random.randint(8, 9)
    bbox = composite(plate, assets.nums[idx][0], 85, 13, 56, 83)
    seq.append((assets.nums[idx][1], bbox))

    # digit 2
    idx = random.randint(0, 9)
    bbox = composite(plate, assets.nums[idx][0], 141, 13, 56, 83)
    seq.append((assets.nums[idx][1], bbox))

    # Korean char
    ci = random.randint(0, len(assets.chars_truck) - 1)
    bbox = composite(plate, assets.chars_truck[ci][0], 197, 13, 60, 83)
    seq.append((assets.chars_truck[ci][1], bbox))

    # digits 3-6
    col = 273
    for _ in range(4):
        idx = random.randint(0, 9)
        bbox = composite(plate, assets.nums[idx][0], col, 13, 56, 83)
        seq.append((assets.nums[idx][1], bbox))
        col += 56

    text = "".join(s for s, _ in seq)
    bboxes = [b for _, b in seq]
    return plate, text, bboxes


def gen_type_e(assets):
    """Type E — 新版 KOR 蓝带防伪车牌 520×110: NNN韩NNNN (8位)"""
    plate_bg = random.choice(assets.plates["type_e"]).copy()
    plate = cv2.resize(plate_bg, (520, 110))
    seq = []

    y, char_h = 25, 60
    num_w, kor_w = 44, 48
    inter, gap = 2, 20

    col = 85
    d1 = random.randint(1, 4)
    bbox = composite(plate, assets.nums[d1][0], col, y, num_w, char_h)
    seq.append((assets.nums[d1][1], bbox))
    col += num_w + inter

    for _ in range(2):
        idx = random.randint(0, 9)
        bbox = composite(plate, assets.nums[idx][0], col, y, num_w, char_h)
        seq.append((assets.nums[idx][1], bbox))
        col += num_w + inter

    ci = random.randint(0, len(assets.chars) - 1)
    bbox = composite(plate, assets.chars[ci][0], col, y, kor_w, char_h)
    seq.append((assets.chars[ci][1], bbox))
    col += kor_w + gap

    for _ in range(4):
        idx = random.randint(0, 9)
        bbox = composite(plate, assets.nums[idx][0], col, y, num_w, char_h)
        seq.append((assets.nums[idx][1], bbox))
        col += num_w + inter

    text = "".join(s for s, _ in seq)
    return plate, text, [b for _, b in seq]


def gen_type_f(assets):
    """Type F — EV 电动车蓝色车牌 520×110: NN韩NNNN (7位)"""
    plate_bg = random.choice(assets.plates["type_f"]).copy()
    plate = cv2.resize(plate_bg, (520, 110))
    seq = []

    y, char_h = 23, 64
    num_w, kor_w = 48, 52
    inter, gap = 2, 24

    col = 80
    for _ in range(2):
        idx = random.randint(0, 9)
        bbox = composite(plate, assets.nums[idx][0], col, y, num_w, char_h)
        seq.append((assets.nums[idx][1], bbox))
        col += num_w + inter

    ci = random.randint(0, len(assets.chars) - 1)
    bbox = composite(plate, assets.chars[ci][0], col, y, kor_w, char_h)
    seq.append((assets.chars[ci][1], bbox))
    col += kor_w + gap

    for _ in range(4):
        idx = random.randint(0, 9)
        bbox = composite(plate, assets.nums[idx][0], col, y, num_w, char_h)
        seq.append((assets.nums[idx][1], bbox))
        col += num_w + inter

    text = "".join(s for s, _ in seq)
    return plate, text, [b for _, b in seq]


def gen_type_g(assets):
    """Type G — 老旧绿色双排车牌 335x170: NN韩NNNN.

    Uses real green plate templates from assets/plates/type_g instead of a
    programmatically drawn base. White glyphs are pasted with safe margins so
    characters do not touch the border.
    """
    plate_bg = random.choice(assets.plates["type_g"]).copy()
    plate = cv2.resize(plate_bg, (335, 170))

    seq = []

    # Real old green plates use slightly gray-white glyphs, not pure white.
    white = random.choice(
        [
            (225, 225, 225),
            (235, 235, 235),
            (245, 245, 245),
        ]
    )

    # Top row: NN韩
    y_top = 18
    digit_w = 48
    digit_h = 58
    kor_w = 54
    kor_h = 58

    # Centered top group, safely away from screws/border.
    col = 74
    for _ in range(2):
        idx = random.randint(0, 9)
        bbox = composite(
            plate,
            assets.nums[idx][0],
            col,
            y_top,
            digit_w,
            digit_h,
            text_color=white,
        )
        seq.append((assets.nums[idx][1], bbox))
        col += digit_w + 3

    ci = random.randint(0, len(assets.chars) - 1)
    bbox = composite(
        plate,
        assets.chars[ci][0],
        col,
        y_top,
        kor_w,
        kor_h,
        text_color=white,
    )
    seq.append((assets.chars[ci][1], bbox))

    # Bottom row: NNNN
    y_bottom = 82
    bottom_w = 55
    bottom_h = 72
    col = 54

    for _ in range(4):
        idx = random.randint(0, 9)
        bbox = composite(
            plate,
            assets.nums[idx][0],
            col,
            y_bottom,
            bottom_w,
            bottom_h,
            text_color=white,
        )
        seq.append((assets.nums[idx][1], bbox))
        col += bottom_w + 4

    text = "".join(s for s, _ in seq)
    return plate, text, [b for _, b in seq]


def gen_type_h(assets):
    """Type H — 新式绿色长条牌 520x110: NNN韩NNNN.

    Uses real green long-plate templates from assets/plates/type_h. The layout
    keeps a left reserved area for the blue/green mark and keeps all glyphs
    inside the safe central text band.
    """
    plate_bg = random.choice(assets.plates["type_h"]).copy()
    plate = cv2.resize(plate_bg, (520, 110))

    seq = []

    # New green long plates normally use dark glyphs. Keep the default glyph
    # foreground from source assets, but use a tighter safe layout than EV plates.
    y = 22
    char_h = 66
    num_w = 43
    kor_w = 47
    inter = 3
    gap = 15

    # Leave safe space for the left mark/blue strip and avoid right border.
    col = 98

    # First three digits: NNN
    first = random.randint(1, 9)
    bbox = composite(plate, assets.nums[first][0], col, y, num_w, char_h)
    seq.append((assets.nums[first][1], bbox))
    col += num_w + inter

    for _ in range(2):
        idx = random.randint(0, 9)
        bbox = composite(plate, assets.nums[idx][0], col, y, num_w, char_h)
        seq.append((assets.nums[idx][1], bbox))
        col += num_w + inter

    # Korean char
    ci = random.randint(0, len(assets.chars) - 1)
    bbox = composite(plate, assets.chars[ci][0], col, y, kor_w, char_h)
    seq.append((assets.chars[ci][1], bbox))
    col += kor_w + gap

    # Last four digits
    for _ in range(4):
        idx = random.randint(0, 9)
        bbox = composite(plate, assets.nums[idx][0], col, y, num_w, char_h)
        seq.append((assets.nums[idx][1], bbox))
        col += num_w + inter

    text = "".join(s for s, _ in seq)
    return plate, text, [b for _, b in seq]


def gen_type_i(assets):
    """Type I — 绿色 2行牌 335x170: NN韩NNNN

    Real green Korean plates use white glyphs. We keep the same glyph assets
    but recolor the pasted foreground to white through composite(text_color=...).
    """
    plate = make_plate_base(
        335,
        170,
        color_bgr=(70, 130, 45),
        border_bgr=(35, 80, 30),
        radius=10,
    )

    seq = []
    white = (245, 245, 245)  # BGR

    y_top = 14
    digit_w = 54
    digit_h = 62
    kor_w = 58
    kor_h = 62

    col = 68
    for _ in range(2):
        idx = random.randint(0, 9)
        bbox = composite(
            plate,
            assets.nums[idx][0],
            col,
            y_top,
            digit_w,
            digit_h,
            text_color=white,
        )
        seq.append((assets.nums[idx][1], bbox))
        col += digit_w

    ci = random.randint(0, len(assets.chars) - 1)
    bbox = composite(
        plate,
        assets.chars[ci][0],
        col,
        y_top,
        kor_w,
        kor_h,
        text_color=white,
    )
    seq.append((assets.chars[ci][1], bbox))

    y_bottom = 76
    col = 42
    bottom_w = 64
    bottom_h = 82

    for _ in range(4):
        idx = random.randint(0, 9)
        bbox = composite(
            plate,
            assets.nums[idx][0],
            col,
            y_bottom,
            bottom_w,
            bottom_h,
            text_color=white,
        )
        seq.append((assets.nums[idx][1], bbox))
        col += bottom_w

    text = "".join(s for s, _ in seq)
    return plate, text, [b for _, b in seq]


# ═══════════════════════════════════════════════════════════════════
#  Worker
# ═══════════════════════════════════════════════════════════════════
GENERATORS = {
    "A": gen_type_a,  # 普通纯白横排车牌
    "B": gen_type_b,  # 老旧双排纯白车牌
    "C": gen_type_c,  # 老旧双行黄色车牌
    "D": gen_type_d,  # 带地点的单排/横向地区牌
    "E": gen_type_e,  # 左侧有蓝色标志的纯白新款车牌
    "F": gen_type_f,  # EV车牌
    "G": gen_type_g,  # 老旧绿色车牌
    "H": gen_type_h,  # 新式绿色长条牌
}

# 增加生成权重
TYPE_WEIGHTS = {
    "A": 3,
    "B": 2,
    "C": 0.4,
    "D": 2,
    "E": 6,
    "F": 2,
    "G": 3,
    "H": 2,
}


def worker(args):
    """每个进程独立加载素材并生成一批图片。"""
    start_idx, count, out_dir, asset_dir, chunk_size, seed = args

    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)

    assets = Assets(asset_dir)

    type_keys = list(TYPE_WEIGHTS.keys())
    type_wts = list(TYPE_WEIGHTS.values())

    for i in range(count):
        global_idx = start_idx + i
        chunk_id = global_idx // chunk_size
        chunk_dir = os.path.join(out_dir, f"chunk_{chunk_id:02d}")
        os.makedirs(chunk_dir, exist_ok=True)

        plate_type = random.choices(type_keys, weights=type_wts, k=1)[0]
        gen_fn = GENERATORS[plate_type]

        img, text, bboxes = gen_fn(assets)
        img, bboxes = augment_plate(img, bboxes, plate_type=plate_type)
        fname = f"{text}_{global_idx:06d}"
        img_path = os.path.join(chunk_dir, fname + ".jpg")
        txt_path = os.path.join(chunk_dir, fname + ".txt")

        cv2.imwrite(img_path, img)
        with open(txt_path, "w") as f:
            for x1, y1, x2, y2 in bboxes:
                f.write(f"{x1} {y1} {x2} {y2}\n")

        if i % 5000 == 0 and i > 0:
            print(f"  [worker {start_idx}] {i}/{count} done")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════
LABELS_NAMES = """
0
1
2
3
4
5
6
7
8
9
가
나
다
라
마
거
너
더
러
머
버
서
어
저
고
노
도
로
모
보
소
오
조
구
누
두
루
무
부
수
우
주
하
허
호
바
사
아
자
배
서울
부산
대구
인천
광주
대전
울산
세종
경기
강원
충북
충남
전북
전남
경북
경남
제주
육
"""


def main():
    parser = argparse.ArgumentParser(
        description="Korean License Plate Generator (KR_LPR format)"
    )
    parser.add_argument("--total", type=int, default=100000, help="总生成数量")
    parser.add_argument(
        "--chunk-size", type=int, default=10000, help="每个文件夹图片数"
    )
    parser.add_argument("--workers", type=int, default=0, help="进程数 (0=CPU核数)")
    parser.add_argument("--assets", type=str, default="./assets", help="素材目录")
    parser.add_argument(
        "--output",
        type=str,
        default="/home/ubuntu/mydata/GTY/korean_license_generator_30w",
        # default="./test_green",
        help="输出目录",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    n_workers = args.workers if args.workers > 0 else cpu_count()
    n_workers = min(n_workers, args.total)

    os.makedirs(args.output, exist_ok=True)

    # 写 labels.names
    with open(os.path.join(args.output, "labels.names"), "w", encoding="utf-8") as f:
        f.write(LABELS_NAMES + "\n")

    # 分配任务
    per_worker = args.total // n_workers
    remainder = args.total % n_workers

    tasks = []
    offset = 0
    for w in range(n_workers):
        cnt = per_worker + (1 if w < remainder else 0)
        worker_seed = args.seed + w * 10007
        tasks.append(
            (offset, cnt, args.output, args.assets, args.chunk_size, worker_seed)
        )
        offset += cnt

    print(f"Generating {args.total} plates with {n_workers} workers ...")
    print(f"  Output: {args.output}/chunk_XX/  ({args.chunk_size} per folder)")
    print(f"  Type weights: {TYPE_WEIGHTS}")

    if n_workers == 1:
        worker(tasks[0])
    else:
        with Pool(n_workers) as pool:
            pool.map(worker, tasks)

    # 统计
    total_imgs = sum(
        1
        for d in Path(args.output).iterdir()
        if d.is_dir()
        for f in d.iterdir()
        if f.suffix == ".jpg"
    )
    print(f"\nDone! Generated {total_imgs} images in {args.output}/")


if __name__ == "__main__":
    main()
