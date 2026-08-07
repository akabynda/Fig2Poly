from __future__ import annotations

import math
import random
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


PALETTES = [
    ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"],
    ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"],
    ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf"],
    ["#111111", "#555555", "#888888", "#bbbbbb"],
]
WORDS = ["Response", "Signal", "Value", "Rate", "Score", "Energy", "Intensity", "Amplitude",
         "Time", "Distance", "Frequency", "Sample", "Trial", "Temperature", "Observed", "Model",
         "Control", "Series", "Curve", "Estimate", "Trend", "Experiment", "Measurement", "Data"]
SYMBOLS = ["x", "y", "t", "f(x)", "P(x)", "g(t)", "delta", "mu", "sigma", "lambda", "a.u."]


def font(size: int) -> ImageFont.ImageFont:
    for candidate in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf",
                      "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf"):
        try:
            return ImageFont.truetype(candidate, size=max(7, size))
        except OSError:
            pass
    return ImageFont.load_default()


def random_text(rng: random.Random, min_words: int = 1, max_words: int = 4) -> str:
    words = rng.sample(WORDS, k=rng.randint(min_words, min(max_words, len(WORDS))))
    if rng.random() < 0.25:
        words.append(str(rng.randint(1, 99)))
    return " ".join(words)


def dashed_line(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], fill, width: int,
                pattern: tuple[float, float]) -> None:
    on, off = pattern
    phase, active = 0.0, True
    for p0, p1 in zip(points, points[1:]):
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        length = math.hypot(dx, dy)
        if length == 0:
            continue
        pos = 0.0
        while pos < length:
            limit = on if active else off
            take = min(limit - phase, length - pos)
            if active and take > 0:
                a, b = pos / length, (pos + take) / length
                draw.line([(p0[0] + dx*a, p0[1] + dy*a), (p0[0] + dx*b, p0[1] + dy*b)],
                          fill=fill, width=width)
            pos, phase = pos + take, phase + take
            if phase >= limit - 1e-6:
                phase, active = 0.0, not active


def draw_polyline(draw: ImageDraw.ImageDraw, segments: Iterable[list[tuple[float, float]]],
                  fill, width: int, style: str) -> None:
    for points in segments:
        if len(points) < 2:
            continue
        if style == "solid":
            draw.line(points, fill=fill, width=width, joint="curve")
        else:
            pattern = {"dashed": (10*width, 5*width), "dashdot": (7*width, 3*width),
                       "dotted": (1.2*width, 3*width)}[style]
            dashed_line(draw, points, fill, width, pattern)


def transform_pair(image: Image.Image, masks: list[Image.Image], rng: random.Random,
                   out_size: tuple[int, int], crop_p: float, rotate_p: float,
                   perspective_p: float, crop_min_keep: float = 0.62,
                   rotation_max_degrees: float = 16.0,
                   perspective_max_strength: float = 0.075) -> tuple[Image.Image, list[Image.Image], dict]:
    meta: dict = {"rotation_deg": 0.0, "crop": None, "perspective": False}
    if rng.random() < rotate_p:
        angle = rng.uniform(-rotation_max_degrees, rotation_max_degrees)
        meta["rotation_deg"] = round(angle, 3)
        fill = (rng.randint(225, 255),) * 3
        image = image.rotate(angle, Image.Resampling.BICUBIC, expand=True, fillcolor=fill)
        masks = [m.rotate(angle, Image.Resampling.NEAREST, expand=True, fillcolor=0) for m in masks]
    if rng.random() < perspective_p:
        w, h = image.size
        strength = rng.uniform(
            min(0.015, perspective_max_strength), perspective_max_strength
        )
        coeffs = (1+rng.uniform(-strength,strength), rng.uniform(-strength,strength), rng.uniform(-strength*w,strength*w),
                  rng.uniform(-strength,strength), 1+rng.uniform(-strength,strength), rng.uniform(-strength*h,strength*h),
                  rng.uniform(-strength/w,strength/w), rng.uniform(-strength/h,strength/h))
        image = image.transform((w,h), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)
        masks = [m.transform((w,h), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.NEAREST) for m in masks]
        meta["perspective"] = True
    w, h = image.size
    ratio = out_size[0] / out_size[1]
    if rng.random() < crop_p:
        keep = rng.uniform(crop_min_keep, 0.98)
        cw, ch = max(64,int(w*keep)), max(64,int(h*keep))
        if cw/ch > ratio: cw = int(ch*ratio)
        else: ch = int(cw/ratio)
        left, top = rng.randint(0,max(0,w-cw)), rng.randint(0,max(0,h-ch))
        box = (left,top,left+cw,top+ch)
        meta["crop"] = list(box)
    else:
        if w/h > ratio: cw,ch=int(h*ratio),h
        else: cw,ch=w,int(w/ratio)
        box=((w-cw)//2,(h-ch)//2,(w+cw)//2,(h+ch)//2)
    image=image.crop(box).resize(out_size,Image.Resampling.LANCZOS)
    masks=[m.crop(box).resize(out_size,Image.Resampling.NEAREST) for m in masks]
    return image,masks,meta


def degrade(image: Image.Image, rng: random.Random, np_rng: np.random.Generator,
            strength: float = 1.0) -> tuple[Image.Image, dict]:
    applied=[]
    if rng.random()<.24*strength:
        w,h=image.size
        factor=rng.uniform(1-(1-.42)*strength,1-(1-.82)*strength)
        image=image.resize((max(32,int(w*factor)),max(32,int(h*factor))),Image.Resampling.BILINEAR)
        image=image.resize((w,h),rng.choice([Image.Resampling.BILINEAR,Image.Resampling.BICUBIC]))
        applied.append("low_resolution")
    if rng.random()<.14*strength:
        gray=image.convert("L")
        image=Image.merge("RGB",(gray,gray,gray)); applied.append("grayscale")
    if rng.random()<.55*strength:
        image=ImageEnhance.Contrast(image).enhance(rng.uniform(1-.35*strength,1+.35*strength))
        image=ImageEnhance.Brightness(image).enhance(rng.uniform(1-.22*strength,1+.20*strength)); applied.append("tone")
    if rng.random()<.42*strength:
        image=image.filter(ImageFilter.GaussianBlur(rng.uniform(.25,.25+1.25*strength))); applied.append("blur")
    if rng.random()<.45*strength:
        arr=np.asarray(image).astype(np.float32); arr+=np_rng.normal(0,rng.uniform(1.5,1.5+10.5*strength),arr.shape)
        image=Image.fromarray(np.clip(arr,0,255).astype(np.uint8),"RGB"); applied.append("noise")
    if rng.random()<.20*strength:
        image=image.filter(ImageFilter.UnsharpMask(radius=2,percent=rng.randint(80,220),threshold=2)); applied.append("sharpen")
    if rng.random()<.16*strength:
        arr=np.asarray(image).copy()
        spacing=rng.randint(3,9); offset=rng.randrange(spacing)
        arr[offset::spacing]=np.clip(arr[offset::spacing].astype(np.int16)+rng.randint(-12,12),0,255)
        image=Image.fromarray(arr.astype(np.uint8),"RGB"); applied.append("scanlines")
    return image,{"applied":applied}


def mask_bbox(mask: Image.Image) -> tuple[list[int], int]:
    a=np.asarray(mask)>0
    ys,xs=np.nonzero(a)
    if not len(xs): return [0,0,0,0],0
    x0,x1,y0,y1=int(xs.min()),int(xs.max()),int(ys.min()),int(ys.max())
    return [x0,y0,x1-x0+1,y1-y0+1],int(a.sum())
