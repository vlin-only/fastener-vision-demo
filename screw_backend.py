"""单颗螺丝识别与反馈训练。直接运行打开上传页，指定 --image 可识别本地照片。"""
import argparse
import hashlib
import hmac
import io
import json
import math
import os
import secrets
import sqlite3
import sys
import webbrowser
from dataclasses import dataclass
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Optional
from urllib.request import urlopen

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn.functional import normalize

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "models/screw.pt"


# 图像测量


@dataclass
class Part:
    box: tuple
    diameter: Optional[float] = None
    segment: Optional[tuple] = None
    reason: str = "unverified shape"
    source: str = "contour"


def tick_band(gray, k):
    """在同一基线上寻找至少六根等距细刻线；返回覆盖区和像素节距。"""
    dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, np.ones((1, k), np.uint8))
    threshold = max(12, 0.65 * cv2.threshold(dark, 0, 255, cv2.THRESH_OTSU)[0])
    strokes = cv2.morphologyEx(np.uint8(dark > threshold), cv2.MORPH_OPEN,
                               np.ones((k, 1), np.uint8))
    stats = cv2.connectedComponentsWithStats(strokes)[2][1:]
    bars = stats[(stats[:, 3] >= k) & (stats[:, 2] < stats[:, 3] * 0.4)]
    best = None
    for start in np.unique(bars[:, 1]):
        group = bars[np.abs(bars[:, 1] - start) <= max(2, k * 0.45)]
        if len(group) < 6:
            continue
        group = group[np.argsort(group[:, 0])]
        centers = group[:, 0] + group[:, 2] / 2
        gaps = np.diff(centers)
        pitch = np.median(gaps)
        if pitch < 4:
            continue
        # 保留连续的等距段，避免相邻文字或边缘扩大尺区。
        breaks = np.flatnonzero(np.abs(gaps - pitch) > max(1.5, pitch * 0.18)) + 1
        for run in np.split(group, breaks):
            if len(run) < 6 or (best is not None and len(run) <= best[0]):
                continue
            xx = run[:, 0] + run[:, 2] / 2
            step = float((xx[-1] - xx[0]) / (len(xx) - 1))
            depth = np.median(run[:, 3])
            if not 0.65 * step <= depth <= 15 * step:
                continue
            y = int(np.median(run[:, 1]))
            best = (len(run), step, (int(xx[0]-1.2*step), y-2,
                    int(np.ceil(xx[-1]+1.2*step)), int(y+2.15*np.max(run[:, 3]))))
    return best


def aligned_ruler(gray):
    best = None
    for turns in range(4):
        rotated = np.ascontiguousarray(np.rot90(gray, turns))
        k = max(5, min(gray.shape)//35) | 1
        found = tick_band(rotated, k)
        if found is None and k >= 11:
            found = tick_band(rotated, max(5,k//2) | 1)
        if found is None and k >= 25:
            found = tick_band(rotated, max(5,k//4) | 1)
        if found is None or (best is not None and found[0] <= best[0]):
            continue
        count, pitch, (x1,y1,x2,y2) = found
        mask = np.zeros(rotated.shape, np.uint8)
        mask[max(y1,0):max(y2,0), max(x1,0):max(x2,0)] = 255
        best = count, pitch, np.ascontiguousarray(np.rot90(mask, -turns))
    if best is None:
        return np.zeros_like(gray), None
    return best[2], best[1]


def ruler_region(gray):
    """先找横竖刻线。斜放的尺按长直线主方向校正，尺区再映回原图。"""
    if min(gray.shape) < 12:
        return np.zeros_like(gray), None
    mask, pitch = aligned_ruler(gray)
    if pitch is not None:
        return mask, pitch
    lines = cv2.createLineSegmentDetector().detect(gray)[0]
    if lines is None:
        return mask, None
    edges = lines.reshape(-1,4)
    vectors = edges[:,2:] - edges[:,:2]
    lengths = np.linalg.norm(vectors, axis=1)
    keep = lengths >= max(12,min(gray.shape)/20)
    if np.count_nonzero(keep) < 6:
        return mask, None
    angles = np.degrees(np.arctan2(vectors[keep,1],vectors[keep,0])) % 90
    counts, _ = np.histogram(angles, bins=np.arange(0,91,3), weights=lengths[keep])
    selected = abs(angles - (3*np.argmax(counts)+1.5)) <= 3
    angle = float(np.median(angles[selected]))
    if min(angle, 90-angle) < 1:
        return mask, None
    h,w = gray.shape
    matrix = cv2.getRotationMatrix2D((w/2,h/2),angle,1)
    dim = np.ceil(abs(matrix[:,:2]) @ [w,h]).astype(int)
    matrix[:,2] += (dim-[w,h])/2
    aligned = cv2.warpAffine(gray, matrix, tuple(dim), borderValue=float(np.median(gray)))
    aligned_mask, pitch = aligned_ruler(aligned)
    return cv2.warpAffine(aligned_mask, cv2.invertAffineTransform(matrix), (w,h),
                          flags=cv2.INTER_NEAREST), pitch


def measure_shank(contour):
    """搜索头部较宽、两侧近似平行的杆段；沿垂直于杆轴的截面取外径。"""
    pts = contour.reshape(-1, 2).astype(float)
    origin = pts.mean(axis=0)
    pts -= origin
    # 限制局部工作图的大小，最终测量仍换回原图像素。
    factor = min(1.0, 240 / max(1.0, np.ptp(pts, axis=0).max()))
    best = None
    for deg in range(0, 180, 3):
        a = np.deg2rad(deg)
        basis = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        projected = pts @ basis * factor
        offset = np.floor(projected.min(axis=0))
        polygon = np.rint(projected - offset).astype(np.int32)
        cols, rows = polygon.max(axis=0) + 1
        if cols < 15 or rows < 8:
            continue
        silhouette = np.zeros((rows, cols), np.uint8)
        cv2.fillPoly(silhouette, [polygon], 1)
        top = silhouette.argmax(axis=0)
        bottom = rows - 1 - silhouette[::-1].argmax(axis=0)
        for reverse in (False, True):
            hi, lo = (top[::-1], bottom[::-1]) if reverse else (top, bottom)
            # 外侧 10% 和头杆过渡区不参与杆径估计。
            start, stop = int(0.46*cols), int(0.89*cols)
            thickness = lo[start:stop] - hi[start:stop] + 1
            d = np.percentile(thickness, 85)
            head = (lo-hi+1)[int(.05*cols):max(2,int(.38*cols))]
            ratio = np.percentile(head, 80) / max(d, 1)
            if d < 5 or cols/d < 1.65 or not 1.32 <= ratio <= 4.5:
                continue
            center = (lo[start:stop]+hi[start:stop])/2
            taper = np.percentile(thickness, 90) - np.percentile(thickness, 10)
            drift = abs(np.median(center[-max(2,len(center)//4):]) -
                        np.median(center[:max(2,len(center)//4)]))
            score = (taper + 2.5 * drift) / d
            if score > 0.36 or (best is not None and score >= best[0]):
                continue
            idx = start + int(np.argmin(abs(thickness-d)))
            x = cols - 1 - idx if reverse else idx
            ends = np.array([[x, hi[idx]], [x, lo[idx]]], dtype=float)
            ends = ((ends + offset) / factor) @ basis.T + origin
            best = score, float(d/factor), tuple(map(tuple, np.rint(ends).astype(int)))
    return None if best is None else best[1:]


def components(gray, exclusion):
    border = np.zeros_like(gray, dtype=bool)
    border[[0,-1],:] = True
    border[:,[0,-1]] = True
    valid = exclusion == 0
    if not valid.any():
        return []
    samples = gray[border & valid]
    samples = samples if samples.size else gray[valid]
    background = np.median(samples)
    difference = np.uint8(np.abs(gray.astype(float) - background))
    threshold = cv2.threshold(difference[valid], 0, 255, cv2.THRESH_OTSU)[0]
    noise = np.median(abs(samples.astype(float) - background))
    if np.percentile(abs(samples.astype(float)-background), 90) < 4:
        threshold = 6  # 均匀背景上也保留低对比度工件
    mask = np.uint8(difference > max(6, min(18, 2*noise), threshold)) * 255
    mask[~valid] = 0
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    height, width = gray.shape
    parts = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if (area < max(50, 0.0015 * gray.size) or area < .23*w*h or
                min(w,h) < 7 or x <= 0 or y <= 0 or x+w >= width or y+h >= height):
            continue
        part = Part((x, y, x+w, y+h))
        local_mask = mask[y:y+h, x:x+w]
        local_gray = gray[y:y+h, x:x+w]
        if np.percentile(local_gray[local_mask != 0], 10) >= 248:
            continue  # 只有高光，缺少工件轮廓
        holes, links = cv2.findContours(local_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        ring = False
        for hole, link in zip(holes, links.reshape(-1,4)):
            hole_area = cv2.contourArea(hole)
            if link[3] >= 0 and 0.07*area < hole_area < .70*area:
                hx,hy,hw,hh = cv2.boundingRect(hole)
                if (area >= .89*cv2.contourArea(cv2.convexHull(c))
                        and abs(hx+hw/2-w/2) < w*.14 and abs(hy+hh/2-h/2) < h*.14
                        and .65 < hw/hh < 1.55
                        and abs(float(local_gray[hy+hh//2,hx+hw//2])-background) < 30):
                    ring = True
                    break
        measurement = None if ring else measure_shank(c)
        if ring:
            part.reason = "bore / inspect"
        if measurement:
            part.diameter, part.segment = measurement
            part.reason = "shank"
        parts.append(part)
    return sorted(parts, key=lambda part: (part.box[1], part.box[0]))


# 螺丝区域定位


GRID=28


@torch.inference_mode()
def patches(encoder,rgb,device):
    reduced=cv2.resize(rgb,(14*GRID,14*GRID),interpolation=cv2.INTER_AREA)
    x=torch.from_numpy(reduced).to(device,dtype=torch.float32).permute(2,0,1)[None]/255
    mean=x.new_tensor([.485,.456,.406])[None,:,None,None]
    std=x.new_tensor([.229,.224,.225])[None,:,None,None]
    return normalize(encoder.forward_features((x-mean)/std)[0,1:],dim=-1).cpu()


def locate(encoder,locator,rgb,device):
    feature=patches(encoder,rgb,device)
    score=((feature-locator["mean"])@locator["weight"]+locator["bias"]).reshape(GRID,GRID).numpy()
    peak=float(score.max())
    mask=np.uint8(score>max(.22,.43*peak))
    count,labels,stats,centers=cv2.connectedComponentsWithStats(mask)
    h,w=rgb.shape[:2]
    groups=[]
    for idx in range(1,count):
        if stats[idx,4]<2:
            continue
        yy,xx=np.where(labels==idx)
        strength=score[yy,xx]
        weight=strength.clip(.01)
        pts=(np.stack([xx,yy],axis=1)+.5)*[w/GRID,h/GRID]
        center=np.average(pts,axis=0,weights=weight)
        local=pts-center
        cov=(local*weight[:,None]).T@local/weight.sum()
        _,axes=np.linalg.eigh(cov)
        along=local@axes[:,-1]
        span=np.ptp(along)+max(w,h)/GRID
        angle=np.degrees(np.arctan2(axes[1,-1],axes[0,-1]))
        merit=float(weight.sum())*np.exp(-2*np.linalg.norm(center/[w,h]-.5)**2)
        groups.append((merit,tuple(center),span,angle,float(strength.max())))
    groups.sort(key=lambda g:g[0],reverse=True)
    return groups


# 型号参考库


def signature(features, mix):
    # 各分支使用工件局部视图，避免用整张图的背景决定型号。
    return torch.cat([
        normalize(features[mode][..., 1:, :].mean(dim=-2), dim=-1) * weight**0.5
        for mode, weight in mix.items()
    ], dim=-1)


def reference_groups(catalog):
    """按型号归组训练参考图和已标注的补充参考图，并预先归一化。"""
    refs, labels = catalog["references"], catalog["labels"]
    if "wild_references" in catalog:
        refs = torch.cat((refs, catalog["wild_references"]))
        labels = torch.cat((labels, catalog["wild_labels"]))
    class_ids = set(range(len(catalog["classes"])))
    if refs.ndim != 2 or labels.shape != (len(refs),) or set(labels.tolist()) != class_ids:
        raise ValueError("型号库的参考特征与类别不匹配")
    if not torch.isfinite(refs).all() or (refs.norm(dim=1) == 0).any():
        raise ValueError("型号库包含无效的参考特征")
    refs = normalize(refs, dim=1)
    return [refs[labels == index] for index in range(len(class_ids))]


def nearest_class_scores(query, groups):
    """每个型号取其最相似参考图的余弦相似度。"""
    query = normalize(query, dim=0)
    return torch.stack([(group @ query).max() for group in groups])


# 用户反馈与训练


def kernel(left, right):
    # 对相近外观作局部调整，距离较远的工件沿用原分类器的得分。
    return np.exp(-24.0 * np.maximum(0.0, 1.0 - left @ right.T))


class FeedbackTrainer:
    def __init__(self, weights, directory=None):
        weights = Path(weights)
        root = Path(directory) if directory is not None else weights.parent.parent / "feedback"
        root.mkdir(parents=True, exist_ok=True)
        with weights.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        self.db = sqlite3.connect(root / (digest + ".sqlite3"), timeout=30, check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS samples (
                digest TEXT PRIMARY KEY, photo BLOB NOT NULL, predicted TEXT NOT NULL,
                label TEXT NOT NULL, feature BLOB NOT NULL, base_scores BLOB NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS adapter (id INTEGER PRIMARY KEY, parameters BLOB NOT NULL);
        """)
        self.version = None
        self.groups = {}

    def similarities(self, query, classes):
        version = self.db.execute("PRAGMA data_version").fetchone()[0]
        if self.version != version:
            groups = {}
            for label, feature in self.db.execute("SELECT label, feature FROM samples"):
                groups.setdefault(label, []).append(np.frombuffer(feature, dtype="<f4"))
            self.groups = {
                label: normalize(torch.from_numpy(np.stack(features)), dim=1)
                for label, features in groups.items()
            }
            self.version = version
        return torch.stack([
            (self.groups[label] @ query).max().clamp(-1, 1)
            if label in self.groups else query.new_tensor(-float("inf"))
            for label in classes
        ])

    def confirmed_label(self, photo):
        row = self.db.execute("SELECT label FROM samples WHERE digest=?", (sha256(photo).hexdigest(),)).fetchone()
        return row[0] if row else None

    def learn(self, photo, predicted, label, feature, scores, classes):
        digest = sha256(photo).hexdigest()
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            old = self.db.execute("SELECT label FROM samples WHERE digest=?", (digest,)).fetchone()
            if old and old[0] == label:
                return {"changed": False, "sample_count": self.count(), "label": label}
            self.db.execute("""
                INSERT INTO samples(digest,photo,predicted,label,feature,base_scores)
                VALUES(?,?,?,?,?,?) ON CONFLICT(digest) DO UPDATE SET
                predicted=excluded.predicted, label=excluded.label, updated_at=CURRENT_TIMESTAMP
            """, (digest, photo, predicted, label, feature.numpy().astype("<f4").tobytes(),
                  scores.numpy().astype("<f4").tobytes()))
            rows = self.db.execute("SELECT feature,base_scores,label FROM samples ORDER BY digest").fetchall()
            features = np.stack([np.frombuffer(row[0], dtype="<f4") for row in rows])
            base = np.stack([np.frombuffer(row[1], dtype="<f4") for row in rows])
            targets = np.zeros_like(base)
            for i, row in enumerate(rows):
                actual = classes.index(row[2])
                competing = np.max(np.delete(base[i], actual))
                targets[i, actual] = max(0.0, competing - base[i, actual] + 0.5)
            # 正则化核岭回归，以基础分类器为起点拟合人工标签的得分修正。
            gram = kernel(features, features).astype(np.float64)
            gram.flat[::len(rows) + 1] += 0.05
            coefficients = np.linalg.solve(gram, targets).astype(np.float32)
            buf = io.BytesIO()
            np.savez(buf, features=features, coefficients=coefficients)
            self.db.execute("INSERT OR REPLACE INTO adapter VALUES(1,?)", (buf.getvalue(),))
        self.version = None
        return {"changed": True, "sample_count": len(rows), "label": label}

    def count(self):
        return self.db.execute("SELECT COUNT(*) FROM samples").fetchone()[0]


# 照片编码与相似度排序


ARCH = "vit_small_patch14_dinov2.lvd142m"
SIDE = 224


def read_photo(path):
    with Image.open(path) as im:
        im.draft("RGB", (1100, 1100))
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((900, 900), Image.Resampling.LANCZOS)
        return np.asarray(im)


def candidate_regions(image):
    """找局部凸出的深浅边缘，优先选择居中、完整、长形的候选。"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    m = min(h,w)
    if m < 12:
        return []
    clahe = cv2.createCLAHE(2, (8,8)).apply(gray)
    edges = cv2.Canny(cv2.GaussianBlur(clahe, (3,3), .8), 42, 112)
    k = max(3, round(m / 115)) | 1
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k)))
    contours = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    regions = []
    for c in contours:
        x,y,bw,bh = cv2.boundingRect(c)
        if x < 3 or y < 3 or x+bw >= w-2 or y+bh >= h-2:
            continue
        (cx,cy),(a,b),angle=cv2.minAreaRect(c)
        long, short = max(a,b),min(a,b)
        area=cv2.contourArea(c)
        if short < max(3,m*.007) or not .07*m < long < .94*m or not 1.3 < long/short < 21:
            continue
        fill=area/max(1,long*short)
        if fill < .22:
            continue
        offset = np.hypot((cx-w/2)/w,(cy-h/2)/h)
        # 矩形直边的宽钢尺往往比工件大，细螺杆和螺头通常有宽度变化。
        score = (np.exp(-8*offset**2) * min(1,long/(m*.25)) * min(1, short/(m*.035))
                 * (.65+.35*fill))
        if fill > .89 and short > .055*m:
            score *= .48
        regions.append((float(score), (cx,cy), long, short, angle if a>=b else angle+90))
    return sorted(regions, key=lambda r:r[0], reverse=True)


def square_crop(rgb, center, span, angle=0):
    matrix=cv2.getRotationMatrix2D(center,angle,SIDE/max(1,span))
    matrix[:,2]+=np.array([SIDE/2,SIDE/2])-center
    return cv2.warpAffine(rgb,matrix,(SIDE,SIDE),flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def photo_views(rgb,locations=None):
    h,w=rgb.shape[:2]
    center=(w/2,h/2)
    if locations:
        _,focus,long,angle,_=locations[0]
    else:
        regions=candidate_regions(rgb)
        if regions:
            _,focus,long,_,angle=regions[0]
        else:
            focus,long,angle=center,min(w,h)*.65,0.
    # 同时保留整图上下文和对齐杆轴的局部图；裁剪失败时仍有整图通道。
    views=[square_crop(rgb,center,max(w,h)),
           square_crop(rgb,focus,max(long*1.48,min(w,h)*.20),angle),
           square_crop(rgb,focus,max(long*1.8,min(w,h)*.33))]
    return views


def backbone():
    import timm
    return timm.create_model(ARCH,pretrained=False,num_classes=0,dynamic_img_size=True)


@torch.inference_mode()
def encode(model, views, device, batch_size=8):
    features=[]
    for start in range(0,len(views),batch_size):
        rgb=np.stack(views[start:start+batch_size])
        x=torch.from_numpy(rgb).to(device,dtype=torch.float32).permute(0,3,1,2)/255
        mean=x.new_tensor([.485,.456,.406])[None,:,None,None]
        std=x.new_tensor([.229,.224,.225])[None,:,None,None]
        tokens=model.forward_features((x-mean)/std)
        grid=tokens[:,1:].reshape(len(x),SIDE//14,SIDE//14,-1)
        central=grid[:,5:11,3:13].mean((1,2))
        features.append(torch.stack((normalize(tokens[:,0],dim=-1),
                                     normalize(central,dim=-1)),dim=1).cpu())
    return torch.cat(features)


def descriptors(encoder, rgb, device, locations=None):
    views=photo_views(rgb,locations)
    flips=[cv2.flip(view,1) for view in views]
    pair=encode(encoder,views+flips,device).reshape(2,3,2,-1).mean(0)
    return normalize(torch.stack((pair[0,0],pair[1,0],pair[2,0],pair[1,1])),dim=-1)


def classification_scores(features, heads, mix):
    """按训练时的视图权重和分支权重融合各型号的分类得分。"""
    scores = []
    for mode, weight in mix.items():
        head = heads[mode]
        views = features[mode] * head["view_weights"][:, None]
        score = (views.flatten() - head["mean"]) @ head["weight"] + head["bias"]
        scores.append(weight * score)
    return torch.stack(scores).sum(dim=0)


class ScrewClassifier:
    def __init__(self, weights=DEFAULT_WEIGHTS, device="cpu", feedback_dir=None):
        if device=="cpu":
            torch.set_num_threads(min(torch.get_num_threads(),4))
        ckpt=torch.load(weights,map_location="cpu",weights_only=True,mmap=True)
        if ckpt.get("format")!="screw-photo-dino-v1":
            raise ValueError("权重格式不匹配")
        self.device=device
        self.model=backbone()
        self.model.load_state_dict(ckpt["encoder"],strict=True)
        self.model.to(device).eval()
        self.mix=ckpt["mix"]
        self.heads=ckpt["heads"]
        self.classes=ckpt["classes"]
        self.locator=ckpt.get("locator")
        self.catalog=ckpt.get("catalog")
        if self.catalog is None or self.locator is None:
            raise ValueError("权重缺少定位器或型号库，请使用随附的 models/screw.pt")
        if self.catalog["classes"] != self.classes:
            raise ValueError("分类权重与型号库的类别顺序不一致")

        self.reference_groups = reference_groups(self.catalog)
        self.feedback = FeedbackTrainer(weights, feedback_dir)

    def _describe(self, image):
        if isinstance(image,bytes):
            rgb=read_photo(io.BytesIO(image))
        else:
            rgb=read_photo(image) if isinstance(image,(str,Path)) else np.asarray(image)
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<12:
            raise ValueError("输入必须是至少 12×12 的 RGB uint8 图片")
        if max(rgb.shape[:2])>900:
            scale=900/max(rgb.shape[:2])
            rgb=cv2.resize(rgb,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        locations=locate(self.model,self.locator,rgb,self.device) if self.locator is not None else []
        features = {
            mode: descriptors(self.model, rgb, self.device, locations if mode == "learned" else None)
            for mode in self.mix
        }
        query = normalize(signature(features, self.mix), dim=0)
        scores = classification_scores(features, self.heads, self.mix)
        center = None
        if locations:
            x, y = locations[0][1]
            center = [float(x / rgb.shape[1]), float(y / rgb.shape[0])]
        return query, scores, center

    def learn(self, photo, predicted, label):
        if label not in self.classes or predicted not in self.classes:
            raise ValueError("请选择已收录的型号")
        feature, base, _ = self._describe(photo)
        return self.feedback.learn(photo, predicted, label, feature, base, self.classes)

    def predict(self, image):
        if isinstance(image, (str, Path)):
            image = Path(image).read_bytes()
        query, _, center = self._describe(image)
        catalog = nearest_class_scores(query, self.reference_groups).clamp(-1, 1)
        feedback = self.feedback.similarities(query, self.classes)
        scores = torch.maximum(catalog, feedback)
        known = self.feedback.confirmed_label(image) if isinstance(image, bytes) else None
        if known in self.classes:
            # 同一照片与自身的相似度为 1，出现并列时采用最近确认的标签。
            scores[self.classes.index(known)] = 1.0
        order = sorted(range(len(scores)), key=lambda i: (
            -float(scores[i]), self.classes[i] != known, -float(feedback[i]), i
        ))
        ranking = [{"class_name": self.classes[i], "score": float(scores[i])} for i in order]
        return {
            "class_name": ranking[0]["class_name"],
            "status": "single_candidate",
            "message": "已返回参考库中最相似的型号",
            "catalog_similarity": float(scores[order[0]]),
            "decision_source": "nearest_reference",
            "center_fraction": center,
            "ranking": ranking,
        }


BINS = {"M4": 1, "M5": 2, "M6": 3, "M8": 4, "M10": 5}


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("距离或比例必须是有限正数")
    return value


def segment_length(points, width, height):
    x1, y1, x2, y2 = points
    if not (0 <= x1 < width and 0 <= x2 < width and 0 <= y1 < height and 0 <= y2 < height):
        raise ValueError("测量端点超出正向显示的原图范围")
    return positive(math.hypot(x2 - x1, y2 - y1))


def full_rgb(photo):
    with Image.open(io.BytesIO(photo) if isinstance(photo, bytes) else photo) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))


class ScrewIdentifier:
    def __init__(self, weights=DEFAULT_WEIGHTS, feedback_dir=None):
        self.classifier = ScrewClassifier(weights, feedback_dir=feedback_dir)
        if set(self.classifier.classes) != set(BINS):
            raise ValueError("权重类别与当前分箱表不一致")

    def predict(self, photo, *, diameter_mm=None, ticks=None, px_per_mm=None,
                tick_mm=None, shank=None):
        if sum(v is not None for v in (diameter_mm, ticks, px_per_mm, tick_mm)) > 1:
            raise ValueError("一次只能使用一种测量或标定方式")
        calibrated = any(v is not None for v in (ticks, px_per_mm, tick_mm))
        if shank is not None and not calibrated:
            raise ValueError("图片杆径端点需要搭配钢尺刻线或实测像素比例")
        result = self.classifier.predict(photo)
        result.update(bin=BINS[result["class_name"]], diameter_mm=None, scale_px_per_mm=None, calibration=None)
        source = None
        if diameter_mm is not None:
            diameter_mm = positive(diameter_mm)
            source = "caliper"
        elif calibrated:
            rgb = full_rgb(photo)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape
            mask = np.zeros_like(gray)
            if ticks is not None:
                scale = segment_length(ticks[:4], w, h) / positive(ticks[4])
                source = "ruler_endpoints"
            elif tick_mm is not None:
                spacing = positive(tick_mm)
                mask, pitch = ruler_region(gray)
                if pitch is None:
                    return result
                scale, source = pitch / spacing, "ruler_ticks"
            else:
                scale, source = positive(px_per_mm), "calibrated_camera"
            result.update(scale_px_per_mm=scale, calibration=source)
            if shank is not None:
                diameter_mm = segment_length(shank, w, h) / scale
            else:
                center = result["center_fraction"]
                if center is None:
                    return result
                x, y = center[0] * w, center[1] * h
                parts = [p for p in components(gray, mask) if p.diameter is not None
                         and p.box[0] <= x < p.box[2] and p.box[1] <= y < p.box[3]]
                if len(parts) != 1:
                    return result
                diameter_mm = parts[0].diameter / scale

        if diameter_mm is not None:
            result.update(diameter_mm=diameter_mm, calibration=source)
        return result


def save_comparison(photo, result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    original = full_rgb(photo)
    lightness, a, b = cv2.split(cv2.cvtColor(original, cv2.COLOR_RGB2LAB))
    enhanced = cv2.createCLAHE(2.0, (8, 8)).apply(lightness)
    enhanced = cv2.cvtColor(cv2.merge((enhanced, a, b)), cv2.COLOR_LAB2RGB)
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
        try:
            axes[0].imshow(original)
            axes[0].set_title("Original")
            axes[1].imshow(enhanced)
            center = result["center_fraction"]
            if center:
                axes[1].plot(center[0]*original.shape[1], center[1]*original.shape[0], "r+")
            title = result["class_name"] or result["status"]
            if result["diameter_mm"] is not None:
                title += f" | measured {result['diameter_mm']:.2f} mm"
            axes[1].set_title(title)
            for ax in axes:
                ax.axis("off")
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=140)
        finally:
            plt.close(fig)


# 内置上传页面


UPLOAD_PAGE = r"""<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>螺丝型号识别</title>
<style>
*{box-sizing:border-box} :root{font-family:"Microsoft YaHei UI","PingFang SC",sans-serif;color:#172e33;background:#f1f4f2;font-size:16px} body{margin:0}
main{max-width:1100px;margin:auto;padding:clamp(24px,6vw,76px) 22px} h1{margin:0;font-size:clamp(30px,4vw,48px);letter-spacing:-.04em} header p{color:#486268;margin:12px 0 30px;line-height:1.6}
.grid{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(260px,1fr);gap:18px}
section{min-width:0;border:1px solid #cfdbd8;border-radius:16px;background:#fff;padding:24px} h2{font-size:18px;margin:0 0 18px}
.drop{width:100%;height:clamp(240px,38vh,370px);display:grid;place-items:center;border:2px dashed #9abab4;border-radius:10px;background:#f6f9f8;color:#385850;cursor:pointer;overflow:hidden;position:relative}
.drop.over,.drop:hover{background:#e8f4f0;border-color:#177765}.drop img{position:absolute;width:100%;height:100%;object-fit:contain;background:#eef2f1}
.empty{text-align:center}.arrow{display:block;margin:0 auto 12px;font-size:32px;color:#197c67}.empty strong,.empty small{display:block}.empty small{margin-top:8px;color:#546d69}
.actions{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-top:18px}.filename{color:#53696c;font-size:13px;overflow-wrap:anywhere}.go{background:#126855;border:0;border-radius:8px;color:white;padding:12px 22px;font:inherit;font-weight:700;cursor:pointer;white-space:nowrap}.go:disabled{background:#dee6e3;color:#586b66;cursor:default}
.summary{padding:22px 0;border-top:1px solid #e1e8e5}.answer{font-size:clamp(22px,3vw,36px);font-weight:750;line-height:1.35;margin:0 0 10px}.desc{color:#506568;line-height:1.7;margin:0}
[data-state="ok"] .answer{color:#126855}[data-state="error"] .answer{color:#8a5212}
button:focus-visible{outline:3px solid #d18a29;outline-offset:2px}[hidden]{display:none!important}
.feedback{border:0;border-top:1px solid #e1e8e5;padding:16px 0 0;margin:0}.feedback legend{padding-right:8px;font-weight:600}.feedback button,.feedback select{font:inherit;border:1px solid #9abab4;border-radius:6px;background:white;color:#214c42;padding:8px 12px;cursor:pointer}.feedback button:hover{background:#e8f4f0}.feedback .choices,.feedback .correct-label{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:10px}#feedback-note{margin-top:12px}
@media(max-width:720px){.grid{grid-template-columns:1fr}section{padding:18px}}
</style></head><body><main>
<header><h1>螺丝型号识别</h1><p>上传单颗螺丝照片，识别 M4、M5、M6、M8 或 M10。</p></header>
<div class="grid"><section><h2>待识别照片</h2>
<input id="file" type="file" accept="image/jpeg,image/png,image/webp" hidden>
<button class="drop" id="drop" type="button" aria-label="选择或更换照片"><span class="empty" id="empty"><span class="arrow" aria-hidden="true">↑</span><strong>选择照片或拖到这里</strong><small>JPG、PNG、WebP · 最大 32 MiB</small></span><img id="preview" alt="所选螺丝照片" hidden></button>
<div class="actions"><span class="filename" id="filename">尚未选择照片</span><button id="go" class="go" type="button" disabled>开始识别</button></div>
</section><section><h2>识别结果</h2><div class="summary" id="summary" aria-live="polite"><p class="answer" id="answer">等待照片</p><p class="desc" id="desc">选择照片后，点击“开始识别”。</p></div>
<fieldset id="feedback" class="feedback" hidden><legend>这个型号对吗？</legend>
<div class="choices"><button type="button" id="yes">识别正确</button><button type="button" id="no">识别有误</button></div>
<div id="correction" class="correct-label" hidden><label for="actual">正确型号</label><select id="actual"></select><button type="button" id="submit-label">提交并训练</button></div>
<p class="desc" id="feedback-note" role="status">确认后，这张照片和真实型号将保存在本机，用于训练。</p></fieldset>
</section></div></main>
<script>
const el=id=>document.getElementById(id);
const fileInput=el('file'),drop=el('drop'),go=el('go'),preview=el('preview');
let photo=null,objectURL=null,version=0,pending=null;
let prediction=null;
function resetFeedback(){
  prediction=null;el('feedback').hidden=true;el('feedback').disabled=false;el('correction').hidden=true;
  el('feedback-note').textContent='确认后，这张照片和真实型号将保存在本机，用于训练。';
}
function show(title,detail,state='idle'){
  el('summary').dataset.state=state;el('answer').textContent=title;el('desc').textContent=detail;
}
function select(file){
  if(!file)return;
  version++;if(pending)pending.abort();
  resetFeedback();
  photo=null;go.disabled=true;go.textContent='开始识别';
  preview.hidden=true;preview.removeAttribute('src');el('empty').hidden=false;
  if(objectURL){URL.revokeObjectURL(objectURL);objectURL=null}
  el('filename').textContent=file.name;
  if(!['image/jpeg','image/png','image/webp'].includes(file.type)||!file.size||file.size>32*1024*1024){
    show('无法读取照片','请选择 32 MiB 以内的 JPG、PNG 或 WebP 图片。','error');return;
  }
  const current=version;
  preview.onload=()=>{if(current!==version)return;preview.hidden=false;el('empty').hidden=true;photo=file;go.disabled=false;show('可以识别','请确认预览中的照片，然后点击“开始识别”。')};
  preview.onerror=()=>{if(current===version)show('照片解码失败','请换一张能正常打开的图片。','error')};
  show('正在读取照片','');objectURL=URL.createObjectURL(file);preview.src=objectURL;
}
drop.onclick=()=>fileInput.click();
fileInput.onchange=()=>{select(fileInput.files[0]);fileInput.value=''};
drop.ondragover=e=>{e.preventDefault();drop.classList.add('over')};
drop.ondragleave=()=>drop.classList.remove('over');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('over');select(e.dataTransfer.files[0])};
el('yes').onclick=()=>sendFeedback(prediction.class_name);
el('no').onclick=()=>{el('correction').hidden=false;el('actual').focus()};
el('submit-label').onclick=()=>{if(el('actual').value)sendFeedback(el('actual').value)};
el('actual').onchange=()=>{el('submit-label').disabled=!el('actual').value};
async function sendFeedback(label){
  if(!prediction||el('feedback').disabled)return;
  const current=version,record=prediction;
  el('feedback').disabled=true;go.disabled=true;
  el('feedback-note').textContent='正在保存样本并训练…';
  try{
    const response=await fetch('/feedback',{method:'POST',body:photo,headers:{
      'Content-Type':photo.type,'X-Prediction-Token':record.feedback_token,
      'X-Predicted-Class':record.class_name,'X-Actual-Class':label}});
    const data=await response.json();
    if(current!==version)return;
    if(!response.ok)throw new Error(data.error||'提交失败，请重试。');
    el('correction').hidden=true;
    el('feedback-note').textContent=data.changed
      ?'已按 '+data.label+' 完成训练，下次识别即生效。累计 '+data.sample_count+' 张。'
      :'此照片已按 '+data.label+' 训练过。累计 '+data.sample_count+' 张。';
  }catch(err){if(current===version)el('feedback-note').textContent=err.message||'提交失败，请重试。'}
  finally{if(current===version){el('feedback').disabled=false;go.disabled=false}}
}
go.onclick=async()=>{
  if(!photo||go.disabled)return;
  const headers={'Content-Type':photo.type};
  const current=version;pending=new AbortController();go.disabled=true;go.textContent='正在识别…';
  resetFeedback();
  show('正在识别','正在本机分析照片，请稍候。');
  try{
    const res=await fetch('/predict',{method:'POST',headers,body:photo,signal:pending.signal});
    if(!res.ok)throw new Error(res.status===400?'图片解码失败，请更换照片。':'本机服务暂时无法完成识别，请重试。');
    const data=await res.json();if(current!==version)return;
    show(data.class_name,'','ok');
    prediction=data;
    const menu=el('actual');
    menu.replaceChildren(new Option('请选择',''));
    for(const item of [...data.ranking].sort((a,b)=>Number(a.class_name.slice(1))-Number(b.class_name.slice(1)))){
      if(item.class_name!==data.class_name)menu.add(new Option(item.class_name,item.class_name));
    }
    el('submit-label').disabled=true;el('feedback').hidden=false;
  }catch(err){if(current===version)show('识别失败',err instanceof TypeError?'无法连接本机服务，请重新启动识别工具。':err.message,'error')}
  finally{if(current===version){pending=null;go.disabled=false;go.textContent='重新识别'}}
};
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def receipt(self, image, predicted):
        message = predicted.encode("ascii") + b":" + hashlib.sha256(image).digest()
        return hmac.digest(self.server.feedback_key, message, "sha256").hex()

    def do_GET(self):
        if self.path != "/":
            return self.send_json(404, {"error": "Unknown route"})
        content = UPLOAD_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Screw-Upload", "single-file-1")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def send_json(self,status,payload):
        data=json.dumps(payload,ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Content-Length",str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path not in {"/predict", "/feedback"}:
            return self.send_json(404,{"error":"Unknown route"})
        if not self.headers.get_content_type().startswith("image/"):
            return self.send_json(415,{"error":"Send raw image bytes with an image/* Content-Type"})
        try:
            size=int(self.headers.get("Content-Length","0"))
        except ValueError:
            size=0
        if not 0<size<=32*1024*1024:
            return self.send_json(413,{"error":"Expected a nonempty image up to 32 MiB"})
        try:
            payload=self.rfile.read(size)
            if len(payload)!=size:
                raise ValueError("Incomplete upload")
            if self.path == "/feedback":
                model = self.server.classifier.classifier
                predicted = self.headers.get("X-Predicted-Class")
                label = self.headers.get("X-Actual-Class")
                if predicted not in model.classes or label not in model.classes:
                    return self.send_json(400, {"error":"请选择有效的螺丝型号"})
                token = self.headers.get("X-Prediction-Token", "")
                if not hmac.compare_digest(token.encode(), self.receipt(payload, predicted).encode()):
                    return self.send_json(409, {"error":"识别记录已失效，请重新识别这张照片后再提交"})
                with self.server.predict_lock:
                    result = model.learn(payload, predicted, label)
            else:
                with self.server.predict_lock:
                    result=self.server.classifier.predict(payload, diameter_mm=self.headers.get("X-Shank-Diameter-MM"))
                result["feedback_token"] = self.receipt(payload, result["class_name"])
        except (ValueError,OSError,Image.DecompressionBombError):
            return self.send_json(400,{"error":"图片无效，或卡尺外径不是有限正数"})
        except (RuntimeError, sqlite3.Error, ArithmeticError):
            self.log_error("识别或反馈训练失败")
            return self.send_json(500,{"error":"本次操作失败，请重试"})
        self.send_json(200,result)


def serve(args):
    url = f"http://127.0.0.1:{args.port}/"
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError:
        if not args.no_browser and args.weights == DEFAULT_WEIGHTS and args.feedback_dir is None:
            try:
                with urlopen(url, timeout=2) as response:
                    if response.headers.get("X-Screw-Upload") == "single-file-1":
                        webbrowser.open(url)
                        return
            except OSError:
                pass
        raise SystemExit(f"无法使用端口 {args.port}，请用 --port 指定其他端口。")
    with server:
        server.classifier = ScrewIdentifier(args.weights, feedback_dir=args.feedback_dir)
        server.predict_lock = Lock()
        server.feedback_key = secrets.token_bytes(32)
        print(f"照片识别页面 {url}", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.classifier.classifier.feedback.db.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="本项目螺丝分类权重")
    p.add_argument("--output", type=Path, help="另存带识别结果的对比图")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--diameter-mm", type=float, help="卡尺实测本颗螺丝的螺纹外径")
    group.add_argument("--ticks", nargs=5, type=float, metavar=("X1","Y1","X2","Y2","MM"))
    group.add_argument("--tick-mm", type=float, help="照片中钢尺相邻刻线的已知间距")
    group.add_argument("--px-per-mm", type=float, help="固定机位在工件平面的实测像素/毫米")
    p.add_argument("--shank", nargs=4, type=float, metavar=("X1","Y1","X2","Y2"),
                   help="原图中垂直于杆轴的两侧牙顶坐标")
    p.add_argument("--feedback-dir", type=Path, help="纠错数据保存目录")
    p.add_argument("--port", type=int, default=8013)
    p.add_argument("--no-browser", action="store_true", help="手动打开浏览器")
    args = p.parse_args()
    if not args.weights.is_file():
        p.error(f"找不到权重 {args.weights}，请用 --weights 指定原有 screw.pt 文件")
    if args.image is None:
        return serve(args)
    if args.output and args.output.resolve() in {args.image.resolve(), args.weights.resolve()}:
        p.error("对比图不能覆盖输入图片或权重")
    try:
        result = ScrewIdentifier(args.weights, feedback_dir=args.feedback_dir).predict(args.image, diameter_mm=args.diameter_mm,
                    ticks=args.ticks, tick_mm=args.tick_mm, px_per_mm=args.px_per_mm, shank=args.shank)
        if args.output:
            save_comparison(args.image, result, args.output)
        print(result["class_name"])
    except (OSError, ValueError, RuntimeError, Image.DecompressionBombError) as exc:
        p.exit(1, f"识别失败：{exc}\n")


if __name__ == "__main__":
    main()
