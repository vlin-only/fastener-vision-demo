"""桌面总后端中的定位、描述子、型号库、标定与纠错训练实现。"""
import io
import sqlite3
from dataclasses import dataclass
from hashlib import file_digest, sha256
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.nn.functional import normalize

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
        left, top, width, height = stats[idx, :4]
        box = (int(left*w/GRID), int(top*h/GRID),
               min(w, int(np.ceil((left+width)*w/GRID))), min(h, int(np.ceil((top+height)*h/GRID))))
        groups.append((merit, tuple(center), span, angle, float(strength.max()), box))
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
            digest = file_digest(stream, "sha256").hexdigest()
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


def extract_contour(rgb, params):
    """CLAHE 增强后进行高斯平滑、Canny 和闭运算，返回增强图、边缘掩模和候选轮廓。"""
    clip, tile, low, high = params
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    m = min(h, w)
    enhanced = cv2.createCLAHE(clip, (tile, tile)).apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), .8)
    edges = cv2.Canny(blurred, low, high)
    k = max(3, round(m / 115)) | 1
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
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
    return enhanced, edges, sorted(regions, key=lambda r:r[0], reverse=True)


def square_crop(rgb, center, span, angle=0):
    matrix=cv2.getRotationMatrix2D(center,angle,SIDE/max(1,span))
    matrix[:,2]+=np.array([SIDE/2,SIDE/2])-center
    return cv2.warpAffine(rgb,matrix,(SIDE,SIDE),flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def photo_views(rgb, locations, regions):
    h,w=rgb.shape[:2]
    center=(w/2,h/2)
    if locations:
        _,focus,long,angle = locations[0][:4]
    else:
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


def descriptors(encoder, rgb, device, locations, regions):
    views=photo_views(rgb, locations, regions)
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


BINS = {"M4": 1, "M5": 2, "M6": 3, "M8": 4, "M10": 5}


class ScrewClassifier:
    def __init__(self, weights, device="cpu", feedback_dir=None):
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
