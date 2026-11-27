"""内存图像识别接口。上传字节仅用于样本指纹和用户明确提交的纠错。"""
import io
from threading import RLock

import cv2
import numpy as np
import torch
from torch.nn.functional import normalize

import screw_core as core


class Resources:
    def __init__(self, weights, feedback_dir):
        # screw.pt 同时包含 DINOv2、图像块定位器和训练参考库。更换时须保持特征维度及类别一致。
        self.model = core.ScrewClassifier(weights, feedback_dir=feedback_dir)
        if set(self.model.classes) != set(core.BINS):
            raise ValueError("权重类别与分箱表不一致")
        self.lock = RLock()
        self.reload_adapter()

    def reload_adapter(self):
        row = self.model.feedback.db.execute("SELECT parameters FROM adapter WHERE id=1").fetchone()
        self.adapter = None
        if row:
            with np.load(io.BytesIO(row[0]), allow_pickle=False) as saved:
                self.adapter = (saved["features"], saved["coefficients"])


def locate_screw(model, rgb):
    # 图像块的线性响应决定候选掩模；连通域和加权 PCA 给出中心、主轴和定位框。
    return core.locate(model.model, model.locator, rgb, model.device)


def extract_dinov2_feature(model, rgb, locations, regions):
    features = {
        mode: core.descriptors(model.model, rgb, model.device,
                               locations if mode == "learned" else [], regions)
        for mode in model.mix
    }
    return (normalize(core.signature(features, model.mix), dim=0),
            core.classification_scores(features, model.heads, model.mix))


def match_model(model, query, raw):
    catalog = core.nearest_class_scores(query, model.reference_groups)
    learned = model.feedback.similarities(query, model.classes)
    scores = torch.maximum(catalog, learned).clamp(-1, 1)
    if not len(scores) or not torch.isfinite(scores).all():
        return None
    known = model.feedback.confirmed_label(raw)
    if known in model.classes:
        scores[model.classes.index(known)] = 1.0
    index = max(range(len(scores)), key=lambda i: (
        float(scores[i]), model.classes[i] == known, float(learned[i])))
    return model.classes[index], float(scores[index])


def estimate_diameter(bgr, center, method, value):
    if method == "未标定":
        return None, "未标定"
    if not np.isfinite(value) or value <= 0:
        raise ValueError("标定值必须为有限正数")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if method == "钢尺刻线":
        exclusion, pitch = core.ruler_region(gray)
        if pitch is None:
            return None, "未找到可靠的钢尺刻线"
        scale = pitch / value
    else:
        exclusion = np.zeros_like(gray)
        scale = value
    x, y = center
    matches = [p for p in core.components(gray, exclusion) if p.diameter is not None
               and p.box[0] <= x < p.box[2] and p.box[1] <= y < p.box[3]]
    if len(matches) != 1:
        return None, "未可靠分离杆段"
    return matches[0].diameter / scale, method


def recognize(resources, bgr, raw, params, calibration, log):
    """输入 OpenCV BGR 矩阵。几何测量使用原始分辨率，模型视图限定在 900 像素内。"""
    height, width = bgr.shape[:2]
    if min(width, height) < 12:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    factor = min(1, 900 / max(width, height))
    if factor < 1:
        rgb = cv2.resize(rgb, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    model = resources.model
    with resources.lock:
        log("正在定位螺丝，计算图像块响应、连通区域和主轴…")
        locations = locate_screw(model, rgb)
        if not locations:
            return None
        log("正在进行 CLAHE 增强、高斯滤波、Canny 检测和形态学闭运算…")
        enhanced, edges, regions = core.extract_contour(rgb, params)
        _, (cx, cy), _, _, _, box = locations[0]
        x1, y1, x2, y2 = box
        if not edges[y1:y2, x1:x2].any():
            return None
        log("正在提取 DINOv2 多视图特征…")
        query, base_scores = extract_dinov2_feature(model, rgb, locations, regions)
        log("正在比对参考库和已确认样本，计算最近邻…")
        match = match_model(model, query, raw)
    if match is None:
        return None
    label, similarity = match
    log("正在核对标定信息与螺杆直径…")
    center = cx * width / rgb.shape[1], cy * height / rgb.shape[0]
    diameter, reason = estimate_diameter(bgr, center, *calibration)
    bin_number = core.BINS[label]
    size = f"{diameter:.2f} mm" if diameter is not None else reason
    # 框来自定位器的实际连通区域，坐标与增强图在同一像素空间内。
    annotated = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(annotated, (x1, y1), (x2-1, y2-1), (0, 0, 255), 2)
    text = f"{label} | BIN {bin_number} | D: " + (size if diameter is not None else "N/A")
    cv2.rectangle(annotated, (0, 0), (min(rgb.shape[1]-1, 320), 25), (30, 30, 30), -1)
    cv2.putText(annotated, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1, cv2.LINE_AA)
    log(f"检测到型号 {label}，杆径：{size}，匹配 {bin_number} 号箱。")
    return dict(label=label, similarity=similarity, bin=bin_number, size=size, image=annotated,
                query=query, base_scores=base_scores)


def correct_and_update(resources, raw, result, actual):
    model = resources.model
    if actual not in model.classes:
        raise ValueError("请选择参考库中的型号")
    with resources.lock:
        # 复用本次提取的特征。一次事务更新参考样本及核岭参数，日常分类仍使用 1-NN。
        updated = model.feedback.learn(raw, result["label"], actual, result["query"],
                                       result["base_scores"], model.classes)
        if updated["changed"]:
            resources.reload_adapter()
        return updated
