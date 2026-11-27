"""启动：streamlit run app.py
依赖：streamlit、Pillow、psutil、streamlit-browser-session-storage、
opencv-python-headless 和 numpy。
"""

import base64
import random
import secrets
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from io import BytesIO
from importlib.resources import files

import cv2
import numpy as np
import streamlit as st
import psutil
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageOps


SPECS = ("M4", "M5", "M6", "M8", "M10")
DEFAULT_PARAMS = {"clip_limit": 2.0, "tile_size": 8, "canny_range": (50, 150)}
TAB_KEY = "vision_workspace_v1"
st.set_page_config(page_title="智造慧眼 · 螺丝检测", layout="wide", page_icon="⚙️")
# 通过 Streamlit 的 Python 组件接口仅访问本应用的一项标签页存储。
tab_storage = components.declare_component(
    "vision_tab_storage", path=str(files("streamlit_session_browser_storage") / "frontend" / "build")
)


@st.cache_resource(ttl=3600, max_entries=32, show_spinner=False)
def refresh_snapshot(tab_id):
    # 标签页仅持有随机恢复编号，图片与历史临时保存在服务端内存。
    return {"inputs": {}, "source": "上传图片", "history": []}


def workspace():
    with st.container():
        if "workbench" not in st.session_state:
            # None 表示浏览器尚未响应；空串才表示首次访问，避免 F5 时覆盖旧编号。
            saved = tab_storage(method="getItem", itemKey=TAB_KEY, key="read_tab",
                                sessionStrageOptions={"defaultValue": {TAB_KEY: ""}}, default=None)
            if saved is None:
                st.stop()
            st.session_state.tab_id = saved or secrets.token_urlsafe(32)
            st.session_state.workbench = refresh_snapshot(st.session_state.tab_id)
        tab_storage(method="setItem", itemKey=TAB_KEY, itemValue=st.session_state.tab_id,
                    key="save_tab")
    return st.session_state.workbench


def mock_detect(image):
    """模拟单图推理延迟和模型输出；返回原图像素坐标。"""
    time.sleep(1)
    width, height = image.size
    w = max(1, width * random.randint(35, 70) // 100)
    h = max(1, height * random.randint(35, 70) // 100)
    x, y = random.randint(0, width - w), random.randint(0, height - h)
    return {"model": random.choice(SPECS), "confidence": random.randint(45, 99) / 100,
            "boxes": [{"x": x, "y": y, "w": w, "h": h}]}


def draw_boxes(image, result):
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    for box in result["boxes"]:
        x, y, w, h = (box[key] for key in ("x", "y", "w", "h"))
        draw.rectangle((x, y, x + w - 1, y + h - 1), outline="#ff3434",
                       width=max(1, min(image.size) // 160))
    return marked


def jpeg_thumb(image, size):
    thumb = image.copy()
    thumb.thumbnail(size, Image.Resampling.LANCZOS)
    buf = BytesIO()
    thumb.convert("RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def reset_params():
    st.session_state.update(DEFAULT_PARAMS)


def parameter_controls(state):
    saved = state.setdefault("params", DEFAULT_PARAMS.copy())
    for key, value in saved.items():
        st.session_state.setdefault(key, value)
    clip = st.slider("CLAHE 对比度限制", 1.0, 10.0, step=0.5, key="clip_limit")
    tile = st.slider("CLAHE 分块数（N × N）", 2, 32, step=2, key="tile_size")
    low, high = st.slider("Canny 低 / 高阈值", 0, 255, key="canny_range")
    st.button("恢复默认参数", on_click=reset_params)
    saved.update(clip_limit=clip, tile_size=tile, canny_range=(low, high))
    return clip, tile, low, high


def preprocess(image, params):
    clip, tile, low, high = params
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    return Image.fromarray(enhanced), cv2.Canny(blurred, low, high)


def render_comparison(item, params, state):
    # 仅缓存当前文件、当前参数对应的预览；指标刷新和普通页面重绘不重复处理。
    if item.get("preview_params") != params:
        with Image.open(BytesIO(item["data"])) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
        enhanced, edges = preprocess(image, params)
        edge_view = Image.fromarray(edges)
        edge_view.thumbnail((720, 540), Image.Resampling.NEAREST)
        buf = BytesIO()
        edge_view.save(buf, format="PNG")
        item["processed_views"] = (jpeg_thumb(enhanced, (720, 540)), buf.getvalue(),
                                   100 * np.count_nonzero(edges) / edges.size)
        item["preview_params"] = params
        clip, tile, low, high = params
        log_event(state, "图像处理", f'{item["name"]}：CLAHE {clip:g} / {tile}×{tile}，Canny {low}/{high}。')
    gray_view, edge_view, ratio = item["processed_views"]
    st.text(item["name"])
    original, gray, edge = st.columns(3)
    original.image(item.get("preview") or item["data"], caption="原图", width="stretch")
    gray.image(gray_view, caption="CLAHE 灰度图", width="stretch")
    edge.image(edge_view, caption=f"边缘图 · {ratio:.2f}%", width="stretch")


def log_event(state, subsystem, message):
    clock = datetime.now(timezone(timedelta(hours=8))).strftime("%H:%M:%S")
    state["events"].appendleft(f"[{clock}] [{subsystem}] {message}")


def init_monitor(state):
    if "stats" not in state:
        state["stats"] = {"status": "在线", "total": len(state["history"]), "duration": 0.0,
                          "last_success": None, "cpu": None}
        state["events"] = deque(maxlen=200)
        log_event(state, "系统", "视觉传感器待机中，等待图像输入。")


@st.fragment(run_every=2)
def metrics_panel():
    """只刷新指标区域：不触发上传、推理、日志写入和历史记录。"""
    stats = st.session_state.workbench["stats"]
    try:
        stats["cpu"] = psutil.cpu_percent(interval=0.1)
    except (OSError, psutil.Error, NotImplementedError):
        stats["cpu"] = None
    rate = 0.0
    if stats["last_success"] is not None and stats["duration"] > 0:
        idle = time.monotonic() - stats["last_success"]
        # 最近一次成功单图耗时（秒）换算为每分钟件数；空闲一分钟后显示 0。
        if idle < 60:
            rate = 60 / stats["duration"]
    status, throughput, cpu, total = st.columns(4)
    status.metric("设备状态", stats["status"], border=True,
                  help="本会话检测流程状态，下一次成功识别可恢复在线。")
    throughput.metric("当前识别吞吐量", f"{rate:.1f} 个/分钟", border=True,
                      help="按最近一次成功单图的处理耗时估算，空闲 60 秒后归零。")
    cpu.metric("端侧算力占用", "—" if stats["cpu"] is None else f'{stats["cpu"]:.1f}%',
               border=True, help="应用运行主机的系统 CPU 实测值。云端部署时采样云主机。")
    total.metric("累计处理", f'{stats["total"]} 件', border=True,
                 help="当前会话成功完成识别的图片数，清空历史不会清零。")


def update_input(source, key):
    state = st.session_state.workbench
    value = st.session_state[key]
    files_now = value if isinstance(value, list) else ([value] if value is not None else [])
    # Streamlit 添加、删除多图时会整体重发文件列表，用上传 ID 保留已有的结果。
    old = state["inputs"].get(source, [])
    existing = {i["id"]: i for i in old if "id" in i} if isinstance(old, list) else {}
    items = []
    for file in files_now:
        if file.file_id in existing:
            items.append(existing[file.file_id])
            continue
        item = {"id": file.file_id, "name": file.name, "result": None}
        try:
            data = file.getvalue()
            with Image.open(BytesIO(data)) as im:
                if im.format not in ("JPEG", "PNG"):
                    raise ValueError("图片格式不支持")
                original = ImageOps.exif_transpose(im)
                item["preview"] = jpeg_thumb(original, (600, 400))
                width, height = original.size
            item["data"] = data
            log_event(state, "视觉系统", f"采集触发：{file.name}，尺寸 {width} × {height}。")
        except (OSError, ValueError, Image.DecompressionBombError):
            item["error"] = "图像读取失败，请选择有效的 JPG / PNG 图片。"
            state["stats"]["status"] = "异常"
            log_event(state, "系统", f"{file.name}：{item['error']}")
        items.append(item)
    state["inputs"][source] = items


def render_log(slot, state):
    slot.code("\n".join(state["events"]), language=None, height=185, wrap_lines=True)


def detect_batch(state, items, slot, params):
    stats = state["stats"]
    failed = False
    if not items:
        stats["status"] = "异常"
        log_event(state, "系统", "识别中止：请先上传图片或使用摄像头拍照。")
    for item in items:
        name = item["name"]
        item["result"] = None
        try:
            if "error" in item:
                raise ValueError(item["error"])
            log_event(state, "算法推理", f"{name}：开始模拟识别。")
            render_log(slot, state)
            # 每张图独立计时，包含解码、1 秒模拟推理和生成标注缩略图。
            started = time.perf_counter()
            with Image.open(BytesIO(item["data"])) as im:
                image = ImageOps.exif_transpose(im).convert("RGB")
            enhanced, _ = preprocess(image, params)
            result = mock_detect(enhanced)
            thumb = jpeg_thumb(draw_boxes(image, result), (160, 120))
            duration = time.perf_counter() - started
            confidence = f'{result["confidence"]:.0%}'
            record = {"时间": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S"),
                      "缩略图": "data:image/jpeg;base64," + base64.b64encode(thumb).decode("ascii"),
                      "文件名": name, "识别结果": result["model"], "置信度": confidence}
        except Exception as exc:
            # 单张失败仍继续处理其他图片；失败项不增加计数，也不产生成功记录。
            failed = True
            stats["status"] = "异常"
            log_event(state, "系统", f"{name}：识别失败（{exc}）。")
        else:
            item["result"] = result
            state["history"].insert(0, record)
            stats.update(total=stats["total"] + 1, duration=duration, last_success=time.monotonic())
            log_event(state, "算法推理", f'{name}：模拟匹配 {result["model"]} 螺钉，'
                      f'置信度 {confidence}，耗时 {duration:.3f} 秒。')
        render_log(slot, state)
    if items:
        stats["status"] = "异常" if failed else "在线"
    render_log(slot, state)


def detection_page():
    state = st.session_state.workbench
    st.caption("型号、置信度和检测框为模拟输出。吞吐量按实测单图耗时估算，CPU 每 2 秒采样运行主机。")
    controls, acquisition, results = st.columns([1, 1, 2])
    with controls.container(border=True):
        st.subheader("预处理参数调节")
        params = parameter_controls(state)
    with acquisition.container(border=True):
        st.subheader("图像输入")
        if "source_picker" not in st.session_state:
            st.session_state.source_picker = state["source"]
        source = st.radio("图像来源", ["上传图片", "摄像头拍照"], key="source_picker", horizontal=True)
        state["source"] = source
        if source == "上传图片":
            st.file_uploader("选择一张或多张 JPG / PNG 图片", type=["jpg", "jpeg", "png"],
                             accept_multiple_files=True, key="upload", on_change=update_input, args=(source, "upload"))
        else:
            st.camera_input("拍摄螺丝照片", key="camera", on_change=update_input, args=(source, "camera"))
        items = state["inputs"].get(source, [])
        if isinstance(items, dict):  # 兼容更新前已打开的单图会话。
            items = [items]
            state["inputs"][source] = items
        start = st.button("开始识别", type="primary", width="stretch")
    with results.container(border=True):
        st.subheader("图像预览与检测结果")
        if not items:
            st.caption("上传或拍照后，在这里查看图像。")
        for item in items:
            if "error" in item:
                st.error(f'{item["name"]}：{item["error"]}')
                continue
            try:
                render_comparison(item, params, state)
            except (cv2.error, ValueError, OSError, Image.DecompressionBombError):
                st.error(f'{item["name"]}：无法生成预处理图像。')
                continue
            if not item.get("result"):
                continue
            with Image.open(BytesIO(item["data"])) as im:
                image = ImageOps.exif_transpose(im).convert("RGB")
            result = item["result"]
            st.image(draw_boxes(image, result), caption="最近一次模拟识别", width="stretch")
            label = f'识别为 {result["model"]} 螺丝，置信度：{result["confidence"]:.0%}'
            if result["confidence"] < 0.6:
                st.warning(f"{label}（低置信度，请复核）")
            elif result["confidence"] > 0.8:
                st.success(label)
            else:
                st.info(label)
    st.subheader("实时工业日志")
    slot = st.empty()
    render_log(slot, state)
    if start:
        with st.spinner("AI识别中..."):
            detect_batch(state, items, slot, params)
        st.rerun()

    st.divider()
    title, clear = st.columns([4, 1])
    title.subheader("历史检测记录")
    if clear.button("清空记录", disabled=not state["history"], width="stretch"):
        state["history"].clear()
        st.rerun()
    if state["history"]:
        st.dataframe(state["history"], hide_index=True, width="stretch", row_height=64,
                     column_config={"缩略图": st.column_config.ImageColumn("缩略图", width="small"),
                                    "时间": st.column_config.TextColumn("时间（北京时间）")})
    else:
        st.caption("暂无检测记录。")


state = workspace()
init_monitor(state)
st.title("智造慧眼 · 螺丝视觉检测")
metrics_panel()
detection_page()
