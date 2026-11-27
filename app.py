"""启动：streamlit run app.py。识别服务随应用按需启动，依赖见 requirements.txt。"""

import base64
import math
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

import backend_client


SPECS = ("M4", "M5", "M6", "M8", "M10")
DEFAULT_PARAMS = {"clip_limit": 2.0, "tile_size": 8, "canny_range": (50, 150)}
TAB_KEY = "vision_workspace_v1"
MAX_IMAGE_BYTES = 32 * 1024 * 1024
IMAGE_MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
CALIBRATION_LABELS = {
    "caliper": "卡尺实测",
    "ruler_endpoints": "钢尺端点",
    "ruler_ticks": "钢尺刻线",
    "calibrated_camera": "固定机位标定",
}
# 余弦相似度的复核提示线，不是概率阈值。
LOW_REFERENCE_SIMILARITY = 0.50

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


def jpeg_thumb(image, size):
    thumb = image.copy()
    thumb.thumbnail(size, Image.Resampling.LANCZOS)
    buf = BytesIO()
    thumb.convert("RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def reset_params():
    st.session_state.update(DEFAULT_PARAMS)


def render_waveform(slot, clip, tile, low, high):
    # 参数驱动的示意动画。浏览器移动两段相同的波形，循环时可无缝衔接。
    peak = 24 - 1.8 * (clip - 1)
    trough = 52 + 10 * (high - low) / 255
    period = 4.2 - 0.06 * tile
    trace = (f"M0 42 H18 Q23 42 26 38 T34 42 H46 L50 46 L55 {peak:.1f} "
             f"L61 {trough:.1f} L67 41 H76 Q82 31 88 38 T98 42 H120")
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 78" '
           f'preserveAspectRatio="none"><path d="{trace}" fill="none" stroke="#67e7bd" '
           f'stroke-width="2" stroke-linejoin="round"/></svg>')
    tile_url = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
    slot.html(f"""
    <style>
      .vision-scope {{background-color:#0c201f; border:1px solid #2a4c47; border-radius:8px;
        padding:10px 12px 0; overflow:hidden; color:#8ee4c9}}
      .vision-scope small {{font:12px/1.4 system-ui,sans-serif; letter-spacing:.06em}}
      .vision-scope-grid {{overflow:hidden;
        background-image:linear-gradient(#26443e70 1px,transparent 1px),
          linear-gradient(90deg,#26443e70 1px,transparent 1px); background-size:16px 16px}}
      .vision-scope-trace {{height:78px; width:150%; background:url("{tile_url}") repeat-x;
        background-size:calc(100% / 3) 100%; animation:vision-sweep {period:.2f}s linear infinite}}
      @keyframes vision-sweep {{to {{transform:translateX(calc(-100% / 3))}}}}
      @media (prefers-reduced-motion:reduce) {{.vision-scope-trace {{animation:none}}}}
    </style>
    <div class="vision-scope"><small>参数波形 · 示意</small>
      <div class="vision-scope-grid" aria-hidden="true"><div class="vision-scope-trace"></div></div>
    </div>""")


def parameter_controls(state):
    saved = state.setdefault("params", DEFAULT_PARAMS.copy())
    for key, value in saved.items():
        st.session_state.setdefault(key, value)
    scope = st.empty()
    clip = st.slider("CLAHE 对比度限制", 1.0, 10.0, step=0.5, key="clip_limit")
    tile = st.slider("CLAHE 分块数（N × N）", 2, 32, step=2, key="tile_size")
    low, high = st.slider("Canny 低 / 高阈值", 0, 255, key="canny_range")
    render_waveform(scope, clip, tile, low, high)
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
    # 这里生成的增强图只用于展示，绝不发送给后端、也不参与识别。
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
    gray.image(gray_view, caption="CLAHE 灰度图（预览）", width="stretch")
    edge.image(edge_view, caption=f"Canny 边缘图（预览） · {ratio:.2f}%", width="stretch")


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
                  help="本会话识别流程状态，下一次成功识别可恢复在线。")
    throughput.metric("当前识别吞吐量", f"{rate:.1f} 个/分钟", border=True,
                      help="按最近一次成功单图的处理耗时估算，空闲 60 秒后归零。")
    cpu.metric("端侧算力占用", "—" if stats["cpu"] is None else f'{stats["cpu"]:.1f}%',
               border=True, help="应用运行主机的系统 CPU 实测值。云端部署时采样云主机。")
    total.metric("累计处理", f'{stats["total"]} 件', border=True,
                 help="当前会话成功完成识别的图片数，清空历史不会清零。")


def parse_diameter(raw):
    """侧栏输入允许为空；只有有限正数才作为 X-Shank-Diameter-MM 上传。"""
    if raw is None or not str(raw).strip():
        return None, None
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return None, "卡尺外径必须是有限正数，留空则不上传。"
    if not math.isfinite(value) or value <= 0:
        return None, "卡尺外径必须是有限正数，留空则不上传。"
    return value, None


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
        item = {"id": file.file_id, "name": file.name, "result": None, "feedback_token": None}
        try:
            data = file.getvalue()
            if not data:
                raise ValueError("图片内容为空")
            if len(data) > MAX_IMAGE_BYTES:
                raise ValueError("图片超过 32 MiB，请压缩后重试")
            with Image.open(BytesIO(data)) as im:
                if im.format not in IMAGE_MIME:
                    raise ValueError("图片格式不支持")
                item["mime"] = IMAGE_MIME[im.format]
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


def result_message(result):
    cls = result["class_name"]
    similarity = float(result["catalog_similarity"])
    bin_text = result.get("bin")
    bin_text = "—" if bin_text is None else str(bin_text)
    message = f"识别为 {cls}，参考相似度 {similarity:.3f}，分箱 {bin_text}。"
    diameter = result.get("diameter_mm")
    if diameter is not None:
        message = f"识别为 {cls}，参考相似度 {similarity:.3f}，分箱 {bin_text}；" \
                  f"卡尺外径 {float(diameter):.2f} mm。"
    return message


def run_inference(state, items, diameter_mm, slot):
    stats = state["stats"]
    failed = False
    if not items:
        stats["status"] = "异常"
        log_event(state, "系统", "识别中止：请先上传图片或使用摄像头拍照。")
        render_log(slot, state)
        return
    try:
        backend_client.health()
    except backend_client.BackendError as exc:
        stats["status"] = "异常"
        log_event(state, "系统", str(exc))
        st.error(str(exc))
        render_log(slot, state)
        return
    for item in items:
        name = item["name"]
        item["result"] = None
        item["feedback_token"] = None
        for key in ("feedback_wrong", "feedback_done", "feedback_count", "feedback_error", "feedback_label"):
            item.pop(key, None)
        try:
            if "error" in item:
                raise ValueError(item["error"])
            log_event(state, f"算法推理 · {name}", "开始识别，等待本机服务返回结果。")
            render_log(slot, state)
            started = time.perf_counter()
            payload = backend_client.predict(item["data"], item["mime"], diameter_mm)
            duration = time.perf_counter() - started
            similarity = payload.get("catalog_similarity")
            if similarity is None:
                raise backend_client.BackendError(backend_client.INVALID_RESULT)
            similarity = float(similarity)
            # token 单独保存，不写入历史表或日志，也不在结果区展示。
            item["result"] = {k: v for k, v in payload.items() if k != "feedback_token"}
            item["feedback_token"] = payload.get("feedback_token")
            diameter = payload.get("diameter_mm")
            record = {
                "时间": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S"),
                "缩略图": "data:image/jpeg;base64," + base64.b64encode(item["preview"]).decode("ascii"),
                "文件名": name,
                "识别结果": payload["class_name"],
                "参考库相似度": f"{similarity:.3f}",
                "分箱": "—" if payload.get("bin") is None else str(payload["bin"]),
            }
            if diameter is not None:
                record["卡尺外径"] = f"{float(diameter):.2f} mm"
            state["history"].insert(0, record)
            stats.update(total=stats["total"] + 1, duration=duration, last_success=time.monotonic())
            log_event(state, f"算法推理 · {name}", f"{result_message(item['result'])} 耗时 {duration:.3f} 秒。")
            if similarity < LOW_REFERENCE_SIMILARITY:
                log_event(state, f"算法推理 · {name}", "参考相似度较低，建议复核。")
        except backend_client.BackendUnavailable as exc:
            failed = True
            stats["status"] = "异常"
            st.error(str(exc))
            log_event(state, "系统", f"{name}：{exc}")
        except Exception as exc:
            # 单张失败仍继续处理其他图片；失败项不增加计数，也不产生成功记录。
            failed = True
            stats["status"] = "异常"
            st.warning(backend_client.INVALID_RESULT)
            log_event(state, "系统", f"{name}：识别失败（{exc}）。")
        render_log(slot, state)
    if items:
        stats["status"] = "异常" if failed else "在线"
    render_log(slot, state)


def submit_feedback(item, actual, state):
    predicted = item["result"]["class_name"]
    try:
        payload = backend_client.feedback(
            item["data"], item["mime"], item.get("feedback_token"), predicted, actual
        )
    except backend_client.BackendError as exc:
        item["feedback_error"] = str(exc)
        if isinstance(exc, backend_client.FeedbackConflict):
            item["feedback_token"] = None
        log_event(state, "纠错反馈", f'{item["name"]}：{exc}')
        return
    item["feedback_done"] = True
    item["feedback_label"] = actual
    item["feedback_count"] = payload.get("sample_count")
    item["feedback_error"] = None
    change = "已按该型号更新" if payload.get("changed") else "该型号已记录过"
    log_event(state, "纠错反馈",
              f'{item["name"]}：{change}（{actual}），样本累计 {payload.get("sample_count")} 张。')


def draw_center(image, center_fraction):
    """只在原图上叠加定位十字/圆点；后端不提供检测框，这里也不画框。"""
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    x = float(center_fraction[0]) * marked.width
    y = float(center_fraction[1]) * marked.height
    radius = max(6, min(marked.size) // 40)
    width = max(2, min(marked.size) // 200)
    color = (18, 120, 104)
    draw.line((x - radius, y, x + radius, y), fill=color, width=width)
    draw.line((x, y - radius, x, y + radius), fill=color, width=width)
    dot = max(2, width + 1)
    draw.ellipse((x - dot, y - dot, x + dot, y + dot), fill=color)
    return marked


def render_result(item, state):
    result = item["result"]
    cls = result["class_name"]
    similarity = float(result["catalog_similarity"])
    st.markdown(f"#### {cls}")
    bin_text = result.get("bin")
    bin_text = "—" if bin_text is None else str(bin_text)
    st.caption(f"规格 {cls} / 分箱 {bin_text}")
    st.metric("参考库相似度", f"{similarity:.3f}")
    if similarity < LOW_REFERENCE_SIMILARITY:
        st.warning("参考相似度较低，建议复核")
    ranking = result.get("ranking") or []
    if ranking:
        st.caption("参考库排名（前三名）")
        st.table([{"型号": str(row.get("class_name") or "—"),
                   "参考库相似度": f'{float(row.get("score") or 0):.3f}'}
                  for row in ranking[:3]])
    diameter = result.get("diameter_mm")
    if diameter is not None:
        st.write(f"卡尺外径：{float(diameter):.2f} mm")
        calibration = result.get("calibration")
        label = CALIBRATION_LABELS.get(calibration, calibration) if calibration else "—"
        st.caption(f"标定方式：{label}")
    center = result.get("center_fraction")
    if center:
        with Image.open(BytesIO(item["data"])) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
        st.image(draw_center(image, center), caption="定位中心（仅供查看）", width="stretch")

    token = item.get("feedback_token")
    if not token:
        return
    if item.get("feedback_done"):
        st.success(f'反馈已记录为 {item.get("feedback_label")}，样本累计 {item.get("feedback_count")} 张。')
        return
    st.caption("这个型号对吗？")
    correct, wrong = st.columns(2)
    if correct.button("识别正确", key=f"fb_ok_{item['id']}"):
        submit_feedback(item, cls, state)
        st.rerun()
    if wrong.button("识别有误", key=f"fb_wrong_{item['id']}"):
        item["feedback_wrong"] = True
        st.rerun()
    if item.get("feedback_wrong"):
        actual = st.selectbox("正确型号", SPECS, key=f"fb_actual_{item['id']}")
        if st.button("提交纠错", key=f"fb_submit_{item['id']}"):
            submit_feedback(item, actual, state)
            st.rerun()
    if item.get("feedback_error"):
        st.warning(item["feedback_error"])


def detection_page():
    state = st.session_state.workbench
    st.caption("型号和参考库相似度由识别服务根据原图计算。CLAHE 和 Canny 用于预览。")
    with st.sidebar:
        st.subheader("预处理参数调节")
        params = parameter_controls(state)
        st.subheader("尺寸输入（可选）")
        raw_diameter = st.text_input("卡尺实测外径 (mm)", value="", key="caliper_diameter",
                                     help="留空则不向后端上传尺寸；填写有限正数才生效。")
        diameter_mm, diameter_error = parse_diameter(raw_diameter)
        if diameter_error:
            st.warning(diameter_error)
    acquisition, results = st.columns([1, 2])
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
        for item in items:
            if "error" in item:
                st.error(f'{item["name"]}：{item["error"]}')
                continue
            try:
                render_comparison(item, params, state)
            except (cv2.error, ValueError, OSError, Image.DecompressionBombError):
                st.error(f'{item["name"]}：无法生成预处理图像。')
        start = st.button("开始识别", type="primary", width="stretch")
    with results.container(border=True):
        st.subheader("识别结果")
        if not items:
            st.caption("上传或拍照后，在这里查看识别结果。")
        for item in items:
            if "error" in item or not item.get("result"):
                continue
            st.markdown(f"**{item['name']}**")
            render_result(item, state)
            st.divider()
    st.subheader("实时工业日志")
    slot = st.empty()
    render_log(slot, state)
    if start:
        if diameter_error:
            st.warning(diameter_error)
        else:
            with st.spinner("正在调用本机识别服务..."):
                run_inference(state, items, diameter_mm, slot)
            st.rerun()

    st.divider()
    title, clear = st.columns([4, 1])
    title.subheader("历史检测记录")
    if clear.button("清空记录", disabled=not state["history"], width="stretch"):
        state["history"].clear()
        st.rerun()
    if state["history"]:
        allowed = {"时间", "缩略图", "文件名", "识别结果", "参考库相似度", "分箱", "卡尺外径"}
        show_diameter = any("卡尺外径" in row for row in state["history"])
        rows = []
        for row in state["history"]:
            clean = {k: v for k, v in row.items() if k in allowed}
            if show_diameter:
                clean.setdefault("卡尺外径", "—")
            rows.append(clean)
        st.dataframe(rows, hide_index=True, width="stretch", row_height=64,
                     column_config={"缩略图": st.column_config.ImageColumn("缩略图", width="small"),
                                    "时间": st.column_config.TextColumn("时间（北京时间）"),
                                    "分箱": st.column_config.TextColumn("规格分箱")})
    else:
        st.caption("暂无检测记录。")


state = workspace()
init_monitor(state)
st.title("智造慧眼 · 螺丝视觉检测")
metrics_panel()
detection_page()
