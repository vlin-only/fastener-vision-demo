"""智造慧眼：螺丝型号库和模拟检测工作台。运行：streamlit run app.py"""

import base64
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from io import BytesIO
from importlib.resources import files
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent
LIBRARY = ROOT / "assets" / "screws"
SPECS = ("M4", "M5", "M6", "M8", "M10")
FORMATS = {".jpg", ".jpeg", ".png"}
TAB_KEY = "vision_workspace_v1"
st.set_page_config(page_title="智造慧眼 · 螺丝检测", layout="wide", page_icon="⚙️")
# 通过 Streamlit 的 Python 组件接口仅访问本应用的一项标签页存储。
tab_storage = components.declare_component(
    "vision_tab_storage", path=str(files("streamlit_session_browser_storage") / "frontend" / "build")
)


@st.cache_resource(ttl=3600, max_entries=32, show_spinner=False)
def refresh_snapshot(tab_id):
    # 标签页仅持有随机恢复编号，图片与历史临时保存在服务端内存。
    return {"selected_model": None, "inputs": {}, "source": "上传图片", "history": []}


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
    """模拟接口：相同像素返回相同结果，可直接替换为真实模型。坐标以原图像素计。"""
    seed = hashlib.sha256(image.resize((48, 48)).convert("RGB").tobytes()).digest()
    width, height = image.size
    w = max(1, width * (35 + seed[2] % 36) // 100)
    h = max(1, height * (35 + seed[3] % 36) // 100)
    x = (width - w) * seed[4] // 255
    y = (height - h) * seed[5] // 255
    return {"model": SPECS[seed[0] % len(SPECS)], "confidence": (45 + seed[1] % 55) / 100,
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


@st.cache_data(show_spinner=False, max_entries=256)
def library_thumb(path, modified):
    # 文件修改时间参与缓存键；只缓存公共型号库，不缓存用户上传文件。
    with Image.open(path) as im:
        return jpeg_thumb(ImageOps.exif_transpose(im), (280, 220))


def model_library(state):
    st.subheader("螺丝型号库")
    folders = [p.name for p in LIBRARY.iterdir() if p.is_dir()] if LIBRARY.is_dir() else []
    folders.sort(key=lambda name: (SPECS.index(name) if name in SPECS else len(SPECS), name))
    if not folders:
        st.caption("型号库为空，请在 assets/screws 下添加分类文件夹。")
        return
    if "model_picker" not in st.session_state:
        saved = state["selected_model"]
        st.session_state.model_picker = saved if saved in folders else folders[0]
    selected = st.selectbox("选择型号", folders, key="model_picker")
    state["selected_model"] = selected
    files = sorted(p for p in (LIBRARY / selected).iterdir()
                   if p.is_file() and p.suffix.lower() in FORMATS)
    st.caption(f"共 {len(files)} 张 · 预览前 {min(6, len(files))} 张")
    for start in range(0, min(6, len(files)), 2):
        for col, path in zip(st.columns(2), files[start:start + 2]):
            try:
                col.image(library_thumb(path, path.stat().st_mtime_ns),
                          caption=path.name, width="stretch")
            except (OSError, ValueError, Image.DecompressionBombError):
                col.caption(f"{path.name} · 图片不可读")


def update_input(source, key):
    # 只有上传、拍照或清除控件时才替换图片。切换型号、折叠面板不会覆盖它。
    state = st.session_state.workbench
    state.pop("input_error", None)
    file = st.session_state[key]
    state["inputs"].pop(source, None)
    if file is None:
        return
    data = file.getvalue()
    try:
        with Image.open(BytesIO(data)) as im:
            if im.format not in ("JPEG", "PNG"):
                raise ValueError("unsupported image")
            im.load()
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
        state["input_error"] = "图片读取失败，请重新选择有效的 JPG 或 PNG 图片。"
        return
    state["inputs"][source] = {"name": file.name, "data": data, "result": None}


def detect_and_record(state, item, image):
    with st.spinner("AI识别中..."):
        result = mock_detect(image)
        marked = draw_boxes(image, result)
        thumbnail = jpeg_thumb(marked, (160, 120))
        record = {"时间": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S"),
                  "缩略图": "data:image/jpeg;base64," + base64.b64encode(thumbnail).decode("ascii"),
                  "文件名": item["name"], "识别结果": result["model"],
                  "置信度": f'{result["confidence"]:.0%}'}
        # 一次点击对应一条记录；普通页面重绘只读取已有结果。
        item["result"] = result
        state["history"].insert(0, record)


def detection_page():
    state = workspace()
    with st.sidebar:
        model_library(state)
    st.title("智造慧眼 · 螺丝视觉检测")
    st.caption("模拟检测演示 · 型号、置信度与红色检测框由 mock_detect 生成。")
    left, right = st.columns(2)
    with left.container(border=True):
        st.subheader("图像输入")
        if "source_picker" not in st.session_state:
            st.session_state.source_picker = state["source"]
        source = st.radio("图像来源", ["上传图片", "摄像头拍照"], key="source_picker", horizontal=True)
        state["source"] = source
        if source == "上传图片":
            st.file_uploader("选择 JPG / PNG 图片", type=["jpg", "jpeg", "png"], key="upload",
                             on_change=update_input, args=(source, "upload"))
        else:
            st.camera_input("拍摄螺丝照片", key="camera", on_change=update_input, args=(source, "camera"))
        if state.get("input_error"):
            st.error(state["input_error"])
        item = state["inputs"].get(source)
        start = st.button("开始识别", type="primary", disabled=item is None, width="stretch")
    with right.container(border=True):
        st.subheader("检测预览")
        if item is None:
            st.info("上传或拍摄一张图片后，可在这里查看原图和检测框。")
        else:
            with Image.open(BytesIO(item["data"])) as im:
                image = ImageOps.exif_transpose(im).convert("RGB")
            if start:
                detect_and_record(state, item, image)
            result = item["result"]
            preview = draw_boxes(image, result) if result else image
            st.image(preview, caption=f'{item["name"]} · {image.width} × {image.height} px', width="stretch")
            if result:
                label = f'识别为 {result["model"]} 螺丝，置信度：{result["confidence"]:.0%}'
                if result["confidence"] < 0.6:
                    st.warning(f"{label}（低置信度，请复核）")
                else:
                    st.success(label)
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


# 主入口默认为检测页；原有 CLAHE / Canny 调参工作台仍可从侧栏打开。
pages = [st.Page(detection_page, title="实时检测", default=True)]
if (ROOT / "preprocessing.py").is_file():
    pages.append(st.Page("preprocessing.py", title="图像预处理调参"))
st.navigation(pages).run()
