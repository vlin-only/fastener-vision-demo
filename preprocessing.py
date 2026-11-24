import hashlib
import html
import time
from datetime import datetime

import cv2
import numpy as np
import streamlit as st

# ----------------------------------------------------------------------------
# 页面配置（必须是第一个 Streamlit 调用）
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="智造慧眼 —— 紧固件柔性视觉中枢",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# 深色工业科技风样式
# ----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
.stApp {
    background: radial-gradient(circle at 20% 0%, #14213d 0%, #0b0f19 45%, #070a12 100%);
    color: #d6e4ff;
}
header[data-testid="stHeader"] { background: transparent; }

h1 {
    color: #00e5ff !important;
    letter-spacing: 2px;
    text-shadow: 0 0 12px rgba(0, 229, 255, 0.45);
    border-bottom: 1px solid rgba(0, 229, 255, 0.25);
    padding-bottom: 0.4rem;
}
h2, h3 { color: #8fd3ff !important; }

/* 侧边栏 */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1424 0%, #090d17 100%);
    border-right: 1px solid rgba(0, 229, 255, 0.25);
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #00e5ff !important; }
[data-testid="stSidebar"] label p { color: #b8c9e8 !important; }
[data-testid="stSidebar"] [data-testid="stSliderThumbValue"],
[data-testid="stSidebar"] [data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stTickBarMax"] {
    color: #00ffa3 !important;
    font-family: "Consolas", "Courier New", monospace;
}

/* 指标卡片 */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #111a2e 0%, #0d1424 100%);
    border: 1px solid rgba(0, 229, 255, 0.25);
    border-left: 4px solid #00e5ff;
    border-radius: 8px;
    padding: 14px 18px;
    box-shadow: 0 0 14px rgba(0, 229, 255, 0.08);
}
[data-testid="stMetricLabel"] p { color: #7f93b5 !important; font-size: 0.9rem; }
[data-testid="stMetricValue"] { color: #00ffa3 !important; font-family: "Consolas", "Courier New", monospace; }

/* 上传控件 */
[data-testid="stFileUploader"] section {
    background: #0f1626;
    border: 1px dashed rgba(0, 229, 255, 0.45);
    border-radius: 8px;
}

/* 空状态虚线框 */
.empty-box {
    height: 320px;
    margin-top: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 2px dashed rgba(0, 229, 255, 0.45);
    border-radius: 10px;
    color: #5f7ba6;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 1.1rem;
    letter-spacing: 2px;
    background: rgba(15, 22, 38, 0.6);
}

/* 图片标题 / 参数说明 */
.img-title {
    color: #00e5ff;
    font-size: 0.95rem;
    font-weight: 600;
    margin: 8px 0 4px 0;
    padding-left: 8px;
    border-left: 3px solid #00e5ff;
}
.img-cap {
    color: #7f93b5;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 0.78rem;
    margin-top: 2px;
}

/* 终端日志 */
.terminal {
    background: #05080f;
    border: 1px solid rgba(0, 255, 163, 0.35);
    border-radius: 8px;
    padding: 14px 16px;
    min-height: 360px;
    max-height: 520px;
    overflow-y: auto;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 0.88rem;
    line-height: 1.7;
    box-shadow: inset 0 0 18px rgba(0, 255, 163, 0.06);
}
.log-line { color: #00ffa3; }
.log-param { color: #4dd0ff; }
.log-warn { color: #ffb020; }
.log-err  { color: #ff5470; }
.log-idle { color: #7f93b5; }
.cursor   { color: #00ffa3; animation: blink 1s steps(1) infinite; }
@keyframes blink { 50% { opacity: 0; } }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 常量与 Session State
# ----------------------------------------------------------------------------
STANDBY_MSG = "[系统] 视觉传感器待机中，等待图像输入..."
MAX_LOG_LINES = 200      # 日志最大保留行数，防止长时间调参导致列表无限增长
VISIBLE_LOG_LINES = 18   # 终端框内最多显示的最新行数（模拟终端滚动效果）

# 滑块默认参数（key 即 session_state 中的键）
DEFAULT_PARAMS = {
    "clip_limit": 2.0,
    "tile_size": 8,
    "canny_range": (50, 150),
}
for _k, _v in DEFAULT_PARAMS.items():
    st.session_state.setdefault(_k, _v)

# 业务状态
DEFAULT_STATE = {
    "file_sig": None,     # 当前文件指纹，用于判断是否为"新文件"
    "result": None,       # 预处理结果（dict）；解码失败时为 None
    "params_sig": None,   # 上一次生效的参数组合，用于判断"参数是否变化"
    "log_lines": [],      # 已输出的日志行
    "log_played": False,  # 首次滚动日志动画是否已播放完毕
}
for _k, _v in DEFAULT_STATE.items():
    st.session_state.setdefault(_k, _v)

def reset_state():
    for k, v in DEFAULT_STATE.items():
        st.session_state[k] = [] if isinstance(v, list) else v

def reset_params():
    """按钮回调：必须在滑块实例化之前修改其 key 对应的 session_state，故使用 on_click。"""
    for k, v in DEFAULT_PARAMS.items():
        st.session_state[k] = v

# ----------------------------------------------------------------------------
# 图像处理（真实 OpenCV 逻辑）
# ----------------------------------------------------------------------------
def decode_image(data: bytes):
    """从内存字节流解码图像（避免 imread 对中文路径/临时文件的问题）。"""
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)  # 返回 BGR，丢弃 alpha 通道

def preprocess(bgr: np.ndarray, clip_limit: float, tile_size: int,
               canny_low: int, canny_high: int) -> dict:
    h, w = bgr.shape[:2]

    # 1) 转灰度
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 2) CLAHE 自适应直方图均衡（模拟去反光/局部对比度增强）
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit),
                            tileGridSize=(int(tile_size), int(tile_size)))
    enhanced = clahe.apply(gray)

    # 3) Canny 边缘提取（先做轻度高斯滤波抑制 CLAHE 放大的噪声）
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blurred, int(canny_low), int(canny_high))

    edge_ratio = float(np.count_nonzero(edges)) / edges.size * 100.0

    return {
        "bgr": bgr, "enhanced": enhanced, "edges": edges,
        "width": w, "height": h, "edge_ratio": edge_ratio,
    }

# ----------------------------------------------------------------------------
# 日志渲染
# ----------------------------------------------------------------------------
def render_log(box, lines, cursor=True, idle=False):
    rows = []
    for line in lines[-VISIBLE_LOG_LINES:]:
        if idle:
            cls = "log-idle"
        elif "错误" in line:
            cls = "log-err"
        elif "警告" in line:
            cls = "log-warn"
        elif "参数已更新" in line:
            cls = "log-param"
        else:
            cls = "log-line"
        rows.append(f'<div class="{cls}">{html.escape(line)}</div>')
    cursor_html = '<span class="cursor">█</span>' if cursor else ""
    # 注意：HTML 不要缩进/换行，避免被 Markdown 当作代码块
    box.markdown(
        f'<div class="terminal">{"".join(rows)}{cursor_html}</div>',
        unsafe_allow_html=True,
    )

def now_ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def append_log(msg: str):
    lines = st.session_state.log_lines
    lines.append(f"[{now_ts()}] {msg}")
    if len(lines) > MAX_LOG_LINES:
        del lines[: len(lines) - MAX_LOG_LINES]

# ----------------------------------------------------------------------------
# 侧边栏：预处理参数调节
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 预处理参数调节")
    st.caption("拖动滑块，右侧图像将实时重新计算，便于演示螺栓螺纹边缘提取效果。")

    st.subheader("CLAHE 去反光增强")
    clip_limit = st.slider(
        "clipLimit（对比度限制）",
        min_value=1.0, max_value=10.0, step=0.5, key="clip_limit",
        help="值越大，局部对比度增强越强，但也会放大噪声与金属纹理；值越小越保守。",
    )
    tile_size = st.slider(
        "tileGridSize（分块数，N×N）",
        min_value=2, max_value=32, step=2, key="tile_size",
        help="图像被划分为 N×N 个小块分别均衡。N 越大，越关注局部细节（如螺纹）；N 越小越接近全局均衡。",
    )

    st.subheader("Canny 边缘提取")
    canny_low, canny_high = st.slider(
        "低 / 高阈值",
        min_value=0, max_value=255, step=1, key="canny_range",
        help="梯度高于「高阈值」的为强边缘；介于两者之间的仅在与强边缘相连时保留；低于「低阈值」的丢弃。"
             "使用区间滑块可保证低阈值始终不大于高阈值。",
    )

    st.button("↺ 恢复默认参数", on_click=reset_params, use_container_width=True)

    st.markdown("---")
    st.markdown(
        f'<div class="img-cap">当前参数<br>'
        f'clipLimit = {clip_limit}<br>'
        f'tileGridSize = {tile_size} × {tile_size}<br>'
        f'Canny = ({canny_low}, {canny_high})</div>',
        unsafe_allow_html=True,
    )

PARAMS = (float(clip_limit), int(tile_size), int(canny_low), int(canny_high))

# ----------------------------------------------------------------------------
# 主体：左（图像处理）右（实时日志）
# ----------------------------------------------------------------------------
left, right = st.columns([6, 4])

with left:
    st.subheader("图像处理区")
    uploaded = st.file_uploader(
        "上传紧固件图像（单张，支持 jpg / png）",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if uploaded is None:
        # 文件被移除 -> 清空业务状态，下次再传同一张图也能重新播放日志
        if st.session_state.file_sig is not None:
            reset_state()
        st.markdown('<div class="empty-box">等待输入图像数据...</div>', unsafe_allow_html=True)
    else:
        data = uploaded.getvalue()
        sig = hashlib.md5(data).hexdigest()

        if st.session_state.file_sig != sig:
            # ---- 情况 1：新文件 -> 解码 + 处理 + 重置日志，等待播放动画 ----
            bgr = decode_image(data)
            st.session_state.file_sig = sig
            st.session_state.result = preprocess(bgr, *PARAMS) if bgr is not None else None
            st.session_state.params_sig = PARAMS
            st.session_state.log_lines = []
            st.session_state.log_played = False

        elif (st.session_state.result is not None
              and st.session_state.params_sig != PARAMS):
            # ---- 情况 2：同一文件但参数变化 -> 仅重算图像，静态追加一行日志，不重播动画 ----
            st.session_state.result = preprocess(st.session_state.result["bgr"], *PARAMS)
            st.session_state.params_sig = PARAMS
            if st.session_state.log_played:
                append_log(
                    f"参数已更新：CLAHE(clipLimit={PARAMS[0]}, tile={PARAMS[1]}x{PARAMS[1]})，"
                    f"Canny({PARAMS[2]}, {PARAMS[3]})，预处理已重新执行。"
                )

        result = st.session_state.result
        if result is None:
            st.error("图像解码失败：文件可能已损坏或格式不受支持。")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown('<div class="img-title">原图</div>', unsafe_allow_html=True)
                st.image(result["bgr"], channels="BGR", use_container_width=True)
                st.markdown(
                    f'<div class="img-cap">{result["width"]} x {result["height"]} px</div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown('<div class="img-title">增强后的灰度图（CLAHE）</div>', unsafe_allow_html=True)
                st.image(result["enhanced"], use_container_width=True)
                st.markdown(
                    f'<div class="img-cap">clipLimit={PARAMS[0]} | tile={PARAMS[1]}x{PARAMS[1]}</div>',
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown('<div class="img-title">边缘特征提取图（Canny）</div>', unsafe_allow_html=True)
                st.image(result["edges"], use_container_width=True)
                st.markdown(
                    f'<div class="img-cap">阈值 {PARAMS[2]}/{PARAMS[3]} | 边缘占比 {result["edge_ratio"]:.2f}%</div>',
                    unsafe_allow_html=True,
                )

with right:
    st.subheader("实时日志")
    log_box = st.empty()  # 日志占位符

    if uploaded is None:
        render_log(log_box, [STANDBY_MSG], cursor=False, idle=True)

    elif st.session_state.result is None:
        render_log(log_box, [f"[{now_ts()}] 错误：图像解码失败，已中止预处理流程。"])

    elif st.session_state.log_played:
        # 同一张图的后续 rerun（含调参）：静态展示历史日志，不 sleep、不重播
        render_log(log_box, st.session_state.log_lines)

    else:
        # 新图像：逐行滚动输出（基于真实处理结果）
        res = st.session_state.result
        messages = [
            "传感器事件：接收到图像数据",
            f"图像尺寸：{res['width']} x {res['height']} px",
            "执行 CLAHE 去反光预处理...",
            "执行 Canny 算子边缘特征提取...",
            "预处理模块执行完毕，特征数据已就绪。",
            "警告：紧固件分类模型尚未挂载，等待接入工厂实测数据集。",
        ]
        st.session_state.log_lines = []
        for msg in messages:
            append_log(msg)
            render_log(log_box, st.session_state.log_lines)
            time.sleep(0.5)

        st.session_state.log_played = True
        render_log(log_box, st.session_state.log_lines)  # 播放完成后去掉中间态，保持一致
