"""智造慧眼 · 本地工业视觉演示。启动命令：streamlit run app.py"""

from collections import deque
from datetime import datetime, timedelta
from html import escape
from io import BytesIO
from pathlib import PurePosixPath
from time import monotonic

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError


TITLE = "智造慧眼 —— 紧固件柔性视觉中枢"
st.set_page_config(page_title=TITLE, page_icon="⚙", layout="wide")


# 模块一：图像输入与网格预览
def preview_image(file):
    """仅在内存中解码，按 EXIF 方向纠正并缩小预览图。"""
    with Image.open(BytesIO(file.getvalue())) as original:
        size = original.size
        original.thumbnail((960, 720))
        return ImageOps.exif_transpose(original).convert("RGBA"), size


def image_workspace():
    st.html('<h2>图像采集</h2>')
    mode = st.radio("上传方式", ["多图上传", "文件夹上传"], horizontal=True)
    directory = mode == "文件夹上传"
    files = st.file_uploader(
        "选择图像文件夹" if directory else "选择或拖入紧固件图像",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        accept_multiple_files="directory" if directory else True,
        key="folder_files" if directory else "image_files",
    )
    st.html(
        '<div class="sectionline"><strong>图像预览</strong>'
        f'<span>{len(files):02d} 个文件</span></div>'
    )
    if not files:
        st.html(
            '<div class="empty-grid"><b>等待图像输入</b>'
            '<span>上传后可在此查看图像和原始尺寸</span></div>'
        )
        return
    for start in range(0, len(files), 2):
        columns = st.columns(2)
        for column, file in zip(columns, files[start:start + 2]):
            name = PurePosixPath(file.name).name
            with column:
                try:
                    image, (width, height) = preview_image(file)
                except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                    st.warning(f"无法预览图像「{name}」，请检查文件内容。")
                    continue
                st.image(image, width="stretch")
                st.html(
                    f'<div class="fileinfo"><strong>{escape(name)}</strong>'
                    f'<span>{width} × {height} px</span></div>'
                )


# 模块二：独立定时刷新的模拟控制台
PARTS = [("M8 六角螺栓", 95, 3), ("M10 六角螺母", 98, 2), ("M6 螺钉", 97, 1)]


def mock_event(sequence, timestamp):
    cycle, step = divmod(sequence, 5)
    part, confidence, channel = PARTS[cycle % len(PARTS)]
    events = [
        ("vision", "视觉系统", "检测到紧固件图像，采集触发。"),
        ("frame", "图像处理", "曝光校正完成，已提取工件轮廓。"),
        ("infer", "算法推理", f"匹配 {part}，置信度 {confidence}%。"),
        ("route", "控制下发", f"发送串口指令 → 激活 {channel} 号包装通道。"),
        ("ack", "设备回执", f"通道 {channel} 分拣完成，工位就绪。"),
    ]
    return (timestamp.strftime("%H:%M:%S"), *events[step])


@st.fragment(run_every=0.5)
def live_console():
    state = st.session_state
    now = datetime.now().astimezone()
    tick = monotonic()
    if "events" not in state:
        # 用过去六秒的模拟事件填充首屏，后续仅增加新事件。
        state.events = deque(maxlen=100)
        for index in range(12):
            state.events.appendleft(mock_event(index, now - timedelta(seconds=(11 - index) * 0.5)))
        state.sequence = 12
        state.last_tick = tick
    else:
        # 上传重绘、浏览器切换后台后，仍按半秒时间轴生成最近的模拟事件。
        due = int((tick - state.last_tick) / 0.5)
        for offset in range(max(0, due - state.events.maxlen), due):
            scheduled = state.last_tick + (offset + 1) * 0.5
            timestamp = now - timedelta(seconds=tick - scheduled)
            state.events.appendleft(mock_event(state.sequence + offset, timestamp))
        state.sequence += due
        state.last_tick += due * 0.5

    rows = "".join(
        f'<div class="event {kind}"><time>[{timestamp}]</time>'
        f'<div><b>{source}</b><span>{message}</span></div></div>'
        for timestamp, kind, source, message in state.events
    )
    slot = st.empty()
    slot.html(
        '<div class="terminal"><div class="termbar"><span>● &nbsp; EVENT STREAM</span>'
        '<span>500 ms</span></div><div class="eventlist" tabindex="0" '
        'aria-label="模拟工业日志，最新事件在前">' + rows + '</div></div>',
    )


# 组合页面
STYLES = """
<style>
:root{--bg:#0b1217;--panel:#101c23;--line:#2b3c45;--text:#e5eef1;--muted:#a0b1b9;
--mint:#65e3bb;--mono:Consolas,"Courier New",monospace}
.stApp{background:var(--bg);color:var(--text);font-family:"Microsoft YaHei",sans-serif;color-scheme:dark}
header[data-testid="stHeader"]{background:transparent;height:0}
[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none}
.stMainBlockContainer{max-width:1600px;padding:2.2rem 3.2vw}
.stMainBlockContainer [data-testid="stVerticalBlock"]{gap:.85rem}
.mast{display:flex;align-items:center;gap:16px;padding:0 0 19px;border-bottom:1px solid var(--line)}
.mark{position:relative;flex:0 0 48px;height:48px;border:1px solid #43796d;border-radius:10px;
background:repeating-linear-gradient(0deg,transparent 0 7px,#25413d 8px),#132c2a;display:grid;place-items:center}
.mark:before{content:"";width:22px;height:22px;border:2px dashed var(--mint);border-radius:4px}
.mark:after{content:"";position:absolute;width:8px;height:8px;border:2px solid var(--mint);border-radius:50%}
.eyebrow,.termbar,.tag,.sectionline span,.fileinfo span{font-family:var(--mono)}
.eyebrow{font-size:12px;letter-spacing:.15em;color:var(--mint);margin-bottom:5px}
.mast h1{font-family:inherit;color:var(--text);font-size:clamp(20px,2.2vw,36px);
font-weight:680;line-height:1.3;letter-spacing:.015em;padding:0;margin:0}
.tag{margin-left:auto;white-space:nowrap;padding:5px 12px;border:1px solid #395b53;
border-radius:20px;color:var(--mint);font-size:13px}
[class*="st-key-kpi_"],.st-key-vision,.st-key-console{background:var(--panel);
border:1px solid var(--line);border-radius:10px}
[class*="st-key-kpi_"]{padding:16px 20px 0;overflow:hidden}
[class*="st-key-kpi_"] [data-testid="stMetricLabel"] p{color:#b6c8ce;font-size:14px}
[class*="st-key-kpi_"] [data-testid="stMetricValue"]{color:#ecf5f7;font:600 36px/1.2 var(--mono)}
.st-key-kpi_status [data-testid="stMetricValue"]{color:var(--mint);font:650 30px/1.45 "Microsoft YaHei",sans-serif}
.st-key-kpi_status [data-testid="stMetricValue"] div:before{content:"";display:inline-block;
height:10px;width:10px;border-radius:50%;background:var(--mint);margin:0 10px 4px 0}
.st-key-kpi_rate [data-testid="stMetricValue"] div:after{content:" 个/分钟"}
.st-key-kpi_total [data-testid="stMetricValue"] div:after{content:" 件"}
[class*="st-key-kpi_"] [data-testid="stMetricValue"] div:after{font:14px "Microsoft YaHei",sans-serif;color:var(--muted)}
.trend{display:flex;align-items:end;gap:3px;height:24px;padding-bottom:1px;margin:0 -1px}
.trend i{flex:1;background:var(--color);opacity:.65;border-radius:1px 1px 0 0}
.st-key-vision,.st-key-console{padding:18px 20px 20px}
.stApp h2{font:650 21px/1.4 "Microsoft YaHei",sans-serif;color:var(--text);padding:0;margin:0}
.stApp h2:before{content:"";display:inline-block;width:4px;height:17px;background:var(--mint);
border-radius:2px;margin-right:10px;vertical-align:-1px}
.stApp [data-testid="stWidgetLabel"] p,.stApp [data-testid="stRadio"] label p{color:#c1d0d5}
.stApp [data-testid="stFileUploaderDropzone"]{background:#14272d;border:1px dashed #427266;color:#d7e7e5}
.stApp [data-testid="stFileUploaderDropzone"] small{color:#b0c5ca}
.stApp [data-testid="stFileUploaderDropzone"] button,.stApp [data-testid="stFileUploaderDropzone"] button p{
background:#213b3a;color:#b8f7dd;border-color:#497e6f}
.stApp [data-testid="stFileUploaderFile"]{color:var(--text)}
.stApp [data-testid="stFileUploaderFile"] button{color:#b2c2ca}
.stApp [data-testid="stFileUploaderFile"] small{color:#9fb3ba}
.sectionline,.fileinfo,.termbar{display:flex;justify-content:space-between;align-items:center;gap:8px}
.sectionline{border-top:1px solid var(--line);padding-top:14px;color:#d3e1e4}
.sectionline span,.fileinfo span{font-size:12px;color:var(--muted)}
.fileinfo{flex-wrap:wrap;align-items:baseline}
.fileinfo strong{font:600 14px var(--mono);overflow-wrap:anywhere}
.stApp [data-testid="stImage"] img{border:1px solid var(--line);border-radius:6px;background:#18252c}
.empty-grid{min-height:150px;background-color:#111e25;background-image:radial-gradient(#2c3c45 1px,transparent 1px);
background-size:14px 14px;display:flex;flex-direction:column;gap:8px;align-items:center;justify-content:center;
border:1px solid var(--line);border-radius:6px}
.empty-grid span{color:var(--muted);font-size:14px}
.terminal{border:1px solid var(--line);border-radius:6px;overflow:hidden;background:#0a141a}
.termbar{background:#19282e;color:var(--mint);font-size:12px;letter-spacing:.08em;padding:10px 13px;
border-bottom:1px solid var(--line)}
.termbar span+span{color:var(--muted);letter-spacing:0}
.eventlist{height:458px;overflow-y:auto;scrollbar-color:#384c54 #101c22}
.event{display:flex;gap:12px;align-items:baseline;padding:10px 12px;border-bottom:1px solid #1c2c35;
font-size:14px;line-height:1.55}.event:first-child{background:#102b28;box-shadow:inset 2px 0 #65e3bb}
.event time{font:12px var(--mono);white-space:nowrap;color:#9eafb7}
.event b{color:#83c3ec;font-weight:600;margin-right:9px}.event span{color:#d2dee2}
.event.infer b,.event.ack b{color:var(--mint)}.event.route b{color:#e7bc83}
@media(max-width:760px){.stMainBlockContainer{padding:1.2rem .9rem}.mast{gap:10px}.tag{padding:4px 7px}
.mark{flex-basis:35px;height:35px}.st-key-vision,.st-key-console{padding:14px}.fileinfo{gap:2px}}
</style>
"""


def dashboard():
    st.html(STYLES)
    st.html(
        '<div class="mast"><div class="mark" aria-hidden="true"></div><div>'
        '<div class="eyebrow">INDUSTRIAL MACHINE VISION</div>'
        f'<h1>{TITLE}</h1></div><span class="tag">● 模拟数据</span></div>',
    )
    metrics = [
        ("status", "设备状态", "在线", "#5ccba5", [9]*28),
        ("rate", "当前识别吞吐量", "45", "#6fb7d9", [6,8,7,9,12,10,11,14,13,12,16,15,17,15]),
        ("compute", "端侧算力占用", "18%", "#d4ab74", [5,7,6,10,8,9,13,10,8,9,12,8,9,8]),
        ("total", "累计处理", "1250", "#97a1e7", [4,5,6,6,7,8,9,9,10,12,13,13,15,16]),
    ]
    for col, (key, label, value, color, heights) in zip(st.columns(4), metrics):
        with col.container(key=f"kpi_{key}"):
            st.metric(label, value)
            bars = "".join(f'<i style="height:{h}px"></i>' for h in heights)
            st.html(f'<div class="trend" aria-hidden="true" style="--color:{color}">{bars}</div>')
    left, right = st.columns([6, 4])
    with left.container(key="vision"):
        image_workspace()
    with right.container(key="console"):
        st.html("<h2>实时工业日志</h2>")
        live_console()


if __name__ == "__main__":
    dashboard()
