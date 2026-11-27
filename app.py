"""运行：python -m streamlit run app.py。依赖见 requirements.txt。"""
import hashlib
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from tempfile import gettempdir

import cv2
import numpy as np
import psutil
import streamlit as st

from panel import parameter_controls
from vision import Resources, correct_and_update, recognize


ROOT = Path(__file__).resolve().parent
# 替换模型时指向相同格式的检查点。参考样本库使用其中的 catalog，纠错库按权重摘要隔离。
WEIGHTS = Path(os.environ.get("SCREW_WEIGHTS", ROOT / "models" / "screw.pt"))
FEEDBACK = Path(os.environ.get("SCREW_FEEDBACK_DIR", Path(gettempdir()) / "screw-feedback"))
NO_TARGET = "未检测到紧固件，请调整角度或重新上传"
st.set_page_config(page_title="智造慧眼 · 紧固件视觉检测", layout="wide")


@st.cache_resource(show_spinner=False)
def load_resources(weights, feedback_dir):
    return Resources(weights, feedback_dir)


def main():
    state = st.session_state
    state.setdefault("history", [])
    state.setdefault("count", 0)
    state.setdefault("elapsed", 0.0)
    state.setdefault("status", "在线")
    with st.sidebar:
        st.subheader("预处理参数调节")
        params = parameter_controls(state)
        st.subheader("杆径标定")
        method = st.selectbox("标定方式", ["未标定", "固定机位", "钢尺刻线"])
        value = None
        if method != "未标定":
            label = "原图比例（像素/mm）" if method == "固定机位" else "相邻刻线实际距离（mm）"
            value = st.number_input(label, min_value=0.0001, value=None, format="%.4f")

    st.title("智造慧眼 · 紧固件视觉检测")
    metrics = st.empty()
    left, right = st.columns([2, 1])
    with right:
        st.subheader("识别日志")
        log_panel = st.empty()
    with left:
        st.subheader("图像输入与检测")
        source = st.radio("图像来源", ["上传图片", "摄像头拍照"], horizontal=True)
        if source == "上传图片":
            image_file = st.file_uploader("选择 JPG / PNG 图片", type=["jpg","jpeg","png"])
        else:
            image_file = st.camera_input("拍摄螺丝照片")
        retry = st.button("重新识别", disabled=image_file is None)
        original, annotated = st.columns(2)
        original.markdown("**原始图像**")
        annotated.markdown("**增强与识别结果图像**")

    def show_logs():
        log_panel.code("\n".join(reversed(state.get("logs", []))) or "等待图像输入。",
                       language=None, wrap_lines=True)

    def log(message):
        state.logs.append(f"[{datetime.now():%H:%M:%S}] {message}")
        show_logs()

    if image_file is None:
        state.update(logs=[], current=None, result=None, bgr=None, image_id=None, notice=None,
                     counted_image=None, status="在线")
    else:
        raw = image_file.getvalue()
        image_id = hashlib.sha256(raw).hexdigest()
        new_image = state.get("image_id") != image_id
        if new_image:
            state.update(image_id=image_id, result=None, current=None, notice=None, logs=[])
            try:
                # OpenCV 直接解码上传缓冲区。原图不会写入临时文件。
                state.bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            except cv2.error:
                state.bgr = None
        bgr = state.get("bgr")
        current = (image_id, params, method, value)
        if bgr is None:
            state.status = "异常"
            original.error("图像无法解码，请重新选择有效的 JPG / PNG。")
        else:
            original.image(bgr, channels="BGR", width="stretch")
            if method != "未标定" and value is None:
                annotated.info("请填写实测标定值，或选择“未标定”直接识别。")
            else:
                if current != state.get("current") or retry:
                    state.update(current=current, result=None, logs=[], notice=None)
                    log(f"收到图像 {image_file.name}，尺寸 {bgr.shape[1]} × {bgr.shape[0]} 像素。")
                    try:
                        resources = load_resources(WEIGHTS, FEEDBACK)
                        start = time.perf_counter()
                        result = recognize(resources, bgr, raw, params, (method, value), log)
                        if result is None:
                            state.notice = NO_TARGET
                            state.status = "未检出"
                            log("未检出紧固件，本次检测结束。")
                        else:
                            state.result = result
                            state.status = "在线"
                            state.elapsed = time.perf_counter() - start
                            if retry or state.get("counted_image") != image_id:
                                state.count += 1
                                state.history.insert(0, {"时间": datetime.now().strftime("%H:%M:%S"),
                                    "文件": image_file.name, "型号": result["label"],
                                    "余弦相似度": round(result["similarity"], 3),
                                    "杆径": result["size"], "分箱": result["bin"]})
                            state.counted_image = image_id
                    except (OSError, ValueError, RuntimeError, KeyError, cv2.error, sqlite3.Error) as exc:
                        state.status = "异常"
                        state.notice = f"识别未完成：{exc}"
                        log(state.notice)
                if state.get("notice"):
                    annotated.warning(state.notice)
                result = state.get("result")
                if result:
                    annotated.image(result["image"], channels="BGR", width="stretch")
                    annotated.success(f'{result["label"]} · {result["bin"]} 号箱')
                    annotated.caption(f'杆径：{result["size"]}　余弦相似度：{result["similarity"]:.3f}')
                    with right.expander("确认或纠正型号"):
                        resources = load_resources(WEIGHTS, FEEDBACK)
                        labels = resources.model.classes
                        actual = st.selectbox("实际型号", labels, index=labels.index(result["label"]),
                                              key=f"label_{image_id}")
                        if st.button("保存到纠错参考库"):
                            try:
                                update = correct_and_update(resources, raw, result, actual)
                                message = "已更新" if update["changed"] else "已记录过"
                                log(f'人工确认 {actual}：样本{message}，参考库反馈共 {update["sample_count"]} 张。')
                                st.success(f"已记录 {actual}。点击“重新识别”可查看最新匹配结果。")
                            except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
                                st.error(f"纠错保存失败：{exc}")
    show_logs()
    with metrics.container():
        status, rate, cpu, total = st.columns(4)
        status.metric("检测状态", state.status)
        rate.metric("最近单图耗时", f"{state.elapsed:.2f} 秒" if state.elapsed else "—")
        cpu.metric("主机 CPU", f"{psutil.cpu_percent(interval=0.1):.1f}%")
        total.metric("累计处理", f"{state.count} 件")
    st.divider()
    st.subheader("历史检测记录")
    if state.history:
        st.dataframe(state.history, hide_index=True, width="stretch")
        if st.button("清空记录"):
            state.history.clear()
            st.rerun()
    else:
        st.caption("暂无检测记录。")


if __name__ == "__main__":
    main()
