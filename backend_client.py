"""Streamlit 前端与本机螺丝识别后端之间的 HTTP 客户端。

只发送原始图片字节（image/*），不加载任何本地识别模型，
也不在本地做任何识别或兜底。后端地址从环境变量 SCREW_API_URL 读取，
默认 http://127.0.0.1:8013。
"""

import os
import secrets
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import gettempdir
from threading import Lock, Thread
from typing import Any, Dict, Optional, Tuple, Union

import requests
import streamlit as st

DEFAULT_API_URL = "http://127.0.0.1:8013"
CONNECT_TIMEOUT = 3.0
READ_TIMEOUT = 120.0
HEALTH_READ_TIMEOUT = 3.0
MAX_IMAGE_BYTES = 32 * 1024 * 1024

CONNECTION_ERROR = "无法连接本机识别服务，请先启动后端"
INVALID_RESULT = "未获得有效识别结果，请调整角度、检查后端服务或重新上传。"
FEEDBACK_CONFLICT = "识别记录已失效，请重新识别这张照片后再提交。"

_STATUS_MESSAGES = {
    400: "图片无效或卡尺外径参数不合法，请重新上传。",
    413: "图片超过 32 MiB，请压缩后重试。",
    415: "后端需要 image/* 类型的原始图片字节。",
    500: "后端识别或反馈训练失败，请稍后重试。",
}

BytesLike = Union[bytes, bytearray]


class BackendError(RuntimeError):
    """后端调用失败的基类。"""


class BackendUnavailable(BackendError):
    """连接不上后端，或后端健康检查不通过。"""


class BackendResponseError(BackendError):
    """后端已响应，但请求或结果不符合约定。"""


class FeedbackConflict(BackendResponseError):
    """反馈 token / 图片 / 预测类别与识别记录不匹配（HTTP 409）。"""


@st.cache_resource(show_spinner=False)
def hosted_service():
    """同一云实例只加载一份权重，识别和反馈沿用桌面版 HTTP 接口。"""
    from screw_backend import Handler, ScrewIdentifier

    weights = Path(__file__).resolve().parent / "models" / "screw.pt"
    model = ScrewIdentifier(weights, feedback_dir=Path(gettempdir()) / "screw-feedback")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.classifier = model
    server.predict_lock = Lock()
    server.feedback_key = secrets.token_bytes(32)
    Thread(target=server.serve_forever, daemon=True).start()
    return server


def base_url() -> str:
    external = os.environ.get("SCREW_API_URL")
    if external:
        return external.rstrip("/")
    return f"http://127.0.0.1:{hosted_service().server_port}"


def _timeouts(read_timeout: float = READ_TIMEOUT) -> Tuple[float, float]:
    return (CONNECT_TIMEOUT, read_timeout)


def _validate_image(image_bytes: BytesLike, mime: str) -> bytes:
    if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
        raise BackendResponseError("图片内容为空，请重新选择图片。")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise BackendResponseError("图片超过 32 MiB，请压缩后重试。")
    if not mime or not str(mime).lower().startswith("image/"):
        raise BackendResponseError("需要 image/* 类型的原始图片字节。")
    return bytes(image_bytes)


def _raise_for_status(response: requests.Response, *, conflict_message: str = "") -> None:
    status = response.status_code
    if status == 200:
        return
    if status == 409 and conflict_message:
        raise FeedbackConflict(conflict_message)
    message = _STATUS_MESSAGES.get(status, "本机识别服务返回了异常状态码 %d。" % status)
    raise BackendResponseError(message)


def _json_body(response: requests.Response) -> Dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise BackendResponseError("识别服务返回了无法解析的结果。") from exc
    if not isinstance(payload, dict):
        raise BackendResponseError("识别服务返回结果格式不正确。")
    return payload


def health() -> bool:
    """GET / 健康检查；必须用根路由，后端没有 /health。"""
    try:
        response = requests.get(
            base_url() + "/",
            timeout=_timeouts(HEALTH_READ_TIMEOUT),
        )
    except requests.exceptions.RequestException as exc:
        raise BackendUnavailable(CONNECTION_ERROR) from exc
    if response.status_code != 200 or response.headers.get("X-Screw-Upload") != "single-file-1":
        raise BackendUnavailable("识别服务健康检查失败，请确认后端版本。")
    return True


def predict(
    image_bytes: BytesLike,
    mime: str,
    diameter_mm: Optional[float] = None,
) -> Dict[str, Any]:
    """POST /predict；body 为原始图片字节，使用表单以外的原始请求体。"""
    body = _validate_image(image_bytes, mime)
    headers = {"Content-Type": str(mime).lower()}
    if diameter_mm is not None:
        headers["X-Shank-Diameter-MM"] = str(diameter_mm)
    try:
        response = requests.post(
            base_url() + "/predict",
            data=body,
            headers=headers,
            timeout=_timeouts(),
        )
    except requests.exceptions.RequestException as exc:
        raise BackendUnavailable(CONNECTION_ERROR) from exc
    _raise_for_status(response)
    payload = _json_body(response)
    if not payload.get("class_name") or not payload.get("ranking"):
        raise BackendResponseError(INVALID_RESULT)
    return payload


def feedback(
    image_bytes: BytesLike,
    mime: str,
    token: str,
    predicted: str,
    actual: str,
) -> Dict[str, Any]:
    """POST /feedback；必须复用 /predict 时的同一份原始字节与 predicted class。"""
    body = _validate_image(image_bytes, mime)
    if not token:
        raise FeedbackConflict(FEEDBACK_CONFLICT)
    headers = {
        "Content-Type": str(mime).lower(),
        "X-Prediction-Token": str(token),
        "X-Predicted-Class": str(predicted),
        "X-Actual-Class": str(actual),
    }
    try:
        response = requests.post(
            base_url() + "/feedback",
            data=body,
            headers=headers,
            timeout=_timeouts(),
        )
    except requests.exceptions.RequestException as exc:
        raise BackendUnavailable(CONNECTION_ERROR) from exc
    _raise_for_status(response, conflict_message=FEEDBACK_CONFLICT)
    payload = _json_body(response)
    if "sample_count" not in payload:
        raise BackendResponseError("反馈服务返回结果不完整，请重试。")
    return payload
