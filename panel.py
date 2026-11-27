import base64

import streamlit as st


DEFAULT_PARAMS = {"clip_limit": 2.0, "tile_size": 8, "canny_range": (42, 112)}


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
