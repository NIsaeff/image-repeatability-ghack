# app.py
import io
import time
import requests
from PIL import Image
import streamlit as st

# --------------------------------------------------
#  Page & global style
# --------------------------------------------------
st.set_page_config(
    page_title="🔮 One-Click Ad Image Generator",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- light modern look ---
st.markdown(
    """
    <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
    .stButton>button {
        border: 1px solid #e5e7eb; border-radius: 8px; padding: 0.6rem 1.4rem;
        background: #ffffff; font-weight: 600; transition: all 0.2s ease;
    }
    .stButton>button:hover { background: #f3f4f6; border-color:#d1d5db; }
    .preset { width: 100%; }
    .title { font-size: 2.3rem; font-weight: 800; margin-bottom:0.1rem; }
    .subtitle { color:#6b7280; font-size: 0.95rem; margin-top:-0.3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
#  Helper to hit your model
# --------------------------------------------------
BACKEND_URL = "http://localhost:8000/generate"  # <<—-- replace

def run_inference(image_bytes: bytes, prompt: str) -> Image.Image:
    """
    Sends the original image & prompt to your model and returns a PIL.Image.
    Assumes the backend returns raw bytes of the final image.
    """
    files = {"file": image_bytes}
    data = {"prompt": prompt}
    # r = requests.post(BACKEND_URL, files=files, data=data)
    # r.raise_for_status()
    # return Image.open(io.BytesIO(r.content))

    # --- stub while backend is not wired ---
    time.sleep(2)
    return Image.open(io.BytesIO(image_bytes))  # echo original until model ready


# --------------------------------------------------
#  UI
# --------------------------------------------------
st.markdown('<div class="title">📸 Ad Image Transformer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a photo, pick a style (or type your own) and get a polished ad-ready image in seconds.</div>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

preset_col1, preset_col2, preset_col3 = st.columns(3)

# --- preset prompt handlers ---
preset_map = {
    "Modern": "Ultra-clean, minimalist, bright studio lighting, crisp product focus, white backdrop.",
    "Retro": "Vintage 1970s magazine aesthetic, warm film grain, subtle halftone texture, soft vignette.",
    "Cinematic": "High-contrast teal-and-orange grade, shallow depth of field, dramatic rim lighting."
}

for label, col in zip(preset_map.keys(), [preset_col1, preset_col2, preset_col3]):
    with col:
        if st.button(label, key=label, help=preset_map[label], use_container_width=True):
            st.session_state["chosen_prompt"] = preset_map[label]

# --- custom prompt ---
custom = st.text_input("…or write your own prompt", placeholder="e.g. Vibrant neon cyberpunk billboard, night cityscape, rain reflections")

prompt = st.session_state.get("chosen_prompt") if custom == "" else custom.strip()
st.write("")  # spacing

# --- generate action ---
disabled = uploaded is None or prompt == ""
go = st.button("✨ Generate", disabled=disabled, type="primary")

if go and uploaded and prompt:
    with st.spinner("Transforming… hang tight!"):
        result_img = run_inference(uploaded.read(), prompt)
        st.success("Done! Preview below ⤵")
        st.image(result_img, use_column_width="auto", caption="Final Creative")
        # Optional download
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        st.download_button("Download PNG", data=buf.getvalue(), file_name="ad_image.png", mime="image/png")
