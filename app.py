# app.py – Final Working Version with gpt-image-1
# Requirements: pip install streamlit pillow openai python-dotenv

import os
import io
import base64
import json
import tempfile
from PIL import Image

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import streamlit as st
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────
# 🔑 API KEY
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Please set OPENAI_API_KEY in .env or environment.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ──────────────────────────────────────────────────────────────────────
# 🎨 Prompt Options
PRESET_PROMPTS = {
    "Modern": "Sleek, minimalist ad style—clean lighting, white background. Preserve framing.",
    "Retro": "1970s retro magazine look—warm grain, gentle vignette. Keep composition.",
    "Cinematic": "Teal-and-orange cinematic lighting, high dynamic range. Movie-style finish.",
}

# ──────────────────────────────────────────────────────────────────────
# 🧠 Run Inference — gpt-image-1 (returns PIL Image + JSON)
def run_inference(image_bytes: bytes, prompt: str) -> tuple[Image.Image, dict]:
    im = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    side = max(im.size)
    if im.width != im.height:
        square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        square.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
        im = square

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
        im.save(tmp_img, format="PNG")
        img_path = tmp_img.name
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_mask:
        Image.new("RGBA", im.size, (0, 0, 0, 0)).save(tmp_mask, format="PNG")
        mask_path = tmp_mask.name

    result = client.images.edit(
        model="gpt-image-1",
        image=open(img_path, "rb"),
        mask=open(mask_path, "rb"),
        prompt=prompt,
        n=1,
        size="1024x1024",
        response_format="b64_json",
    )

    b64 = result.data[0].b64_json
    image = Image.open(io.BytesIO(base64.b64decode(b64)))
    return image, result.model_dump()

# ──────────────────────────────────────────────────────────────────────
# 🌐 Streamlit UI
st.set_page_config(page_title="Ad Image Transformer", layout="centered")
st.title("📸 Ad Image Transformer")
st.caption("Upload a photo, pick a style or write your own prompt.")

uploaded = st.file_uploader("Upload PNG or JPG", type=["png", "jpg", "jpeg"])

cols = st.columns(3)
for key, col in zip(PRESET_PROMPTS, cols):
    with col:
        if st.button(key):
            st.session_state["preset_prompt"] = PRESET_PROMPTS[key]

custom_prompt = st.text_input("…or enter your own prompt")
prompt = custom_prompt.strip() or st.session_state.get("preset_prompt", "")

if st.button("✨ Generate", disabled=not uploaded or not prompt):
    with st.spinner("Generating… please wait"):
        image, metadata = run_inference(uploaded.read(), prompt)

        st.image(image, caption="🎨 Final Result", use_container_width=True)

        # Download PNG
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        st.download_button("Download PNG", data=buf.getvalue(), file_name="output.png", mime="image/png")

        # Download JSON
        st.download_button("Download Metadata JSON", data=json.dumps(metadata, indent=2), file_name="metadata.json", mime="application/json")
