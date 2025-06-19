# app.py
import io
import time
import requests
import os
from PIL import Image
import streamlit as st
from enhanced_image_deconstructor import EnhancedImageDeconstructor
from openai_image_deconstructor import OpenAIImageDeconstructor
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# --------------------------------------------------
#  Page & global style
# --------------------------------------------------
st.set_page_config(
    page_title="🔮 Ad Image Generator",
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
#  API Configuration
# --------------------------------------------------
FRIENDLI_TOKEN = "flp_10WNT4j5Nw2AP2ND2ITYvw7Ft0meCBZzVdXVCnFbVLrea"
ENDPOINT_ID = "dep7rl676gqk463"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check for required API keys
if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not found. Please set it in your .env file or environment variables.")
    st.info("You can get an OpenAI API key from: https://platform.openai.com/api-keys")

# Initialize the image deconstructors
@st.cache_resource
def get_image_deconstructor():
    try:
        deconstructor = EnhancedImageDeconstructor(FRIENDLI_TOKEN, ENDPOINT_ID)
        st.success("✅ Friendli AI connection initialized successfully!")
        return deconstructor
    except Exception as e:
        st.error(f"❌ Failed to initialize Friendli AI: {e}")
        return None

@st.cache_resource
def get_openai_deconstructor():
    if not OPENAI_API_KEY:
        return None
    try:
        deconstructor = OpenAIImageDeconstructor(OPENAI_API_KEY, FRIENDLI_TOKEN, ENDPOINT_ID)
        st.success("✅ OpenAI + Friendli AI connection initialized successfully!")
        return deconstructor
    except Exception as e:
        st.error(f"❌ Failed to initialize OpenAI deconstructor: {e}")
        return None

# --------------------------------------------------
#  Helper Functions
# --------------------------------------------------
def download_image_from_url(url):
    """Download image from URL and return as PIL Image"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        st.error(f"Error downloading image: {e}")
        return None

# --------------------------------------------------
#  Original Backend Helper (stub)
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
st.markdown('<div class="title">📸 Ad Image Generator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a photo, pick a style (or type your own) and get a polished ad-ready image in seconds.</div>',
    unsafe_allow_html=True,
)

# Create tabs for different features
tab1, tab2, tab3 = st.tabs(["🎨 Style Transfer", "🔍 Image-Based Generation", "🤖 AI-Enhanced Generation"])

# Tab 1: Original Style Transfer Feature
with tab1:
    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"], key="style_uploader")

    preset_col1, preset_col2, preset_col3 = st.columns(3)

    # --- preset prompt handlers ---
    preset_map = {
        "Modern": "Ultra-clean, minimalist, bright studio lighting, crisp product focus, white backdrop.",
        "Retro": "Vintage 1970s magazine aesthetic, warm film grain, subtle halftone texture, soft vignette.",
        "Cinematic": "High-contrast teal-and-orange grade, shallow depth of field, dramatic rim lighting."
    }

    for label, col in zip(preset_map.keys(), [preset_col1, preset_col2, preset_col3]):
        with col:
            if st.button(label, key=f"style_{label}", help=preset_map[label], use_container_width=True):
                st.session_state["chosen_prompt"] = preset_map[label]

    # --- custom prompt ---
    custom = st.text_input("…or write your own prompt", placeholder="e.g. Vibrant neon cyberpunk billboard, night cityscape, rain reflections")

    prompt = st.session_state.get("chosen_prompt") if custom == "" else custom.strip()
    st.write("")  # spacing

    # --- generate action ---
    disabled = uploaded is None or prompt == ""
    go = st.button("✨ Generate", disabled=disabled, type="primary", key="style_generate")

    if go and uploaded and prompt:
        with st.spinner("Transforming… hang tight!"):
            result_img = run_inference(uploaded.read(), prompt)
            st.success("Done! Preview below ⤵")
            st.image(result_img, use_column_width="auto", caption="Final Creative")
            # Optional download
            buf = io.BytesIO()
            result_img.save(buf, format="PNG")
            st.download_button("Download PNG", data=buf.getvalue(), file_name="ad_image.png", mime="image/png")

# Tab 2: Image-Based Generation Feature
with tab2:
    st.markdown("### 🔍 Object Consistency Generator")
    st.markdown("Upload an image with an object and generate multiple versions of the same object in different scenarios while maintaining consistency.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Generation Settings")
        
        # Generation parameters
        st.subheader("AI Settings")
        num_steps = st.slider("Inference Steps", 5, 50, 10, help="More steps = higher quality but slower", key="analysis_steps")
        guidance_scale = st.slider("Guidance Scale", 1.0, 10.0, 3.5, step=0.5, help="Higher values = more prompt adherence", key="analysis_guidance")
        
        # Show analysis results
        if 'object_analysis' in st.session_state:
            st.subheader("📊 Object Analysis")
            analysis = st.session_state.object_analysis
            st.json(analysis)

    reference_uploaded = st.file_uploader("Upload an image with an object", type=["png", "jpg", "jpeg"], key="reference_uploader")

    if reference_uploaded:
        # Display reference image
        st.subheader("📷 Reference Object")
        reference_image = Image.open(reference_uploaded)
        st.image(reference_image, use_column_width="auto", caption="Reference Object")
        
        # Scenario inputs
        st.subheader("🎯 Define Scenarios")
        st.markdown("Enter different scenarios where you want to see the same object:")
        
        # Predefined scenarios
        preset_scenarios = [
            "on a beach",
            "in a modern kitchen",
            "on a wooden desk",
            "in a futuristic city",
            "in a cozy living room",
            "on a mountain top"
        ]
        
        # Let user select from presets or add custom ones
        selected_presets = st.multiselect(
            "Choose from preset scenarios:",
            preset_scenarios,
            default=["on a beach", "in a modern kitchen"]
        )
        
        # Custom scenarios
        custom_scenarios = st.text_area(
            "Add custom scenarios (one per line):",
            placeholder="in a coffee shop\nat a picnic table\nin a laboratory",
            help="Enter additional scenarios where you want to see the same object"
        )
        
        # Combine scenarios
        all_scenarios = selected_presets.copy()
        if custom_scenarios.strip():
            custom_list = [s.strip() for s in custom_scenarios.split('\n') if s.strip()]
            all_scenarios.extend(custom_list)
        
        # Show final scenarios
        if all_scenarios:
            st.subheader("📋 Final Scenarios")
            for i, scenario in enumerate(all_scenarios, 1):
                st.write(f"{i}. {scenario}")
        
        # Generate button
        generate_button = st.button(
            "✨ Generate Object Series", 
            type="primary", 
            disabled=len(all_scenarios) == 0, 
            key="consistency_generate"
        )
        
        if generate_button:
            with st.spinner(f"Analyzing object and generating {len(all_scenarios)} scenarios..."):
                try:
                    # Get the image deconstructor
                    deconstructor = get_openai_deconstructor()
                    
                    if deconstructor is None:
                        st.error("❌ Friendli AI not available. Please check your configuration.")
                    else:
                        # Process the image
                        image_bytes = reference_uploaded.read()
                        
                        # Debug: Check image bytes
                        print(f"Uploaded image bytes length: {len(image_bytes)}")
                        print(f"First 20 bytes: {image_bytes[:20]}")
                        print(f"Uploaded file name: {reference_uploaded.name}")
                        print(f"Uploaded file type: {reference_uploaded.type}")
                        
                        # Debug info
                        st.info(f"🔍 Processing reference object: {len(image_bytes)} bytes")
                        st.info(f"🎯 Generating {len(all_scenarios)} scenarios")
                        
                        results = deconstructor.generate_consistent_object_series(
                            image_bytes=image_bytes,
                            scenarios=all_scenarios,
                            num_inference_steps=num_steps,
                            guidance_scale=guidance_scale
                        )
                        
                        # Store analysis in session state
                        st.session_state.object_analysis = results["analysis"]
                        
                        # Display results
                        st.success("✅ Object series generated successfully!")
                        
                        # Show object analysis
                        st.subheader("📊 Object Analysis")
                        with st.expander("View detailed object analysis"):
                            st.json(results["analysis"])
                        # Show BLIP caption if available
                        blip_caption = results["analysis"].get("blip_caption", None)
                        if blip_caption:
                            st.info(f"📝 BLIP Caption: {blip_caption}")
                        # Show thumbnail if available
                        thumb_b64 = results["analysis"].get("thumbnail_b64", None)
                        if thumb_b64:
                            import base64
                            from PIL import Image as PILImage
                            import io as _io
                            thumb_bytes = base64.b64decode(thumb_b64)
                            thumb_img = PILImage.open(_io.BytesIO(thumb_bytes))
                            st.image(thumb_img, caption="Extracted Thumbnail", width=96)
                        # Show object description
                        st.subheader("🎯 Detected Object")
                        st.text_area(
                            "Object Description", 
                            results["object_description"], 
                            height=100, 
                            disabled=True
                        )
                        
                        # Show generated images
                        st.subheader("🖼️ Generated Object Series")
                        
                        # Create columns for displaying images
                        cols = st.columns(2)
                        col_index = 0
                        
                        for scenario, image_data in results["generated_images"].items():
                            with cols[col_index % 2]:
                                st.markdown(f"**{scenario}**")
                                
                                if image_data["url"]:
                                    st.info(f"📥 Downloading image for: {scenario}")
                                    
                                    # Download and display the generated image
                                    generated_image = download_image_from_url(image_data["url"])
                                    if generated_image:
                                        st.image(generated_image, use_column_width="auto", caption=scenario)
                                        
                                        # Download button for generated image
                                        img_buffer = io.BytesIO()
                                        generated_image.save(img_buffer, format="PNG")
                                        st.download_button(
                                            f"Download {scenario}",
                                            data=img_buffer.getvalue(),
                                            file_name=f"object_{scenario.replace(' ', '_')}.png",
                                            mime="image/png"
                                        )
                                        
                                        # Show the prompt used
                                        with st.expander(f"Prompt for {scenario}"):
                                            st.text_area("Generated Prompt", image_data["prompt"], height=100, disabled=True)
                                    else:
                                        st.error("❌ Failed to download image")
                                else:
                                    st.error(f"❌ Failed to generate image: {image_data.get('error', 'Unknown error')}")
                            
                            col_index += 1
                        
                except Exception as e:
                    st.error(f"❌ Error processing image: {e}")
                    st.info("🔧 Make sure you have all dependencies installed.")
                    
                    # Show detailed error for debugging
                    with st.expander("🔍 Debug Information"):
                        st.code(f"Error: {str(e)}\nType: {type(e)}")

# Tab 3: AI-Enhanced Generation Feature
with tab3:
    st.markdown("### 🤖 AI-Enhanced Image Generation")
    st.markdown("Upload an image and let OpenAI analyze it, then generate an enhanced version using the analysis.")
    
    if not OPENAI_API_KEY:
        st.error("❌ OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.")
        st.info("This feature requires OpenAI GPT-4o for image analysis.")
    else:
        # Sidebar for configuration
        with st.sidebar:
            st.header("⚙️ Enhanced Generation Settings")
            
            # Generation parameters
            st.subheader("AI Settings")
            enhanced_num_steps = st.slider("Inference Steps", 5, 50, 10, help="More steps = higher quality but slower", key="enhanced_steps")
            enhanced_guidance = st.slider("Guidance Scale", 1.0, 10.0, 3.5, step=0.5, help="Higher values = more prompt adherence", key="enhanced_guidance")
            
            # Show analysis results
            if 'enhanced_analysis' in st.session_state:
                st.subheader("📊 OpenAI Analysis")
                analysis = st.session_state.enhanced_analysis
                st.json(analysis)

        enhanced_uploaded = st.file_uploader("Upload an image for AI analysis", type=["png", "jpg", "jpeg"], key="enhanced_uploader")

        if enhanced_uploaded:
            # Display uploaded image
            st.subheader("📷 Original Image")
            original_image = Image.open(enhanced_uploaded)
            st.image(original_image, use_column_width="auto", caption="Original Image")
            
            # Style presets for enhanced generation
            st.subheader("🎨 Style Options")
            enhanced_preset_col1, enhanced_preset_col2, enhanced_preset_col3 = st.columns(3)
            
            enhanced_preset_map = {
                "Modern": "Modern",
                "Retro": "Retro", 
                "Cinematic": "Cinematic",
                "Professional": "Professional",
                "Artistic": "Artistic"
            }
            
            selected_style = None
            for label, col in zip(enhanced_preset_map.keys(), [enhanced_preset_col1, enhanced_preset_col2, enhanced_preset_col3]):
                with col:
                    if st.button(label, key=f"enhanced_{label}", use_container_width=True):
                        selected_style = enhanced_preset_map[label]
                        st.session_state["enhanced_style"] = selected_style
            
            # Custom prompt
            enhanced_custom = st.text_input(
                "Custom enhancement prompt (optional):", 
                placeholder="e.g. Make it more vibrant, add dramatic lighting, change background to cityscape",
                key="enhanced_custom_prompt"
            )
            
            # Show selected style
            if st.session_state.get("enhanced_style"):
                st.info(f"🎨 Selected style: {st.session_state['enhanced_style']}")
            
            # Generate button
            enhanced_generate = st.button(
                "🤖 Generate AI-Enhanced Image", 
                type="primary", 
                key="enhanced_generate"
            )
            
            if enhanced_generate:
                with st.spinner("🔍 Analyzing image with OpenAI and generating enhanced version..."):
                    try:
                        # Get the OpenAI deconstructor
                        openai_deconstructor = get_openai_deconstructor()
                        
                        if openai_deconstructor is None:
                            st.error("❌ OpenAI deconstructor not available. Please check your configuration.")
                        else:
                            # Process the image
                            image_bytes = enhanced_uploaded.read()
                            
                            # Get user inputs
                            user_prompt = enhanced_custom.strip()
                            style_preset = st.session_state.get("enhanced_style", "")
                            
                            # Process with enhanced pipeline
                            results = openai_deconstructor.process_image_with_enhanced_generation(
                                image_bytes=image_bytes,
                                user_prompt=user_prompt,
                                style_preset=style_preset,
                                num_inference_steps=enhanced_num_steps,
                                guidance_scale=enhanced_guidance
                            )
                            
                            # Store analysis in session state
                            st.session_state.enhanced_analysis = results["analysis"]
                            
                            # Display results
                            st.success("✅ AI-Enhanced image generated successfully!")
                            
                            # Show OpenAI analysis
                            st.subheader("📊 OpenAI Image Analysis")
                            with st.expander("View detailed OpenAI analysis"):
                                st.json(results["analysis"])
                            
                            # Show enhanced prompt
                            st.subheader("📝 Enhanced Prompt")
                            st.text_area(
                                "Generated Enhanced Prompt", 
                                results["enhanced_prompt"], 
                                height=150, 
                                disabled=True
                            )
                            
                            # Show generated image
                            st.subheader("🖼️ AI-Enhanced Result")
                            
                            generation_result = results["generation_result"]
                            if generation_result["success"] and generation_result["url"]:
                                # Download and display the generated image
                                enhanced_image = download_image_from_url(generation_result["url"])
                                if enhanced_image:
                                    st.image(enhanced_image, use_column_width="auto", caption="AI-Enhanced Image")
                                    
                                    # Download button for generated image
                                    img_buffer = io.BytesIO()
                                    enhanced_image.save(img_buffer, format="PNG")
                                    st.download_button(
                                        "Download Enhanced Image",
                                        data=img_buffer.getvalue(),
                                        file_name="ai_enhanced_image.png",
                                        mime="image/png"
                                    )
                                else:
                                    st.error("❌ Failed to download enhanced image")
                            else:
                                st.error(f"❌ Failed to generate enhanced image: {generation_result.get('error', 'Unknown error')}")
                            
                    except Exception as e:
                        st.error(f"❌ Error in AI-enhanced generation: {e}")
                        
                        # Show detailed error for debugging
                        with st.expander("🔍 Debug Information"):
                            st.code(f"Error: {str(e)}\nType: {type(e)}")

        # Instructions for AI-enhanced generation
        with st.expander("ℹ️ How to use AI-Enhanced Generation"):
            st.markdown("""
            **Step 1:** Upload an image you want to enhance
            
            **Step 2:** Choose a style preset or add a custom enhancement prompt
            
            **Step 3:** Click "Generate AI-Enhanced Image" to create an improved version
            
            **What OpenAI analyzes:**
            - Objects and their properties (color, material, position)
            - Text content and font styles
            - Visual style and composition
            - Background and setting
            - Branding elements and colors
            - Overall mood and lighting
            
            **Enhanced Features:**
            - Uses GPT-4o for comprehensive image analysis
            - Creates detailed prompts based on analysis
            - Maintains visual consistency with original elements
            - Applies style presets while preserving important details
            - Generates high-quality images with Friendli AI
            
            **Best for:**
            - Product photography enhancement
            - Brand consistency across different styles
            - Maintaining text and logo accuracy
            - Professional image transformations
            """)

# Installation instructions
with st.expander("🔧 Installation Requirements"):
    st.markdown("""
    **Required System Dependencies:**
    
    **macOS:**
    ```bash
    brew install tesseract
    ```
    
    **Ubuntu/Debian:**
    ```bash
    sudo apt update && sudo apt install tesseract-ocr
    ```
    
    **Windows:**
    1. Download Tesseract from: https://tesseract-ocr.github.io/tessdoc/Downloads.html
    2. Install and note the installation path
    3. Uncomment and modify the tesseract_cmd line in image_deconstruction.py
    
    **Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    
    **Environment Variables:**
    Create a `.env` file with:
    ```
    OPENAI_API_KEY=your_openai_api_key_here
    ```
    
    **API Keys Required:**
    - OpenAI API Key (for GPT-4o analysis)
    - Friendli AI Token (already configured)
    """)
