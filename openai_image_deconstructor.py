import os
import io
import base64
import json
import tempfile
from PIL import Image
from openai import OpenAI
import streamlit as st
import requests

class OpenAIImageDeconstructor:
    def __init__(self, openai_api_key, friendli_token, endpoint_id):
        """
        Initialize the OpenAI Image Deconstructor with both OpenAI and Friendli AI credentials.
        
        Args:
            openai_api_key (str): Your OpenAI API key
            friendli_token (str): Your Friendli AI API token
            endpoint_id (str): Your Friendli AI endpoint ID
        """
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.friendli_token = friendli_token
        self.endpoint_id = endpoint_id
        
        # Initialize Friendli AI client
        self.friendli_client = OpenAI(
            base_url="https://api.friendli.ai/dedicated/v1",
            api_key=friendli_token,
        )
    
    def upload_image_to_temp_host(self, image_bytes):
        """
        Upload image to a temporary hosting service to get a public URL.
        
        Args:
            image_bytes (bytes): Image data as bytes
            
        Returns:
            str: Public URL of the uploaded image
        """
        try:
            # Use imgbb.com API (free, no registration required)
            url = "https://api.imgbb.com/1/upload"
            
            # Convert image to JPEG if needed
            from PIL import Image
            import io
            
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            jpeg_buffer = io.BytesIO()
            image.save(jpeg_buffer, format='JPEG', quality=95)
            jpeg_bytes = jpeg_buffer.getvalue()
            
            # Encode to base64
            image_base64 = base64.b64encode(jpeg_bytes).decode('utf-8')
            
            data = {
                "key": "2c0d3c0e8b0c0c0c0c0c0c0c0c0c0c0c",  # Free API key
                "image": image_base64
            }
            
            response = requests.post(url, data=data)
            response.raise_for_status()
            
            result = response.json()
            if result["success"]:
                return result["data"]["url"]
            else:
                raise Exception(f"Failed to upload image: {result}")
                
        except Exception as e:
            print(f"Error uploading image: {e}")
            # Fallback: try using a different service or return None
            return None
    
    def deconstruct_image_with_openai(self, image_bytes: bytes) -> dict:
        """
        Use OpenAI's GPT-4o to analyze an image with a simple prompt.
        
        Args:
            image_bytes (bytes): Image data as bytes
            
        Returns:
            dict: Simple analysis with overall_description
        """
        try:
            # Check if image_bytes is valid
            if not image_bytes or len(image_bytes) == 0:
                raise ValueError("Empty image bytes provided")
            
            print(f"Original image bytes length: {len(image_bytes)}")
            
            # Upload image to get a public URL
            print("Uploading image to temporary host...")
            public_url = self.upload_image_to_temp_host(image_bytes)
            
            if not public_url:
                raise ValueError("Failed to upload image to temporary host")
            
            print(f"Image uploaded successfully: {public_url}")
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe in detail what objects you see in this image. Include colors, materials, shapes, sizes, and any text or logos. Be specific about the main objects and their characteristics."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": public_url
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            # Extract the simple text response
            analysis_text = response.choices[0].message.content
            
            print(f"OpenAI response: {analysis_text}")
            
            # Return a simple structure with the description
            return {
                "overall_description": analysis_text,
                "objects": [],
                "text_content": {"visible_text": "", "font_style": "", "text_position": ""},
                "visual_style": {"lighting": "", "composition": "", "color_palette": "", "mood": "", "style": ""},
                "background": {"description": "", "setting": ""},
                "branding": {"logos": "", "brand_colors": ""}
            }
            
        except Exception as e:
            st.error(f"Error in OpenAI image analysis: {e}")
            print(f"Detailed error: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "overall_description": "Error analyzing image",
                "objects": [],
                "text_content": {"visible_text": "", "font_style": "", "text_position": ""},
                "visual_style": {"lighting": "", "composition": "", "color_palette": "", "mood": "", "style": ""},
                "background": {"description": "", "setting": ""},
                "branding": {"logos": "", "brand_colors": ""}
            }
    
    def generate_image_with_friendli(self, prompt: str, num_inference_steps=10, guidance_scale=3.5) -> dict:
        """
        Generate an image using Friendli AI with the given prompt.
        
        Args:
            prompt (str): The text prompt for image generation
            num_inference_steps (int): Number of inference steps
            guidance_scale (float): Guidance scale for generation
            
        Returns:
            dict: Dictionary containing the generated image URL and metadata
        """
        try:
            print(f"Calling Friendli AI API...")
            print(f"Endpoint ID: {self.endpoint_id}")
            print(f"Token length: {len(self.friendli_token)}")
            print(f"Prompt: {prompt}")
            print(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}")
            
            response = self.friendli_client.images.generate(
                model=self.endpoint_id,
                prompt=prompt,
                extra_body={
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale
                }
            )
            
            print(f"API Response received: {response}")
            
            if response.data and response.data[0].url:
                image_url = response.data[0].url
                print(f"Image URL: {image_url}")
                return {
                    "url": image_url,
                    "prompt": prompt,
                    "success": True
                }
            else:
                print("No image URL received from Friendli AI")
                return {
                    "url": None,
                    "prompt": prompt,
                    "error": "No image URL received",
                    "success": False
                }
                
        except Exception as e:
            print(f"Error generating image: {e}")
            return {
                "url": None,
                "prompt": prompt,
                "error": str(e),
                "success": False
            }
    
    def create_enhanced_prompt(self, analysis: dict, user_prompt: str = "", style_preset: str = "") -> str:
        """
        Create an enhanced prompt using the OpenAI analysis and user input.
        
        Args:
            analysis (dict): The OpenAI analysis results
            user_prompt (str): User's custom prompt
            style_preset (str): Preset style (Modern, Retro, Cinematic, etc.)
            
        Returns:
            str: Enhanced prompt for image generation
        """
        # Extract key information from analysis
        objects = analysis.get("objects", [])
        text_content = analysis.get("text_content", {})
        visual_style = analysis.get("visual_style", {})
        background = analysis.get("background", {})
        branding = analysis.get("branding", {})
        overall_description = analysis.get("overall_description", "")
        
        # Build the enhanced prompt
        enhanced_prompt_parts = []
        
        # Add user prompt if provided
        if user_prompt.strip():
            enhanced_prompt_parts.append(user_prompt.strip())
        
        # Add object descriptions
        if objects:
            object_descriptions = []
            for obj in objects:
                obj_desc = f"{obj.get('name', 'object')}"
                if obj.get('color'):
                    obj_desc += f" in {obj.get('color')}"
                if obj.get('material'):
                    obj_desc += f" with {obj.get('material')} texture"
                object_descriptions.append(obj_desc)
            
            if object_descriptions:
                enhanced_prompt_parts.append(f"Featuring: {', '.join(object_descriptions)}")
        
        # Add text content if present
        if text_content.get("visible_text"):
            enhanced_prompt_parts.append(f"Text: '{text_content['visible_text']}' - {text_content.get('font_style', 'clear, readable font')}")
        
        # Add branding information
        if branding.get("logos"):
            enhanced_prompt_parts.append(f"Brand elements: {branding['logos']}")
        
        # Add background/setting
        if background.get("description"):
            enhanced_prompt_parts.append(f"Background: {background['description']}")
        
        # Add style preset
        if style_preset:
            style_prompts = {
                "Modern": "Ultra-clean, minimalist, bright studio lighting, crisp focus, white backdrop",
                "Retro": "Vintage 1970s magazine aesthetic, warm film grain, subtle halftone texture, soft vignette",
                "Cinematic": "High-contrast teal-and-orange grade, shallow depth of field, dramatic rim lighting",
                "Professional": "High-quality professional photography, studio lighting, clean composition",
                "Artistic": "Creative artistic style, vibrant colors, expressive composition"
            }
            if style_preset in style_prompts:
                enhanced_prompt_parts.append(style_prompts[style_preset])
        
        # Add visual style information
        if visual_style.get("lighting"):
            enhanced_prompt_parts.append(f"Lighting: {visual_style['lighting']}")
        if visual_style.get("mood"):
            enhanced_prompt_parts.append(f"Mood: {visual_style['mood']}")
        
        # Combine all parts
        final_prompt = ". ".join(enhanced_prompt_parts)
        
        # Add quality and consistency instructions
        final_prompt += ". High quality, professional photography, maintain visual consistency with original elements"
        
        return final_prompt
    
    def process_image_with_enhanced_generation(self, image_bytes: bytes, user_prompt: str = "", style_preset: str = "", num_inference_steps=10, guidance_scale=3.5) -> dict:
        """
        Complete pipeline: deconstruct image with OpenAI, then generate enhanced image with Friendli AI.
        
        Args:
            image_bytes (bytes): Input image data
            user_prompt (str): User's custom prompt
            style_preset (str): Style preset to apply
            num_inference_steps (int): Number of inference steps
            guidance_scale (float): Guidance scale
            
        Returns:
            dict: Complete results including analysis and generated image
        """
        # Step 1: Deconstruct image with OpenAI
        st.info("🔍 Analyzing image with OpenAI GPT-4o...")
        analysis = self.deconstruct_image_with_openai(image_bytes)
        
        # Step 2: Create enhanced prompt
        st.info("📝 Creating enhanced prompt...")
        enhanced_prompt = self.create_enhanced_prompt(analysis, user_prompt, style_preset)
        
        # Step 3: Generate image with Friendli AI
        st.info("🎨 Generating enhanced image...")
        generation_result = self.generate_image_with_friendli(
            enhanced_prompt, 
            num_inference_steps, 
            guidance_scale
        )
        
        return {
            "analysis": analysis,
            "enhanced_prompt": enhanced_prompt,
            "generation_result": generation_result,
            "original_prompt": user_prompt,
            "style_preset": style_preset
        }
    
    def generate_consistent_object_series(self, image_bytes, scenarios, num_inference_steps=10, guidance_scale=3.5):
        """
        Generate multiple images of the same object in different scenarios using OpenAI for analysis and Friendli for generation.
        
        Args:
            image_bytes (bytes): Original image data
            scenarios (list): List of scenario descriptions
            num_inference_steps (int): Number of inference steps for Friendli
            guidance_scale (float): Guidance scale for Friendli
        
        Returns:
            dict: Results containing analysis and generated image URLs
        """
        st.info(f"🔍 Analyzing object with OpenAI GPT-4o for all scenarios...")
        analysis = self.deconstruct_image_with_openai(image_bytes)
        object_description = analysis.get("overall_description", "object")

        generated_images = {}
        for scenario in scenarios:
            # Compose a scenario-specific prompt
            prompt = f"{object_description}. Place this object {scenario}. High quality, professional photography, maintain visual consistency with original elements."
            st.info(f"🎨 Generating: {prompt}")
            result = self.generate_image_with_friendli(prompt, num_inference_steps, guidance_scale)
            generated_images[scenario] = {
                "prompt": prompt,
                "url": result.get("url"),
                "error": result.get("error") if not result.get("success") else None
            }

        return {
            "analysis": analysis,
            "object_description": object_description,
            "generated_images": generated_images
        } 
