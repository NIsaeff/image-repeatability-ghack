import os
import io
import base64
import json
import hashlib
import tempfile
from PIL import Image, ImageOps
import pytesseract
from openai import OpenAI
import cv2
import numpy as np
import time
import math
from sklearn.cluster import KMeans
from collections import Counter
from ultralytics import YOLO
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

class EnhancedImageDeconstructor:
    def __init__(self, friendli_token, endpoint_id, openai_api_key=None):
        """
        Initialize the Enhanced Image Deconstructor with multiple AI capabilities.
        
        Args:
            friendli_token (str): Your Friendli AI API token
            endpoint_id (str): Your Friendli AI endpoint ID
            openai_api_key (str): Optional OpenAI API key for advanced analysis
        """
        self.friendli_token = friendli_token
        self.endpoint_id = endpoint_id
        self.openai_api_key = openai_api_key
        
        # Initialize OpenAI client if API key provided
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            self.openai_client = None
        
        # Initialize Friendli AI client
        self.friendli_client = OpenAI(
            base_url="https://api.friendli.ai/dedicated/v1",
            api_key=friendli_token,
        )
        
        # FLUX.1-schnell specific configuration
        self.flux_config = {
            "preferred_steps": 4,
            "preferred_guidance": 1.0,
            "max_steps": 8,
            "max_guidance": 2.0
        }
        
        # Initialize YOLO model for object detection
        try:
            self.yolo_model = YOLO('yolo8n.pt')  # Use nano model for speed
            print("✅ YOLO model loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load YOLO model: {e}")
            self.yolo_model = None
    
    def robust_load_image(self, image_bytes):
        # Try PIL
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()  # Verify it's a valid image
            img = Image.open(io.BytesIO(image_bytes))  # Reopen after verify
            return img
        except Exception as e:
            print(f"PIL failed: {e}")
        # Try OpenCV
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_cv is not None:
                return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"OpenCV failed: {e}")
        print("First 100 bytes:", image_bytes[:100])
        return None

    def get_blip_caption(self, pil_image):
        try:
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            inputs = processor(pil_image, return_tensors="pt")
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            print(f"BLIP captioning failed: {e}")
            return None

    def analyze_image_for_objects(self, image_bytes):
        """
        Robust image analysis: always extract some information, even if image is corrupted or unreadable.
        """
        # Compute hash of raw bytes for fallback
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        fallback = self._get_fallback_analysis("Image could not be loaded", image_hash=image_hash)
        try:
            image = self.robust_load_image(image_bytes)
            if image is None:
                # Try BLIP captioning as a last resort
                try:
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    caption = self.get_blip_caption(pil_image)
                    fallback["blip_caption"] = caption
                    fallback["object_description"] = caption or "object"
                except Exception as e:
                    print(f"BLIP fallback failed: {e}")
                return fallback
            # At this point, image is a valid PIL Image
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img_array
            # --- Robust Extraction ---
            thumb_b64 = self._extract_thumbnail_b64(image)
            palette = self._extract_color_palette(image)
            edge_b64 = self._extract_edge_map_b64(img_bgr)
            mean, std, hist = self._extract_low_level_stats(img_array)
            ocr_text = self._extract_text_robust(image)
            yolo_detection = self._yolo_object_detection(img_bgr)
            objects = self._detect_objects_enhanced(img_bgr, image)
            # BLIP caption as extra info
            blip_caption = self.get_blip_caption(image)
            analysis = {
                "size": image.size,
                "format": image.format,
                "mode": image.mode,
                "aspect_ratio": image.size[0] / image.size[1],
                "dominant_colors": palette,
                "brightness": self._analyze_brightness(img_bgr),
                "contrast": self._analyze_contrast(img_bgr),
                "composition": self._analyze_composition(img_bgr),
                "text": ocr_text,
                "objects": objects,
                "shape_analysis": self._analyze_shapes(img_bgr),
                "material_analysis": self._analyze_materials(img_bgr),
                "lighting_analysis": self._analyze_lighting(img_bgr),
                "perspective_analysis": self._analyze_perspective(img_bgr),
                "yolo_detection": yolo_detection,
                "object_description": self._generate_comprehensive_object_description({
                    "objects": objects,
                    "yolo_detection": yolo_detection,
                    "shape_analysis": self._analyze_shapes(img_bgr),
                    "material_analysis": self._analyze_materials(img_bgr),
                    "lighting_analysis": self._analyze_lighting(img_bgr),
                    "text": ocr_text,
                    "brightness": self._analyze_brightness(img_bgr),
                    "contrast": self._analyze_contrast(img_bgr)
                }),
                "thumbnail_b64": thumb_b64,
                "color_palette": palette,
                "edge_map_b64": edge_b64,
                "mean_pixel": mean,
                "std_pixel": std,
                "histogram": hist,
                "image_hash": image_hash,
                "blip_caption": blip_caption
            }
            # If object description is generic, use BLIP caption
            if (not analysis["object_description"] or analysis["object_description"] == "object") and blip_caption:
                analysis["object_description"] = blip_caption
            print(f"Robust object analysis completed successfully")
            return analysis
        except Exception as e:
            print(f"Error in robust image analysis: {e}")
            fallback["error"] = str(e)
            return fallback
    
    def _get_fallback_analysis(self, error_msg, image_hash=None):
        """Provide fallback analysis when main analysis fails"""
        return {
            "error": error_msg,
            "size": (0, 0),
            "format": "unknown",
            "mode": "unknown",
            "aspect_ratio": 1.0,
            "dominant_colors": [],
            "brightness": "unknown",
            "contrast": "unknown",
            "composition": [],
            "text": "No text detected",
            "objects": ["object"],
            "shape_analysis": {},
            "material_analysis": {},
            "lighting_analysis": {},
            "perspective_analysis": {},
            "yolo_detection": [],
            "object_description": "object",
            "thumbnail_b64": None,
            "color_palette": [],
            "edge_map_b64": None,
            "mean_pixel": None,
            "std_pixel": None,
            "histogram": None,
            "image_hash": image_hash
        }
    
    def _yolo_object_detection(self, img_bgr):
        """Use YOLO for advanced object detection"""
        try:
            if self.yolo_model is None:
                return []
            
            # Run YOLO detection
            results = self.yolo_model(img_bgr, verbose=False)
            
            detected_objects = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class name and confidence
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = result.names[cls]
                        
                        # Only include high-confidence detections
                        if conf > 0.5:
                            detected_objects.append({
                                "name": class_name,
                                "confidence": conf,
                                "class_id": cls
                            })
            
            return detected_objects
            
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            return []
    
    def _extract_thumbnail_b64(self, image, size=(64, 64)):
        try:
            thumb = image.copy()
            thumb.thumbnail(size)
            buf = io.BytesIO()
            thumb.save(buf, format='PNG')
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Thumbnail extraction failed: {e}")
            return None
    
    def _extract_color_palette(self, image, n_colors=5):
        try:
            small = image.copy()
            small.thumbnail((64, 64))
            arr = np.array(small).reshape(-1, 3)
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(arr)
            colors = kmeans.cluster_centers_.astype(int)
            return [tuple(map(int, c)) for c in colors]
        except Exception as e:
            print(f"Color palette extraction failed: {e}")
            return []
    
    def _extract_edge_map_b64(self, img_bgr):
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_img = Image.fromarray(edges)
            buf = io.BytesIO()
            edge_img.save(buf, format='PNG')
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Edge map extraction failed: {e}")
            return None
    
    def _extract_low_level_stats(self, img_array):
        try:
            mean = np.mean(img_array, axis=(0, 1)).tolist() if img_array.ndim == 3 else float(np.mean(img_array))
            std = np.std(img_array, axis=(0, 1)).tolist() if img_array.ndim == 3 else float(np.std(img_array))
            hist = None
            if img_array.ndim == 3:
                hist = [np.histogram(img_array[..., i], bins=16, range=(0, 255))[0].tolist() for i in range(3)]
            else:
                hist = np.histogram(img_array, bins=16, range=(0, 255))[0].tolist()
            return mean, std, hist
        except Exception as e:
            print(f"Low-level stats extraction failed: {e}")
            return None, None, None
    
    def _extract_text_robust(self, image):
        try:
            # Preprocessing for better OCR
            img = image.copy()
            # Convert to grayscale
            img = img.convert('L')
            # Increase contrast
            img = ImageOps.autocontrast(img)
            # Binarize
            img = img.point(lambda x: 0 if x < 128 else 255, '1')
            # Resize if too small
            if img.size[0] < 200 or img.size[1] < 50:
                scale = max(200 / img.size[0], 50 / img.size[1])
                new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
                img = img.resize(new_size, Image.LANCZOS)
            # Try multiple psm modes
            texts = []
            for psm in [6, 7, 11, 3]:
                try:
                    config = f'--psm {psm}'
                    text = pytesseract.image_to_string(img, config=config)
                    if text.strip():
                        texts.append(text.strip())
                except Exception as e:
                    continue
            # Return the longest non-empty result
            if texts:
                return max(texts, key=len)
            return "No text detected"
        except Exception as e:
            print(f"Robust OCR failed: {e}")
            return "No text detected"
    
    def _detect_objects_enhanced(self, img_bgr, image):
        """Enhanced object detection combining multiple methods"""
        try:
            objects = []
            
            # Get YOLO detections
            yolo_objects = self._yolo_object_detection(img_bgr)
            if yolo_objects:
                # Add YOLO detections to objects list
                for obj in yolo_objects:
                    objects.append(f"{obj['name']} (confidence: {obj['confidence']:.2f})")
            
            # Traditional color-based detection as fallback
            if not objects:
                objects = self._traditional_object_detection(img_bgr)
            
            return objects
            
        except Exception as e:
            print(f"Error in enhanced object detection: {e}")
            return ["object"]
    
    def _traditional_object_detection(self, img_bgr):
        """Traditional color-based object detection as fallback"""
        try:
            objects = []
            
            # Convert to HSV for color detection
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            height, width = img_bgr.shape[:2]
            
            # Enhanced color-based object detection
            color_ranges = {
                "red": ([0, 50, 50], [10, 255, 255]),
                "orange": ([10, 50, 50], [25, 255, 255]),
                "yellow": ([25, 50, 50], [35, 255, 255]),
                "green": ([40, 50, 50], [80, 255, 255]),
                "blue": ([100, 50, 50], [130, 255, 255]),
                "purple": ([130, 50, 50], [160, 255, 255]),
                "pink": ([160, 50, 50], [180, 255, 255])
            }
            
            for color_name, (lower, upper) in color_ranges.items():
                lower = np.array(lower)
                upper = np.array(upper)
                mask = cv2.inRange(hsv, lower, upper)
                
                if np.sum(mask) > 1000:
                    # Analyze shape of colored regions
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > 500:
                            x, y, w, h = cv2.boundingRect(contour)
                            aspect_ratio = w / h if h > 0 else 0
                            
                            # Classify object type based on shape and color
                            if 0.8 < aspect_ratio < 1.2:
                                objects.append(f"{color_name} cylindrical object")
                            elif aspect_ratio > 1.5:
                                objects.append(f"{color_name} rectangular object")
                            elif aspect_ratio < 0.7:
                                objects.append(f"{color_name} tall object")
                            else:
                                objects.append(f"{color_name} object")
            
            # Detect metallic/silver objects
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            if np.sum(thresh) > 1000:
                objects.append("metallic/silver object")
            
            # Detect transparent objects (glass, plastic)
            low_sat_mask = cv2.inRange(hsv, np.array([0, 0, 100]), np.array([180, 30, 255]))
            if np.sum(low_sat_mask) > 2000:
                objects.append("transparent object")
            
            return objects
            
        except Exception as e:
            print(f"Error in traditional object detection: {e}")
            return ["object"]
    
    def _analyze_brightness(self, img_bgr):
        """Analyze image brightness with more granular levels"""
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]
            
            mean_brightness = np.mean(v_channel)
            
            if mean_brightness < 60:
                return "very_dark"
            elif mean_brightness < 100:
                return "dark"
            elif mean_brightness < 140:
                return "medium_dark"
            elif mean_brightness < 180:
                return "medium"
            elif mean_brightness < 220:
                return "bright"
            else:
                return "very_bright"
        except Exception as e:
            print(f"Error analyzing brightness: {e}")
            return "unknown"
    
    def _analyze_contrast(self, img_bgr):
        """Analyze image contrast with more detail"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # Calculate multiple contrast measures
            contrast_std = np.std(gray)
            contrast_range = np.max(gray) - np.min(gray)
            
            # Calculate local contrast
            kernel = np.ones((3,3), np.float32) / 9
            blurred = cv2.filter2D(gray, -1, kernel)
            local_contrast = np.mean(np.abs(gray - blurred))
            
            # Combine measures
            if contrast_std < 20 and contrast_range < 100:
                return "very_low"
            elif contrast_std < 35 and contrast_range < 150:
                return "low"
            elif contrast_std < 50 and contrast_range < 200:
                return "medium"
            elif contrast_std < 70 and contrast_range < 250:
                return "high"
            else:
                return "very_high"
        except Exception as e:
            print(f"Error analyzing contrast: {e}")
            return "unknown"
    
    def _analyze_composition(self, img_bgr):
        """Enhanced composition analysis"""
        try:
            height, width = img_bgr.shape[:2]
            
            composition = []
            
            # Aspect ratio analysis
            aspect_ratio = width / height
            if aspect_ratio > 2.0:
                composition.append("ultra_wide")
            elif aspect_ratio > 1.5:
                composition.append("landscape")
            elif aspect_ratio > 1.2:
                composition.append("wide")
            elif aspect_ratio < 0.5:
                composition.append("ultra_tall")
            elif aspect_ratio < 0.8:
                composition.append("portrait")
            elif aspect_ratio < 0.9:
                composition.append("tall")
            else:
                composition.append("square")
            
            # Rule of thirds analysis
            third_w, third_h = width // 3, height // 3
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # Check if main object is at rule of thirds intersection
            center_region = gray[third_h:2*third_h, third_w:2*third_w]
            edge_region = gray.copy()
            edge_region[third_h:2*third_h, third_w:2*third_w] = 0
            
            if np.mean(center_region) > np.mean(edge_region):
                composition.append("centered")
            else:
                composition.append("rule_of_thirds")
            
            # Symmetry analysis
            left_half = gray[:, :width//2]
            right_half = cv2.flip(gray[:, width//2:], 1)
            
            if left_half.shape[1] == right_half.shape[1]:
                symmetry_score = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
                if symmetry_score > 0.8:
                    composition.append("highly_symmetrical")
                elif symmetry_score > 0.6:
                    composition.append("symmetrical")
                else:
                    composition.append("asymmetrical")
            
            return composition
        except Exception as e:
            print(f"Error analyzing composition: {e}")
            return ["unknown"]
    
    def _analyze_shapes(self, img_bgr):
        """Analyze geometric shapes in the image"""
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            shape_analysis = {
                "primary_shapes": [],
                "shape_count": len(contours),
                "dominant_shape": "unknown"
            }
            
            shape_scores = {"circle": 0, "rectangle": 0, "triangle": 0, "polygon": 0}
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 100:  # Skip small contours
                    continue
                
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Analyze shape based on number of vertices
                vertices = len(approx)
                
                if vertices == 3:
                    shape_scores["triangle"] += area
                elif vertices == 4:
                    # Check if it's a rectangle
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.8 < aspect_ratio < 1.2:
                        shape_scores["rectangle"] += area
                    else:
                        shape_scores["polygon"] += area
                elif vertices > 4 and vertices < 12:
                    # Check if it's approximately circular
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0
                    
                    if solidity > 0.8:
                        shape_scores["circle"] += area
                    else:
                        shape_scores["polygon"] += area
                else:
                    shape_scores["polygon"] += area
            
            # Determine dominant shape
            if shape_scores:
                dominant_shape = max(shape_scores, key=shape_scores.get)
                shape_analysis["dominant_shape"] = dominant_shape
                shape_analysis["primary_shapes"] = [shape for shape, score in shape_scores.items() if score > 0]
            
            return shape_analysis
        except Exception as e:
            print(f"Error analyzing shapes: {e}")
            return {"primary_shapes": [], "shape_count": 0, "dominant_shape": "unknown"}
    
    def _analyze_materials(self, img_bgr):
        """Analyze material properties based on texture and reflectivity"""
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            material_analysis = {
                "texture": "unknown",
                "reflectivity": "unknown",
                "material_type": "unknown"
            }
            
            # Analyze texture using Local Binary Patterns or simple variance
            kernel = np.ones((5,5), np.float32) / 25
            blurred = cv2.filter2D(gray, -1, kernel)
            texture_variance = np.var(gray - blurred)
            
            if texture_variance < 100:
                material_analysis["texture"] = "smooth"
            elif texture_variance < 300:
                material_analysis["texture"] = "slightly_textured"
            elif texture_variance < 600:
                material_analysis["texture"] = "textured"
            else:
                material_analysis["texture"] = "highly_textured"
            
            # Analyze reflectivity using brightness distribution
            brightness_std = np.std(gray)
            if brightness_std < 30:
                material_analysis["reflectivity"] = "matte"
            elif brightness_std < 60:
                material_analysis["reflectivity"] = "semi_glossy"
            else:
                material_analysis["reflectivity"] = "glossy"
            
            # Determine material type based on combination
            if material_analysis["texture"] == "smooth" and material_analysis["reflectivity"] == "glossy":
                material_analysis["material_type"] = "metallic"
            elif material_analysis["texture"] == "smooth" and material_analysis["reflectivity"] == "matte":
                material_analysis["material_type"] = "plastic"
            elif material_analysis["texture"] == "textured":
                material_analysis["material_type"] = "fabric"
            elif material_analysis["texture"] == "highly_textured":
                material_analysis["material_type"] = "rough_surface"
            
            return material_analysis
        except Exception as e:
            print(f"Error analyzing materials: {e}")
            return {"texture": "unknown", "reflectivity": "unknown", "material_type": "unknown"}
    
    def _analyze_lighting(self, img_bgr):
        """Analyze lighting conditions and shadows"""
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            lighting_analysis = {
                "lighting_type": "unknown",
                "shadow_intensity": "unknown",
                "light_direction": "unknown"
            }
            
            # Analyze overall brightness distribution
            mean_brightness = np.mean(gray)
            brightness_std = np.std(gray)
            
            # Determine lighting type
            if brightness_std < 20:
                lighting_analysis["lighting_type"] = "flat_lighting"
            elif brightness_std < 40:
                lighting_analysis["lighting_type"] = "soft_lighting"
            elif brightness_std < 70:
                lighting_analysis["lighting_type"] = "dramatic_lighting"
            else:
                lighting_analysis["lighting_type"] = "high_contrast_lighting"
            
            # Analyze shadows
            # Use histogram analysis to detect shadow regions
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            dark_pixels = np.sum(hist[:50])  # Very dark pixels
            total_pixels = np.sum(hist)
            shadow_ratio = dark_pixels / total_pixels if total_pixels > 0 else 0
            
            if shadow_ratio < 0.1:
                lighting_analysis["shadow_intensity"] = "minimal_shadows"
            elif shadow_ratio < 0.25:
                lighting_analysis["shadow_intensity"] = "soft_shadows"
            elif shadow_ratio < 0.4:
                lighting_analysis["shadow_intensity"] = "moderate_shadows"
            else:
                lighting_analysis["shadow_intensity"] = "strong_shadows"
            
            return lighting_analysis
        except Exception as e:
            print(f"Error analyzing lighting: {e}")
            return {"lighting_type": "unknown", "shadow_intensity": "unknown", "light_direction": "unknown"}
    
    def _analyze_perspective(self, img_bgr):
        """Analyze perspective and depth cues"""
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            perspective_analysis = {
                "perspective_type": "unknown",
                "depth_cues": [],
                "viewpoint": "unknown"
            }
            
            # Simple perspective analysis based on vanishing points
            # This is a simplified version - more sophisticated analysis would use Hough lines
            
            # Check for strong horizontal and vertical lines
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                horizontal_lines = 0
                vertical_lines = 0
                
                for line in lines:
                    rho, theta = line[0]
                    if theta < np.pi/6 or theta > 5*np.pi/6:
                        vertical_lines += 1
                    elif np.pi/3 < theta < 2*np.pi/3:
                        horizontal_lines += 1
                
                if vertical_lines > horizontal_lines * 2:
                    perspective_analysis["perspective_type"] = "vertical_dominance"
                elif horizontal_lines > vertical_lines * 2:
                    perspective_analysis["perspective_type"] = "horizontal_dominance"
                else:
                    perspective_analysis["perspective_type"] = "balanced"
            
            # Analyze depth cues
            # Check for size gradients (objects getting smaller towards edges)
            height, width = gray.shape
            center_region = gray[height//4:3*height//4, width//4:3*width//4]
            edge_region = gray.copy()
            edge_region[height//4:3*height//4, width//4:3*width//4] = 0
            
            if np.mean(center_region) > np.mean(edge_region) * 1.2:
                perspective_analysis["depth_cues"].append("size_gradient")
            
            return perspective_analysis
        except Exception as e:
            print(f"Error analyzing perspective: {e}")
            return {"perspective_type": "unknown", "depth_cues": [], "viewpoint": "unknown"}
    
    def _generate_comprehensive_object_description(self, analysis):
        """Generate a comprehensive object description optimized for FLUX.1-schnell"""
        try:
            description_parts = []
            
            # Get basic object information
            objects = analysis.get("objects", [])
            yolo_detection = analysis.get("yolo_detection", [])
            shape_analysis = analysis.get("shape_analysis", {})
            material_analysis = analysis.get("material_analysis", {})
            lighting_analysis = analysis.get("lighting_analysis", {})
            text = analysis.get("text", "")
            
            # Start with object type - prioritize YOLO detections
            if yolo_detection:
                # Use the highest confidence YOLO detection
                best_detection = max(yolo_detection, key=lambda x: x['confidence'])
                description_parts.append(best_detection['name'])
            elif objects:
                # Use the most specific object description
                primary_object = objects[0] if objects else "object"
                description_parts.append(primary_object)
            else:
                # Fallback based on shape analysis
                dominant_shape = shape_analysis.get("dominant_shape", "object")
                if dominant_shape == "circle":
                    description_parts.append("circular object")
                elif dominant_shape == "rectangle":
                    description_parts.append("rectangular object")
                elif dominant_shape == "triangle":
                    description_parts.append("triangular object")
                else:
                    description_parts.append("object")
            
            # Add material properties
            material_type = material_analysis.get("material_type", "")
            if material_type and material_type != "unknown":
                description_parts.append(material_type)
            
            texture = material_analysis.get("texture", "")
            if texture and texture != "unknown":
                description_parts.append(texture)
            
            # Add lighting information
            lighting_type = lighting_analysis.get("lighting_type", "")
            if lighting_type and lighting_type != "unknown":
                description_parts.append(f"with {lighting_type}")
            
            # Add text information
            if text and text != "No text detected" and not text.startswith("Error"):
                # Clean and truncate text for prompt
                clean_text = text.replace('\n', ' ').replace('\r', ' ').strip()
                if len(clean_text) > 50:
                    clean_text = clean_text[:50] + "..."
                description_parts.append(f"with text '{clean_text}'")
            
            # Add visual style characteristics
            brightness = analysis.get("brightness", "")
            contrast = analysis.get("contrast", "")
            
            if brightness and brightness != "unknown":
                description_parts.append(f"{brightness} lighting")
            
            if contrast and contrast != "unknown":
                description_parts.append(f"{contrast} contrast")
            
            # Ensure we have a meaningful description
            if not description_parts:
                description_parts = ["object"]
            
            final_description = " ".join(description_parts)
            
            # Optimize for FLUX.1-schnell by keeping it concise but descriptive
            if len(final_description) > 100:
                # Truncate while keeping essential parts
                parts = final_description.split()
                if len(parts) > 8:
                    final_description = " ".join(parts[:8]) + "..."
            
            return final_description
            
        except Exception as e:
            print(f"Error generating comprehensive object description: {e}")
            return "object"
    
    def generate_flux_optimized_prompt(self, analysis, new_scenario):
        """
        Generate a prompt specifically optimized for FLUX.1-schnell model.
        
        Args:
            analysis (dict): Object analysis results
            new_scenario (str): New scenario/background description
            
        Returns:
            str: FLUX-optimized prompt
        """
        try:
            # Get the comprehensive object description
            object_desc = analysis.get("object_description", "object")
            
            # Extract key visual characteristics
            brightness = analysis.get("brightness", "")
            contrast = analysis.get("contrast", "")
            material_analysis = analysis.get("material_analysis", {})
            lighting_analysis = analysis.get("lighting_analysis", {})
            
            # Build FLUX-optimized prompt
            prompt_parts = []
            
            # Main object with high emphasis (FLUX responds well to clear object descriptions)
            prompt_parts.append(f"a {object_desc}")
            
            # New scenario
            prompt_parts.append(f"in {new_scenario}")
            
            # Add material and surface properties (FLUX is good at materials)
            material_type = material_analysis.get("material_type", "")
            if material_type and material_type != "unknown":
                prompt_parts.append(f"made of {material_type}")
            
            # Add lighting (FLUX handles lighting well)
            lighting_type = lighting_analysis.get("lighting_type", "")
            if lighting_type and lighting_type != "unknown":
                prompt_parts.append(f"with {lighting_type}")
            
            # Add quality descriptors that FLUX responds to
            prompt_parts.append("high quality, detailed, professional photography")
            
            # FLUX-specific optimizations
            prompt_parts.append("sharp focus, clear details")
            
            final_prompt = ", ".join(prompt_parts)
            
            # FLUX works best with concise but descriptive prompts
            if len(final_prompt) > 150:
                # Prioritize object and scenario, then add key visual elements
                core_parts = [f"a {object_desc}", f"in {new_scenario}", "high quality, detailed"]
                final_prompt = ", ".join(core_parts)
            
            print(f"Generated FLUX-optimized prompt: {final_prompt}")
            return final_prompt
            
        except Exception as e:
            print(f"Error generating FLUX-optimized prompt: {e}")
            return f"a {analysis.get('object_description', 'object')} in {new_scenario}, high quality"
    
    def generate_image(self, prompt, num_inference_steps=None, guidance_scale=None):
        """
        Generate an image using Friendli AI API with FLUX.1-schnell optimizations.
        
        Args:
            prompt (str): The prompt for image generation
            num_inference_steps (int): Number of inference steps (None for FLUX defaults)
            guidance_scale (float): Guidance scale for generation (None for FLUX defaults)
            
        Returns:
            str: URL of the generated image or None if error
        """
        # Use FLUX.1-schnell optimized parameters if not specified
        if num_inference_steps is None:
            num_inference_steps = self.flux_config["preferred_steps"]
        if guidance_scale is None:
            guidance_scale = self.flux_config["preferred_guidance"]
        
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                print(f"Calling Friendli AI API with FLUX.1-schnell... (Attempt {attempt + 1}/{max_retries})")
                print(f"Endpoint ID: {self.endpoint_id}")
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
                    return image_url
                else:
                    print("No image URL in response")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        return None
                
            except Exception as e:
                print(f"Error generating image (Attempt {attempt + 1}): {e}")
                print(f"Error type: {type(e)}")
                print(f"Error details: {str(e)}")
                
                # Check if it's an endpoint availability issue
                if "Endpoint is unavailable" in str(e) or "endpoint" in str(e).lower():
                    if attempt < max_retries - 1:
                        print(f"Endpoint unavailable, retrying in {retry_delay * (attempt + 1)} seconds...")
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        print("Endpoint still unavailable after all retries")
                        return None
                else:
                    # For other errors, don't retry
                    return None
        
        return None
    
    def generate_consistent_object_series(self, image_bytes, scenarios):
        """
        Generate multiple images of the same object in different scenarios using FLUX.1-schnell.
        If the main prompt fails, use the fallback correlated prompt to guarantee correlation with the uploaded image.
        """
        print(f"Starting FLUX.1-schnell consistent object series generation...")
        print(f"Image bytes length: {len(image_bytes)}")
        print(f"Scenarios: {scenarios}")
        # Step 1: Analyze the uploaded image for object consistency
        analysis = self.analyze_image_for_objects(image_bytes)
        # Ensure we have a valid object description
        if "error" in analysis or not analysis.get("object_description"):
            analysis["object_description"] = "object"
        # Step 2: Generate images for each scenario
        generated_images = {}
        for i, scenario in enumerate(scenarios):
            print(f"Generating image {i+1}/{len(scenarios)} for scenario: {scenario}")
            # Generate FLUX-optimized prompt for this scenario
            prompt = self.generate_flux_optimized_prompt(analysis, scenario)
            # Generate image with FLUX-optimized parameters
            image_url = self.generate_image(prompt)
            if image_url:
                generated_images[scenario] = {
                    "prompt": prompt,
                    "url": image_url
                }
            else:
                # If generation fails, try with a fallback correlated prompt
                fallback_prompt = self.generate_fallback_correlated_prompt(analysis, scenario)
                print(f"Trying fallback correlated prompt: {fallback_prompt}")
                fallback_image_url = self.generate_image(fallback_prompt)
                if fallback_image_url:
                    generated_images[scenario] = {
                        "prompt": fallback_prompt,
                        "url": fallback_image_url
                    }
                else:
                    generated_images[scenario] = {
                        "prompt": fallback_prompt,
                        "url": None,
                        "error": "Failed to generate image after retries (including fallback prompt)"
                    }
        return {
            "analysis": analysis,
            "object_description": analysis.get("object_description", "object"),
            "generated_images": generated_images
        }
    
    def generate_fallback_correlated_prompt(self, analysis, new_scenario):
        """
        Generate a prompt for image-based generation that always uses extracted features (color palette, edge map, OCR, etc.)
        so the generated image is correlated to the uploaded image, even if object detection fails.
        """
        try:
            prompt_parts = []
            # Use object description if available
            object_desc = analysis.get("object_description", "object")
            if object_desc and object_desc != "object":
                prompt_parts.append(f"a {object_desc}")
            # Use BLIP caption if available
            blip_caption = analysis.get("blip_caption", None)
            if blip_caption and blip_caption != object_desc:
                prompt_parts.append(f"BLIP description: {blip_caption}")
            # Use OCR text if available
            text = analysis.get("text", "")
            if text and text != "No text detected":
                prompt_parts.append(f"with the text '{text}'")
            # Use color palette
            palette = analysis.get("color_palette", [])
            if palette:
                hex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in palette]
                prompt_parts.append(f"with a color palette of {', '.join(hex_colors)}")
            # Use edge/shape info
            shape_info = analysis.get("shape_analysis", {})
            if shape_info and shape_info.get("dominant_shape") and shape_info.get("dominant_shape") != "unknown":
                prompt_parts.append(f"with prominent {shape_info['dominant_shape']} shapes")
            # Use brightness/contrast
            brightness = analysis.get("brightness", "")
            if brightness and brightness != "unknown":
                prompt_parts.append(f"{brightness} lighting")
            contrast = analysis.get("contrast", "")
            if contrast and contrast != "unknown":
                prompt_parts.append(f"{contrast} contrast")
            # Add scenario
            if new_scenario:
                prompt_parts.append(f"in {new_scenario}")
            # Always add a fallback
            if not prompt_parts:
                prompt_parts.append("an abstract image inspired by the uploaded photo")
            # Add quality descriptors
            prompt_parts.append("high quality, detailed, professional photography")
            prompt = ", ".join(prompt_parts)
            return prompt
        except Exception as e:
            print(f"Error generating fallback correlated prompt: {e}")
            return f"an abstract image inspired by the uploaded photo in {new_scenario}, high quality" 
