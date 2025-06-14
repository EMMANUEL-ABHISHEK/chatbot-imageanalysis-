import torch
import logging
import os
import time
from typing import Tuple, Dict, Any, List
from PIL import Image
import torch.nn.functional as F
import io
from dotenv import load_dotenv
import timm
import torchvision.transforms as transforms

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_service")

class ModelService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conversation_history = {}
        self.image_context = {}
        self.max_history_length = 10

        # API keys
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")

        # Initialize all services
        self._init_services()

    def _init_services(self):
        """Initialize all APIs and models"""
        # Initialize API availability flags
        self.gemini_client = None
        self.gemini_available = False
        self.google_vision_available = False
        self.groq_client = None
        self.groq_available = False
        self.blip_available = False
        self.clip_available = False
        self.vit_available = False

        # Initialize APIs
        if self.gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_client = genai.GenerativeModel('gemini-1.5-flash')
                self.gemini_available = True
            except Exception as e:
                logger.warning(f"Gemini initialization failed: {e}")

        if self.google_api_key:
            try:
                import google.generativeai as genai
                self.google_vision_available = True
            except Exception as e:
                logger.warning(f"Google Vision initialization failed: {e}")

        if self.groq_api_key:
            try:
                import groq
                self.groq_client = groq.Groq(api_key=self.groq_api_key)
                self.groq_available = True
            except Exception as e:
                logger.warning(f"Groq initialization failed: {e}")

        # Initialize models
        try:
            self._load_blip()
            self._load_clip()
            self._load_vit()
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            
    def _load_blip(self):
        """Load BLIP model for image captioning"""
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
            self.blip_available = True
        except Exception as e:
            logger.warning(f"BLIP loading failed: {e}")
            self.blip_processor = None
            self.blip_model = None

    def _load_clip(self):
        """Load CLIP model for multimodal understanding"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_available = True
        except Exception as e:
            logger.warning(f"CLIP loading failed: {e}")
            self.clip_processor = None
            self.clip_model = None

    def _load_vit(self):
        """Load Vision Transformer model for feature extraction"""
        try:
            self.vit_model = timm.create_model('vit_base_patch16_224', pretrained=True)
            self.vit_model.eval().to(self.device)
            self.vit_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.vit_available = True
        except Exception as e:
            logger.warning(f"ViT loading failed: {e}")
            self.vit_model = None
            self.vit_transform = None

    def generate_text_response(self, text: str, session_id: str, with_image_context: bool = False) -> Tuple[str, float]:
        """Generate a response using available APIs or local models"""
        try:
            self._update_conversation_history(session_id, text)
            context = self._get_image_context(session_id) if with_image_context else ""

            if self.gemini_available:
                return self._generate_with_gemini(text, session_id, context)
            elif self.groq_available:
                return self._generate_with_groq(text, session_id, context)
            else:
                return self._generate_fallback_response(text, session_id)
        except Exception as e:
            logger.error(f"Error in text generation: {e}")
            return "I'm sorry, I encountered an error. Please try again.", 0.0

    def _update_conversation_history(self, session_id: str, text: str):
        """Update conversation history for session"""
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        self.conversation_history[session_id].append({"role": "user", "content": text})
        if len(self.conversation_history[session_id]) > self.max_history_length:
            self.conversation_history[session_id] = self.conversation_history[session_id][-self.max_history_length:]

    def _get_image_context(self, session_id: str) -> str:
        """Get image context for session"""
        if session_id in self.image_context:
            image_info = self.image_context[session_id]
            return f"[Image Context: {image_info.get('caption', 'No caption available')}] "
        return ""
    


    def _generate_with_gemini(self, text: str, session_id: str, context: str) -> Tuple[str, float]:
        """Generate response using Gemini API (Primary)"""
        try:
            # Build conversation context
            conversation_context = ""
            if session_id in self.conversation_history:
                recent_messages = self.conversation_history[session_id][-5:]
                for msg in recent_messages[:-1]:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    conversation_context += f"{role}: {msg['content']}\n"

            # Prepare prompt
            prompt = context + (f"Previous conversation:\n{conversation_context}\n" if conversation_context else "") + f"User: {text}\nAssistant:"

            response = self.gemini_client.generate_content(
                prompt,
                generation_config={'temperature': 0.7, 'max_output_tokens': 150, 'top_p': 0.9}
            )

            response_text = response.text.strip()
            self.conversation_history[session_id].append({"role": "assistant", "content": response_text})
            return response_text, 0.95
        except Exception as e:
            logger.warning(f"Gemini failed: {e}")
            raise

    def _generate_with_groq(self, text: str, session_id: str, context: str) -> Tuple[str, float]:
        """Generate response using Groq API"""
        try:
            messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
            messages.extend(self.conversation_history[session_id][-5:])
            if context:
                messages[-1]["content"] = context + messages[-1]["content"]

            response = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=messages,
                max_tokens=150,
                temperature=0.7
            )

            response_text = response.choices[0].message.content.strip()
            self.conversation_history[session_id].append({"role": "assistant", "content": response_text})
            return response_text, 0.9
        except Exception as e:
            logger.warning(f"Groq failed: {e}")
            raise
    

    def _generate_fallback_response(self, text: str, session_id: str) -> Tuple[str, float]:
        """Generate fallback response when no models available"""
        fallback_responses = [
            "I understand. Could you tell me more?",
            "That's interesting! What would you like to know?",
            "I'm here to help. What can I assist you with?",
            "Thanks for sharing. How can I help further?",
            "I see. What would you like to explore about this?"
        ]
        response_text = fallback_responses[hash(text) % len(fallback_responses)]
        self.conversation_history[session_id].append({"role": "assistant", "content": response_text})
        return response_text, 0.5

    def process_text(self, text: str, session_id: str = "default") -> Tuple[str, float]:
        """Process text input and return response with confidence score"""
        try:
            if not self.gemini_available and not self.groq_available:
                return "Text processing service is unavailable.", 0.0
            return self.generate_text_response(text, session_id)
        except Exception as e:
            logger.error(f"Error in text processing: {e}")
            return f"Sorry, there was an error processing your text: {str(e)}", 0.0

    def extract_image_features(self, image: Image.Image) -> Dict[str, Any]:
        """Extract features from image using available methods"""
        try:
            # Try Google Vision API first
            if self.google_vision_available and self.google_api_key:
                google_result = self._extract_with_google_vision(image)
                if not google_result.get("error", False):
                    return google_result

            # Fallback to BLIP model
            if self.blip_available:
                return self._extract_with_blip(image)
            else:
                return self._extract_basic_features(image)
        except Exception as e:
            logger.error(f"Error extracting image features: {e}")
            return {"error": True, "message": str(e)}





    def _extract_with_google_vision(self, image: Image.Image) -> Dict[str, Any]:
        """Extract features using Google Vision API for direct image analysis"""
        try:
            import base64
            import google.generativeai as genai

            # Convert image to base64
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG', quality=95)
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

            # Analysis prompt
            analysis_prompt = """Analyze this image comprehensively and provide:
1. DETAILED DESCRIPTION: Describe what you see in detail
2. OBJECTS: List all objects you can identify
3. COLORS: List all colors you can see with confidence
4. SCENE: Describe the overall scene/setting
5. SPECIFIC DETAILS: Any notable features, textures, or characteristics

Please be specific about colors (red, blue, green, etc.) and objects (cat, ball, car, person, etc.).
Format your response as a natural, conversational description that mentions colors and objects clearly."""

            # Configure and generate
            genai.configure(api_key=self.google_api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content([analysis_prompt, {"mime_type": "image/jpeg", "data": img_base64}])

            if response and response.text:
                google_description = response.text.strip()
                colors, objects = self._parse_google_vision_response(google_description)

                # Create attributes
                attributes = []
                for color, confidence in colors:
                    attributes.append({"label": color.title(), "confidence": confidence, "type": "color"})
                for obj, confidence in objects:
                    attributes.append({"label": obj.title(), "confidence": confidence, "type": "object"})

                attributes.sort(key=lambda x: x['confidence'], reverse=True)
                return {
                    "error": False,
                    "caption": google_description,
                    "attributes": attributes[:10],
                    "confidence": 0.95,
                    "used_hf_api": False,
                    "model": "Google-Gemini-1.5-Flash-Vision"
                }
            else:
                return {"error": True, "message": "Google Vision returned empty response"}
        except Exception as e:
            logger.warning(f"Google Vision API failed: {e}")
            return {"error": True, "message": str(e)}

    def _parse_google_vision_response(self, description: str) -> tuple:
        """Parse Google's vision response to extract colors and objects"""
        try:
            desc_lower = description.lower()
            colors, objects = [], []

            # Color and object keywords
            color_keywords = {'red': ['red', 'crimson'], 'blue': ['blue', 'navy'], 'green': ['green', 'emerald'], 'yellow': ['yellow', 'gold'], 'orange': ['orange'], 'purple': ['purple', 'violet'], 'pink': ['pink'], 'brown': ['brown', 'tan'], 'black': ['black', 'dark'], 'white': ['white'], 'gray': ['gray', 'grey']}
            object_keywords = {'cat': ['cat', 'kitten'], 'dog': ['dog', 'puppy'], 'ball': ['ball'], 'car': ['car', 'vehicle'], 'person': ['person', 'people', 'man', 'woman'], 'house': ['house', 'building'], 'tree': ['tree'], 'flower': ['flower'], 'grass': ['grass'], 'sky': ['sky'], 'bird': ['bird'], 'chair': ['chair'], 'table': ['table']}

            # Extract colors and objects
            for color, keywords in color_keywords.items():
                for keyword in keywords:
                    if keyword in desc_lower:
                        colors.append((color, min(0.8 + (desc_lower.count(keyword) * 0.05), 0.95)))
                        break
            for obj, keywords in object_keywords.items():
                for keyword in keywords:
                    if keyword in desc_lower:
                        objects.append((obj, min(0.85 + (desc_lower.count(keyword) * 0.03), 0.95)))
                        break

            return sorted(list(set(colors)), key=lambda x: x[1], reverse=True)[:8], sorted(list(set(objects)), key=lambda x: x[1], reverse=True)[:8]
        except Exception as e:
            logger.warning(f"Failed to parse vision response: {e}")
            return [], []









    def _optimize_caption_quality(self, caption: str) -> str:
        """Clean and optimize caption"""
        try:
            caption = caption.strip()
            for prefix in ["a detailed description of", "this image shows", "the image shows", "there is", "there are", "this is"]:
                if caption.lower().startswith(prefix):
                    caption = caption[len(prefix):].strip()
                    break
            if caption and not caption[0].isupper():
                caption = caption[0].upper() + caption[1:]
            words = caption.split()
            if len(words) > 12:
                caption = ' '.join(words[:12]) + "..."
            return caption
        except Exception:
            return caption

    def _needs_enhancement(self, caption: str) -> bool:
        """Check if caption needs enhancement"""
        try:
            words = caption.split()
            if len(words) < 4:
                return True
            caption_lower = caption.lower()
            for phrase in ["a picture of", "an image of", "a photo of", "a man", "a woman", "a person", "a group", "something", "anything", "nothing"]:
                if phrase in caption_lower and len(words) < 8:
                    return True
            return False
        except Exception:
            return False

    def _extract_meaningful_attributes(self, caption: str) -> List[Dict[str, Any]]:
        """Enhanced attribute extraction with perfect color and object detection"""
        try:
            attributes = []
            caption_lower = caption.lower()
            keywords = {
                "object": ["cat", "kitten", "dog", "puppy", "ball", "toy", "person", "people", "man", "woman", "child", "car", "truck", "bike", "tree", "flower", "house", "building", "chair", "table", "book", "phone"],
                "color": ["red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white", "gray", "grey", "crimson", "scarlet", "navy", "emerald", "golden", "silver"],
                "action": ["playing", "sitting", "standing", "walking", "running", "jumping", "eating", "drinking", "sleeping", "holding", "carrying", "wearing", "looking", "lying", "climbing"],
                "descriptor": ["small", "large", "big", "little", "cute", "beautiful", "bright", "colorful", "fluffy", "soft", "round", "playful", "happy", "young", "old"]
            }

            for attr_type, word_list in keywords.items():
                for word in word_list:
                    if word in caption_lower:
                        confidence = (0.98 if attr_type == "color" else 0.95) if f" {word} " in f" {caption_lower} " or caption_lower.startswith(word) or caption_lower.endswith(word) else (0.90 if attr_type == "color" else 0.85)
                        attributes.append({"label": word.title(), "confidence": confidence, "type": attr_type})

            seen = set()
            unique_attributes = []
            for attr in sorted(attributes, key=lambda x: x["confidence"], reverse=True):
                if attr["label"] not in seen:
                    seen.add(attr["label"])
                    unique_attributes.append(attr)
            return unique_attributes[:10]
        except Exception as e:
            logger.warning(f"Enhanced attribute extraction failed: {e}")
            return []

    def _enhance_caption_with_ai(self, caption: str) -> str:
        """Enhance basic captions using AI (Gemini primary, Groq fallback)"""
        try:
            words = caption.split()
            if len(words) > 12 or (len(words) >= 8 and any(color in caption.lower() for color in ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'gray']) and caption.count(',') >= 1):
                return caption

            prompt = f'Transform this basic image caption into a detailed description that captures ALL visual elements, especially colors and objects: "{caption}". Requirements: ALWAYS mention specific colors of objects, be specific about ALL objects/people/animals, include actions, keep natural but comprehensive, maximum 25 words. Enhanced detailed caption:'

            enhanced = None
            if self.gemini_available:
                try:
                    enhanced = self.gemini_client.generate_content(prompt, generation_config={'temperature': 0.7, 'max_output_tokens': 50, 'top_p': 0.9}).text.strip()
                except Exception as e:
                    logger.warning(f"Gemini enhancement failed: {e}")

            if not enhanced and self.groq_available:
                try:
                    enhanced = self.groq_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama3-8b-8192", temperature=0.7, max_tokens=50, top_p=0.9).choices[0].message.content.strip()
                except Exception as e:
                    logger.warning(f"Groq enhancement failed: {e}")

            if enhanced:
                enhanced = enhanced.replace('"', '').replace("Enhanced caption:", "").replace("Perfect caption:", "").strip()
                if enhanced.lower().startswith(("the image shows", "this image depicts", "the picture shows")):
                    enhanced = enhanced.split(" ", 3)[-1] if len(enhanced.split()) > 3 else enhanced
                if len(enhanced.split()) > len(words) and len(enhanced) < 200 and enhanced.lower() != caption.lower():
                    return enhanced
            return caption
        except Exception as e:
            logger.warning(f"AI caption enhancement failed: {e}")
            return caption

    def _clean_and_validate_blip_caption(self, raw_caption: str) -> str:
        """Clean and validate BLIP caption to ensure quality"""
        try:
            caption = raw_caption.strip()
            for phrase in ["the earth's surface", "the earth ' s surface", "earth's surface", "earth surface"]:
                if phrase in caption.lower():
                    return "An image with various objects and details"
            for phrase in ["a detailed description of", "this is a", "this is an", "the image shows", "there is a", "there is an"]:
                caption = caption.replace(phrase, "").strip()

            if caption and not caption[0].isupper():
                caption = caption[0].upper() + caption[1:]
            if caption and not caption.lower().startswith(('a ', 'an ', 'the ')):
                first_word = caption.split()[0] if caption.split() else ""
                caption = ("An " if first_word and first_word[0].lower() in 'aeiou' else "A ") + caption.lower()
                caption = caption[0].upper() + caption[1:]
            return caption if len(caption.split()) >= 3 else "An image with interesting content"
        except Exception:
            return "An image with various elements"

    def _extract_with_blip(self, image: Image.Image) -> Dict[str, Any]:
        """FALLBACK: Local BLIP model"""
        try:
            if image.size[0] > 1024 or image.size[1] > 1024:
                image = image.resize((512, 512), Image.LANCZOS)

            with torch.no_grad():
                best_caption = None
                best_score = 0

                # Try unconditional generation
                try:
                    inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
                    outputs = self.blip_model.generate(**inputs, max_length=50, num_beams=8, early_stopping=True, temperature=0.8, do_sample=True, top_p=0.9, repetition_penalty=1.2, length_penalty=1.0)
                    caption1 = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
                    if len(caption1.split()) >= 3 and "earth" not in caption1.lower():
                        best_caption, best_score = caption1, len(caption1.split())
                except Exception as e:
                    logger.warning(f"Unconditional generation failed: {e}")

                # Try with prompt
                try:
                    prompt_inputs = self.blip_processor(image, "this is", return_tensors="pt").to(self.device)
                    outputs = self.blip_model.generate(**prompt_inputs, max_length=50, num_beams=6, early_stopping=True, temperature=0.7, do_sample=True, top_p=0.9, repetition_penalty=1.1)
                    caption2 = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
                    if caption2.startswith("this is "):
                        caption2 = caption2[8:].strip()
                    if len(caption2.split()) > best_score and "earth" not in caption2.lower():
                        best_caption = caption2
                except Exception as e:
                    logger.warning(f"Prompted generation failed: {e}")

                # Use best caption or fallback
                raw_caption = best_caption if best_caption else self.blip_processor.decode(self.blip_model.generate(**self.blip_processor(image, return_tensors="pt").to(self.device), max_length=30, num_beams=5)[0], skip_special_tokens=True)

                # Clean and enhance caption
                caption = self._clean_and_validate_blip_caption(raw_caption)
                if self._needs_enhancement(caption):
                    enhanced_caption = self._enhance_caption_with_ai(caption)
                    if enhanced_caption and len(enhanced_caption.strip()) > len(caption.strip()):
                        caption = enhanced_caption

                return {"error": False, "caption": caption, "attributes": self._extract_meaningful_attributes(caption), "confidence": 0.85, "used_hf_api": False, "model": "Salesforce/blip-image-captioning-base"}
        except Exception as e:
            logger.error(f"BLIP local extraction failed: {e}")
            return {"error": True, "message": str(e)}

    def _extract_basic_features(self, image: Image.Image) -> Dict[str, Any]:
        """Extract basic features when no models available"""
        try:
            width, height = image.size
            rgb_image = image.convert('RGB') if image.mode != 'RGB' else image
            pixels = list(rgb_image.resize((50, 50)).getdata())
            color_counts = {'red': 0, 'green': 0, 'blue': 0, 'white': 0, 'black': 0}

            for r, g, b in pixels:
                if r > 200 and g < 100 and b < 100:
                    color_counts['red'] += 1
                elif r < 100 and g > 200 and b < 100:
                    color_counts['green'] += 1
                elif r < 100 and g < 100 and b > 200:
                    color_counts['blue'] += 1
                elif r > 200 and g > 200 and b > 200:
                    color_counts['white'] += 1
                elif r < 50 and g < 50 and b < 50:
                    color_counts['black'] += 1

            dominant_colors = [color for color, count in color_counts.items() if count > len(pixels) * 0.15]
            caption = f"An image with {' and '.join(dominant_colors[:2])} colors" if dominant_colors else f"A {width}x{height} image"
            attributes = [{"label": f"Resolution: {width}x{height}", "confidence": 1.0}] + [{"label": color, "confidence": 0.9} for color in dominant_colors]

            return {"error": False, "caption": caption, "attributes": attributes, "confidence": 0.7, "fallback": True}
        except Exception as e:
            logger.error(f"Basic feature extraction failed: {e}")
            return {"error": True, "message": str(e)}

    def process_image(self, image: Image.Image, session_id: str = "default") -> Dict[str, Any]:
        """Process image input and return analysis results"""
        try:
            if image.size[0] <= 0 or image.size[1] <= 0:
                return {"message": "Invalid image dimensions", "error": True}

            # Resize if too large
            width, height = image.size
            if width > 1024 or height > 1024:
                ratio = 1024 / max(width, height)
                image = image.resize((int(width * ratio), int(height * ratio)), Image.LANCZOS)

            # Extract features
            image_features = self.extract_image_features(image)
            if image_features.get("error", False):
                return {"message": f"Feature extraction failed: {image_features.get('message', 'Unknown error')}", "error": True}

            # Update image context with timestamp
            image_features["processed_at"] = time.time()
            self.image_context[session_id] = image_features

            return {"caption": image_features.get("caption", "No caption available"), "attributes": image_features.get("attributes", []), "message": "Image processed successfully", "error": False, "confidence": image_features.get("confidence", 0.9)}
        except Exception as e:
            logger.error(f"Error in image processing: {e}")
            return {"message": f"Error processing image: {str(e)}", "error": True, "caption": "Failed to analyze image", "confidence": 0.0}

    def process_multimodal(self, text: str, image: Image.Image, session_id: str = "default") -> Dict[str, Any]:
        """Process both text and image inputs together"""
        try:
            if not self.clip_available and not self.blip_available:
                return {"text_response": "Multimodal processing is unavailable.", "text_confidence": 0.0, "error": True}

            image_results = self.process_image(image, session_id)
            response, confidence = self.generate_text_response(text, session_id, with_image_context=True)

            similarity = 0.0
            if self.clip_available:
                try:
                    with torch.no_grad():
                        text_features = self.clip_model.get_text_features(**self.clip_processor.tokenizer(text, padding=True, return_tensors="pt").to(self.device))
                        image_features = self.clip_model.get_image_features(**self.clip_processor(images=image, return_tensors="pt").to(self.device))
                        similarity = F.cosine_similarity(text_features, image_features).item()
                except Exception as e:
                    logger.error(f"Error calculating similarity: {e}")

            multimodal_response = f"{response}\n\nI notice your message relates to the image you've shared." if similarity > 0.25 else response
            return {"text_response": multimodal_response, "text_confidence": confidence, "image_caption": image_results.get("caption", ""), "image_attributes": image_results.get("attributes", []), "text_image_similarity": similarity, "message": "Multimodal processing complete"}
        except Exception as e:
            logger.error(f"Error in multimodal processing: {e}")
            return {"text_response": f"I encountered an error processing your input: {str(e)}", "text_confidence": 0.0, "error": True}

    def clear_session(self, session_id: str) -> bool:
        """Clear conversation history and image context for a session"""
        try:
            self.conversation_history.pop(session_id, None)
            self.image_context.pop(session_id, None)
            return True
        except Exception as e:
            logger.error(f"Error clearing session {session_id}: {e}")
            return False

try:
    model_service = ModelService()
except Exception as e:
    logger.critical(f"Failed to initialize ModelService: {e}")
    class FallbackModelService:
        def process_text(self, _text: str, _session_id: str = "default") -> Tuple[str, float]:
            return "The model service is currently unavailable. Please try again later.", 0.0
        def process_image(self, _image: Image.Image, _session_id: str = "default") -> Dict[str, Any]:
            return {"message": "Image processing is currently unavailable", "error": True}
        def process_multimodal(self, _text: str, _image: Image.Image, _session_id: str = "default") -> Dict[str, Any]:
            return {"text_response": "Multimodal processing is currently unavailable", "text_confidence": 0.0, "error": True}
        def clear_session(self, _session_id: str) -> bool:
            return True
    model_service = FallbackModelService()
