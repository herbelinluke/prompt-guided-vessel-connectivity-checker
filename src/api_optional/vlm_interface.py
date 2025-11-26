"""
VLM Interface - Wrapper for Vision-Language Model APIs.
Supports OpenAI GPT-4o/GPT-4o-mini and local LLaVA-Med.
"""

import os
import base64
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
import numpy as np
from PIL import Image
import io
import json


class VLMBase(ABC):
    """Abstract base class for VLM interfaces."""
    
    @abstractmethod
    def ask(
        self, 
        image: Union[np.ndarray, str, Path],
        prompt: str,
        segmentation: Optional[Union[np.ndarray, str, Path]] = None,
        **kwargs
    ) -> str:
        """
        Send an image and prompt to the VLM and get a response.
        
        Args:
            image: Image as numpy array or path
            prompt: Text prompt for the VLM
            segmentation: Optional vessel segmentation
            **kwargs: Additional model-specific parameters
            
        Returns:
            Text response from the VLM
        """
        pass


class OpenAIVLM(VLMBase):
    """
    OpenAI GPT-4o/GPT-4o-mini Vision interface.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_tokens: int = 1024,
        temperature: float = 0.3
    ):
        """
        Initialize OpenAI VLM interface.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model to use (gpt-4o, gpt-4o-mini)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Import here to make dependency optional
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )
    
    def _image_to_base64(self, image: Union[np.ndarray, str, Path]) -> str:
        """Convert image to base64 data URL."""
        if isinstance(image, (str, Path)):
            with open(image, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            # Determine format from extension
            ext = str(image).lower().split('.')[-1]
            mime_type = "image/png" if ext == "png" else "image/jpeg"
        else:
            # numpy array
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            mime_type = "image/png"
        
        return f"data:{mime_type};base64,{image_data}"
    
    def _create_composite(
        self, 
        image: np.ndarray, 
        segmentation: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """Create composite image with segmentation overlay."""
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        
        # Create red overlay for vessels
        overlay = image.copy()
        vessel_pixels = segmentation > 127
        overlay[vessel_pixels] = [255, 100, 100]  # Light red tint
        
        composite = (image * (1 - alpha) + overlay * alpha).astype(np.uint8)
        return composite
    
    def ask(
        self, 
        image: Union[np.ndarray, str, Path],
        prompt: str,
        segmentation: Optional[Union[np.ndarray, str, Path]] = None,
        include_composite: bool = True,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Send image(s) and prompt to GPT-4o Vision.
        
        Args:
            image: Original image
            prompt: Analysis prompt
            segmentation: Optional vessel segmentation
            include_composite: Whether to include composite overlay image
            system_prompt: Optional system prompt for context
            **kwargs: Additional parameters (max_tokens, temperature)
            
        Returns:
            VLM text response
        """
        # Load images if paths provided
        if isinstance(image, (str, Path)):
            image = np.array(Image.open(image).convert('RGB'))
        if segmentation is not None and isinstance(segmentation, (str, Path)):
            segmentation = np.array(Image.open(segmentation).convert('L'))
        
        # Build message content
        content = []
        
        # Add original image
        content.append({
            "type": "image_url",
            "image_url": {
                "url": self._image_to_base64(image),
                "detail": "high"
            }
        })
        
        # Add segmentation if provided
        if segmentation is not None:
            # Convert segmentation to RGB for visibility
            seg_rgb = np.stack([segmentation] * 3, axis=-1)
            content.append({
                "type": "image_url", 
                "image_url": {
                    "url": self._image_to_base64(seg_rgb),
                    "detail": "high"
                }
            })
            
            # Add composite overlay
            if include_composite:
                composite = self._create_composite(image, segmentation)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": self._image_to_base64(composite),
                        "detail": "high"
                    }
                })
        
        # Add text prompt
        content.append({
            "type": "text",
            "text": prompt
        })
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        
        # Make API call
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature)
        )
        
        return response.choices[0].message.content


class LLaVAMedVLM(VLMBase):
    """
    LLaVA-Med local model interface using Hugging Face transformers.
    Requires GPU for reasonable performance.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/llava-med-v1.5-mistral-7b",
        device: str = "auto",
        torch_dtype: str = "float16"
    ):
        """
        Initialize LLaVA-Med interface.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('auto', 'cuda', 'cpu')
            torch_dtype: Model precision
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        
        try:
            import torch
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            self.torch = torch
            self.torch_dtype = getattr(torch, torch_dtype)
        except ImportError:
            raise ImportError(
                "transformers and torch required for LLaVA-Med. "
                "Install with: pip install transformers torch"
            )
    
    def _load_model(self):
        """Lazy load the model."""
        if self.model is not None:
            return
        
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        
        print(f"Loading LLaVA-Med model: {self.model_name}...")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device
        )
        print("Model loaded successfully.")
    
    def ask(
        self, 
        image: Union[np.ndarray, str, Path],
        prompt: str,
        segmentation: Optional[Union[np.ndarray, str, Path]] = None,
        max_new_tokens: int = 512,
        **kwargs
    ) -> str:
        """
        Send image and prompt to LLaVA-Med.
        
        Args:
            image: Input image
            prompt: Analysis prompt
            segmentation: Optional vessel segmentation (will be composited)
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Model response text
        """
        self._load_model()
        
        # Load and prepare image
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert('RGB')
        else:
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)
        
        # If segmentation provided, create composite
        if segmentation is not None:
            if isinstance(segmentation, (str, Path)):
                segmentation = np.array(Image.open(segmentation).convert('L'))
            image_array = np.array(pil_image)
            composite = self._create_composite(image_array, segmentation)
            pil_image = Image.fromarray(composite)
        
        # Format prompt for LLaVA
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=text_prompt,
            images=pil_image,
            return_tensors="pt"
        ).to(self.model.device)
        
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3
        )
        
        response = self.processor.decode(output[0], skip_special_tokens=True)
        
        # Extract just the assistant response
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        
        return response
    
    def _create_composite(
        self, 
        image: np.ndarray, 
        segmentation: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """Create composite image with segmentation overlay."""
        overlay = image.copy()
        vessel_pixels = segmentation > 127
        overlay[vessel_pixels] = [255, 100, 100]
        composite = (image * (1 - alpha) + overlay * alpha).astype(np.uint8)
        return composite


class VLMInterface:
    """
    Unified interface for different VLM backends.
    """
    
    def __init__(
        self,
        backend: str = "openai",
        **kwargs
    ):
        """
        Initialize VLM interface with specified backend.
        
        Args:
            backend: 'openai' or 'llava-med'
            **kwargs: Backend-specific configuration
        """
        self.backend = backend
        
        if backend == "openai":
            self.vlm = OpenAIVLM(**kwargs)
        elif backend == "llava-med":
            self.vlm = LLaVAMedVLM(**kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def ask(
        self,
        image: Union[np.ndarray, str, Path],
        prompt: str,
        segmentation: Optional[Union[np.ndarray, str, Path]] = None,
        **kwargs
    ) -> str:
        """
        Ask the VLM about an image.
        
        Args:
            image: Input image
            prompt: Analysis prompt
            segmentation: Optional vessel segmentation
            **kwargs: Additional parameters
            
        Returns:
            VLM response text
        """
        return self.vlm.ask(image, prompt, segmentation, **kwargs)
    
    def ask_vlm(
        self,
        image: Union[np.ndarray, str, Path],
        segmentation: Optional[Union[np.ndarray, str, Path]],
        prompt: str,
        **kwargs
    ) -> str:
        """
        Alias for ask() with different argument order.
        Matches the signature specified in requirements.
        
        Args:
            image: Input image
            segmentation: Vessel segmentation (can be None)
            prompt: Analysis prompt
            
        Returns:
            VLM response text
        """
        return self.ask(image, prompt, segmentation, **kwargs)
