"""
Segmentation Loader - Load and preprocess medical images and segmentation masks.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Union
import numpy as np
from PIL import Image
import io
import base64


class SegmentationLoader:
    """
    Loads and preprocesses medical images and their corresponding segmentation masks.
    Handles normalization, resizing, and format conversion for VLM input.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        """
        Initialize the loader with target dimensions.
        
        Args:
            target_size: Target (width, height) for resizing images
        """
        self.target_size = target_size
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load an image from disk.
        
        Args:
            image_path: Path to the image file (PNG, JPG, TIFF)
            
        Returns:
            numpy array of the image (RGB format)
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return np.array(image)
    
    def load_mask(self, mask_path: Union[str, Path]) -> np.ndarray:
        """
        Load a binary segmentation mask from disk.
        
        Args:
            mask_path: Path to the mask file
            
        Returns:
            numpy array of the binary mask (0 and 255 values)
        """
        mask_path = Path(mask_path)
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        mask = Image.open(mask_path)
        
        # Convert to grayscale if necessary
        if mask.mode != 'L':
            mask = mask.convert('L')
        
        mask_array = np.array(mask)
        
        # Binarize the mask
        threshold = 127
        mask_array = (mask_array > threshold).astype(np.uint8) * 255
        
        return mask_array
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image array
            
        Returns:
            Resized image array
        """
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize(self.target_size, Image.Resampling.LANCZOS)
        return np.array(pil_image)
    
    def resize_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Resize mask to target size using nearest neighbor (preserves binary values).
        
        Args:
            mask: Input mask array
            
        Returns:
            Resized mask array
        """
        pil_mask = Image.fromarray(mask)
        pil_mask = pil_mask.resize(self.target_size, Image.Resampling.NEAREST)
        return np.array(pil_mask)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to 0-1 range.
        
        Args:
            image: Input image array (0-255)
            
        Returns:
            Normalized image array (0-1)
        """
        return image.astype(np.float32) / 255.0
    
    def load_pair(
        self, 
        image_path: Union[str, Path], 
        mask_path: Union[str, Path],
        resize: bool = True,
        normalize: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load an image-mask pair and optionally preprocess them.
        
        Args:
            image_path: Path to the image file
            mask_path: Path to the mask file
            resize: Whether to resize to target size
            normalize: Whether to normalize image values
            
        Returns:
            Tuple of (image, mask) numpy arrays
        """
        image = self.load_image(image_path)
        mask = self.load_mask(mask_path)
        
        if resize:
            image = self.resize_image(image)
            mask = self.resize_mask(mask)
        
        if normalize:
            image = self.normalize_image(image)
        
        return image, mask
    
    def create_composite(
        self, 
        image: np.ndarray, 
        mask: np.ndarray, 
        alpha: float = 0.5,
        mask_color: Tuple[int, int, int] = (255, 0, 0)
    ) -> np.ndarray:
        """
        Create a composite image with mask overlay for VLM input.
        
        Args:
            image: Original image (RGB)
            mask: Binary mask
            alpha: Transparency of the overlay
            mask_color: Color for the mask overlay (RGB)
            
        Returns:
            Composite image with mask overlay
        """
        # Ensure image is in correct format
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        
        # Create colored mask
        colored_mask = np.zeros_like(image)
        mask_binary = mask > 127
        colored_mask[mask_binary] = mask_color
        
        # Blend images
        composite = image.copy().astype(np.float32)
        composite[mask_binary] = (
            composite[mask_binary] * (1 - alpha) + 
            colored_mask[mask_binary].astype(np.float32) * alpha
        )
        
        return composite.astype(np.uint8)
    
    def to_base64(self, image: np.ndarray, format: str = "PNG") -> str:
        """
        Convert image array to base64 string for API transmission.
        
        Args:
            image: Image numpy array
            format: Image format (PNG, JPEG)
            
        Returns:
            Base64 encoded string
        """
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def create_side_by_side(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        gap: int = 10
    ) -> np.ndarray:
        """
        Create a side-by-side view of image and mask for VLM analysis.
        
        Args:
            image: Original image (RGB)
            mask: Binary mask
            gap: Pixel gap between images
            
        Returns:
            Combined side-by-side image
        """
        # Ensure same dimensions
        h, w = image.shape[:2]
        
        # Convert mask to RGB for display
        if len(mask.shape) == 2:
            mask_rgb = np.stack([mask] * 3, axis=-1)
        else:
            mask_rgb = mask
        
        # Create combined image with gap
        combined = np.ones((h, w * 2 + gap, 3), dtype=np.uint8) * 255
        combined[:, :w] = image
        combined[:, w + gap:] = mask_rgb
        
        return combined


def generate_synthetic_mask(image: np.ndarray, method: str = "otsu") -> np.ndarray:
    """
    Generate a synthetic segmentation mask using basic thresholding.
    Useful for testing when no ground truth mask is available.
    
    Args:
        image: Input image (RGB or grayscale)
        method: Thresholding method ('otsu', 'adaptive', 'simple')
        
    Returns:
        Binary mask numpy array
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2).astype(np.uint8)
    else:
        gray = image
    
    if method == "otsu":
        # Simple Otsu's method implementation
        histogram, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
        total = gray.size
        
        sum_total = np.sum(np.arange(256) * histogram)
        sum_bg = 0
        weight_bg = 0
        
        max_variance = 0
        threshold = 0
        
        for t in range(256):
            weight_bg += histogram[t]
            if weight_bg == 0:
                continue
            
            weight_fg = total - weight_bg
            if weight_fg == 0:
                break
            
            sum_bg += t * histogram[t]
            mean_bg = sum_bg / weight_bg
            mean_fg = (sum_total - sum_bg) / weight_fg
            
            variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
            
            if variance > max_variance:
                max_variance = variance
                threshold = t
        
        mask = (gray < threshold).astype(np.uint8) * 255
        
    elif method == "adaptive":
        # Simple adaptive thresholding
        from scipy.ndimage import uniform_filter
        block_size = 51
        offset = 10
        local_mean = uniform_filter(gray.astype(float), size=block_size)
        mask = (gray < local_mean - offset).astype(np.uint8) * 255
        
    else:  # simple
        threshold = 127
        mask = (gray < threshold).astype(np.uint8) * 255
    
    return mask

