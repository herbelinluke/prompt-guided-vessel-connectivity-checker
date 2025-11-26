"""
Segmentation Loader - Load and preprocess medical images and vessel segmentations.
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
    Loads and preprocesses medical images and their corresponding vessel segmentations.
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
    
    def load_segmentation(self, segmentation_path: Union[str, Path]) -> np.ndarray:
        """
        Load a binary vessel segmentation from disk.
        
        Args:
            segmentation_path: Path to the segmentation file
            
        Returns:
            numpy array of the binary segmentation (0 and 255 values)
        """
        segmentation_path = Path(segmentation_path)
        if not segmentation_path.exists():
            raise FileNotFoundError(f"Segmentation not found: {segmentation_path}")
        
        segmentation = Image.open(segmentation_path)
        
        # Convert to grayscale if necessary
        if segmentation.mode != 'L':
            segmentation = segmentation.convert('L')
        
        seg_array = np.array(segmentation)
        
        # Binarize the segmentation
        threshold = 127
        seg_array = (seg_array > threshold).astype(np.uint8) * 255
        
        return seg_array
    
    # Alias for backwards compatibility
    def load_mask(self, mask_path: Union[str, Path]) -> np.ndarray:
        """Deprecated: Use load_segmentation instead."""
        return self.load_segmentation(mask_path)
    
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
    
    def resize_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Resize segmentation to target size using nearest neighbor (preserves binary values).
        
        Args:
            segmentation: Input segmentation array
            
        Returns:
            Resized segmentation array
        """
        pil_seg = Image.fromarray(segmentation)
        pil_seg = pil_seg.resize(self.target_size, Image.Resampling.NEAREST)
        return np.array(pil_seg)
    
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
        segmentation_path: Union[str, Path],
        resize: bool = True,
        normalize: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load an image-segmentation pair and optionally preprocess them.
        
        Args:
            image_path: Path to the image file
            segmentation_path: Path to the segmentation file
            resize: Whether to resize to target size
            normalize: Whether to normalize image values
            
        Returns:
            Tuple of (image, segmentation) numpy arrays
        """
        image = self.load_image(image_path)
        segmentation = self.load_segmentation(segmentation_path)
        
        if resize:
            image = self.resize_image(image)
            segmentation = self.resize_segmentation(segmentation)
        
        if normalize:
            image = self.normalize_image(image)
        
        return image, segmentation
    
    def create_composite(
        self, 
        image: np.ndarray, 
        segmentation: np.ndarray, 
        alpha: float = 0.5,
        overlay_color: Tuple[int, int, int] = (255, 0, 0)
    ) -> np.ndarray:
        """
        Create a composite image with segmentation overlay for VLM input.
        
        Args:
            image: Original image (RGB)
            segmentation: Binary segmentation
            alpha: Transparency of the overlay
            overlay_color: Color for the segmentation overlay (RGB)
            
        Returns:
            Composite image with segmentation overlay
        """
        # Ensure image is in correct format
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        
        # Create colored overlay for vessels
        colored_overlay = np.zeros_like(image)
        vessel_pixels = segmentation > 127
        colored_overlay[vessel_pixels] = overlay_color
        
        # Blend images
        composite = image.copy().astype(np.float32)
        composite[vessel_pixels] = (
            composite[vessel_pixels] * (1 - alpha) + 
            colored_overlay[vessel_pixels].astype(np.float32) * alpha
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
        segmentation: np.ndarray,
        gap: int = 10
    ) -> np.ndarray:
        """
        Create a side-by-side view of image and segmentation for VLM analysis.
        
        Args:
            image: Original image (RGB)
            segmentation: Binary segmentation
            gap: Pixel gap between images
            
        Returns:
            Combined side-by-side image
        """
        # Ensure same dimensions
        h, w = image.shape[:2]
        
        # Convert segmentation to RGB for display
        if len(segmentation.shape) == 2:
            seg_rgb = np.stack([segmentation] * 3, axis=-1)
        else:
            seg_rgb = segmentation
        
        # Create combined image with gap
        combined = np.ones((h, w * 2 + gap, 3), dtype=np.uint8) * 255
        combined[:, :w] = image
        combined[:, w + gap:] = seg_rgb
        
        return combined


def generate_synthetic_segmentation(image: np.ndarray, method: str = "otsu") -> np.ndarray:
    """
    Generate a synthetic vessel segmentation using basic thresholding.
    Useful for testing when no ground truth segmentation is available.
    
    Args:
        image: Input image (RGB or grayscale)
        method: Thresholding method ('otsu', 'adaptive', 'simple')
        
    Returns:
        Binary segmentation numpy array
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
        
        segmentation = (gray < threshold).astype(np.uint8) * 255
        
    elif method == "adaptive":
        # Simple adaptive thresholding
        from scipy.ndimage import uniform_filter
        block_size = 51
        offset = 10
        local_mean = uniform_filter(gray.astype(float), size=block_size)
        segmentation = (gray < local_mean - offset).astype(np.uint8) * 255
        
    else:  # simple
        threshold = 127
        segmentation = (gray < threshold).astype(np.uint8) * 255
    
    return segmentation
