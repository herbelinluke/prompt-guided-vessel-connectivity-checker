"""
Visualization - Display and save analysis results with overlays.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import io
import base64


class Visualizer:
    """
    Visualization tools for vessel segmentation analysis.
    Creates overlays, comparisons, and annotated images.
    """
    
    # Color schemes
    COLORS = {
        "vessel_overlay": (220, 50, 50),      # Red for vessels
        "vessel_overlay_alt": (50, 200, 50),   # Green alternative
        "highlight": (255, 255, 0),            # Yellow for highlights
        "warning": (255, 165, 0),              # Orange for warnings
        "error": (255, 0, 0),                  # Red for errors
        "success": (0, 255, 0),                # Green for success
        "info": (100, 149, 237),               # Cornflower blue for info
    }
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size (width, height) in inches
        """
        self.figsize = figsize
        self._check_matplotlib()
    
    def _check_matplotlib(self):
        """Check if matplotlib is available."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            self.plt = plt
            self.has_matplotlib = True
        except ImportError:
            self.has_matplotlib = False
    
    def overlay_segmentation(
        self,
        image: np.ndarray,
        segmentation: np.ndarray,
        alpha: float = 0.4,
        color: Tuple[int, int, int] = (220, 50, 50)
    ) -> np.ndarray:
        """
        Overlay vessel segmentation on original image.
        
        Args:
            image: Original image (RGB, uint8)
            segmentation: Binary segmentation (grayscale, vessels=white)
            alpha: Transparency of overlay (0-1)
            color: RGB color for segmentation overlay
            
        Returns:
            Composite image with overlay
        """
        # Ensure correct dtypes
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Create colored overlay
        overlay = image.copy()
        vessel_pixels = segmentation > 127
        
        # Apply color to vessel regions
        for i, c in enumerate(color):
            overlay[:, :, i] = np.where(
                vessel_pixels,
                np.clip(image[:, :, i] * (1 - alpha) + c * alpha, 0, 255),
                image[:, :, i]
            )
        
        return overlay.astype(np.uint8)
    
    # Alias for backwards compatibility
    def overlay_mask(self, image, mask, alpha=0.4, color=(220, 50, 50)):
        """Deprecated: Use overlay_segmentation instead."""
        return self.overlay_segmentation(image, mask, alpha, color)
    
    def create_comparison(
        self,
        image: np.ndarray,
        segmentation: np.ndarray,
        overlay: Optional[np.ndarray] = None,
        titles: List[str] = None
    ) -> np.ndarray:
        """
        Create a side-by-side comparison image.
        
        Args:
            image: Original image
            segmentation: Vessel segmentation
            overlay: Optional pre-computed overlay
            titles: Optional titles for each panel
            
        Returns:
            Combined comparison image
        """
        if overlay is None:
            overlay = self.overlay_segmentation(image, segmentation)
        
        # Ensure all images are RGB and same size
        h, w = image.shape[:2]
        
        # Convert segmentation to RGB for display
        if len(segmentation.shape) == 2:
            segmentation_rgb = np.stack([segmentation] * 3, axis=-1)
        else:
            segmentation_rgb = segmentation
        
        # Create combined image with spacing
        gap = 10
        combined_width = w * 3 + gap * 2
        combined = np.ones((h, combined_width, 3), dtype=np.uint8) * 40  # Dark gray background
        
        # Place images
        combined[:, :w] = image
        combined[:, w + gap:w * 2 + gap] = segmentation_rgb
        combined[:, w * 2 + gap * 2:] = overlay
        
        return combined
    
    def create_annotated_overlay(
        self,
        image: np.ndarray,
        segmentation: np.ndarray,
        annotations: List[Dict[str, Any]],
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Create overlay with text annotations.
        
        Args:
            image: Original image
            segmentation: Vessel segmentation
            annotations: List of annotation dicts with 'text', 'position', optional 'color'
            alpha: Segmentation transparency
            
        Returns:
            Annotated image
        """
        overlay = self.overlay_segmentation(image, segmentation, alpha=alpha)
        pil_image = Image.fromarray(overlay)
        draw = ImageDraw.Draw(pil_image)
        
        # Try to use a nicer font, fall back to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        for ann in annotations:
            text = ann.get('text', '')
            position = ann.get('position', (10, 10))
            color = ann.get('color', self.COLORS['info'])
            
            # Draw text with background
            bbox = draw.textbbox(position, text, font=font)
            draw.rectangle(
                [bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2],
                fill=(0, 0, 0, 180)
            )
            draw.text(position, text, fill=color, font=font)
        
        return np.array(pil_image)
    
    def highlight_regions(
        self,
        image: np.ndarray,
        regions: List[Tuple[int, int, int, int]],
        color: Tuple[int, int, int] = (255, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Highlight rectangular regions on image.
        
        Args:
            image: Input image
            regions: List of (x1, y1, x2, y2) bounding boxes
            color: Highlight color
            thickness: Line thickness
            
        Returns:
            Image with highlighted regions
        """
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        
        for x1, y1, x2, y2 in regions:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        
        return np.array(pil_image)
    
    def create_report_figure(
        self,
        image: np.ndarray,
        segmentation: np.ndarray,
        result: Dict[str, Any],
        title: str = "Vessel Connectivity Analysis"
    ) -> Optional[Any]:
        """
        Create a matplotlib figure with analysis results.
        
        Args:
            image: Original image
            segmentation: Vessel segmentation
            result: Analysis result dictionary
            title: Figure title
            
        Returns:
            Matplotlib figure or None if matplotlib unavailable
        """
        if not self.has_matplotlib:
            print("Warning: matplotlib not available for figure creation")
            return None
        
        fig, axes = self.plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Segmentation
        axes[0, 1].imshow(segmentation, cmap='gray')
        axes[0, 1].set_title("Vessel Segmentation")
        axes[0, 1].axis('off')
        
        # Overlay
        overlay = self.overlay_segmentation(image, segmentation)
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title("Overlay")
        axes[1, 0].axis('off')
        
        # Results text panel
        axes[1, 1].axis('off')
        result_text = self._format_result_text(result)
        axes[1, 1].text(
            0.1, 0.9, result_text,
            transform=axes[1, 1].transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        axes[1, 1].set_title("Analysis Results")
        
        self.plt.tight_layout()
        return fig
    
    def _format_result_text(self, result: Dict[str, Any]) -> str:
        """Format analysis result as display text."""
        lines = []
        
        if 'continuous' in result:
            status = "✓ Continuous" if result['continuous'] else "✗ Discontinuous"
            lines.append(f"Status: {status}")
        
        if 'confidence' in result:
            lines.append(f"Confidence: {result['confidence']:.2f}")
        
        if 'quality_score' in result and result['quality_score'] is not None:
            lines.append(f"Quality Score: {result['quality_score']:.2f}")
        
        if 'broken_segments' in result and result['broken_segments']:
            lines.append(f"\nBroken Segments ({len(result['broken_segments'])}):")
            for seg in result['broken_segments'][:5]:  # Limit to 5
                lines.append(f"  • {seg[:40]}...")
        
        if 'bifurcation_quality' in result and result['bifurcation_quality']:
            lines.append(f"\nBifurcation Quality: {result['bifurcation_quality']}")
        
        if 'anatomically_plausible' in result and result['anatomically_plausible'] is not None:
            anat = "Yes" if result['anatomically_plausible'] else "No"
            lines.append(f"Anatomically Plausible: {anat}")
        
        return '\n'.join(lines)
    
    def save_figure(
        self,
        fig: Any,
        output_path: Union[str, Path],
        dpi: int = 150
    ) -> None:
        """
        Save a matplotlib figure.
        
        Args:
            fig: Matplotlib figure
            output_path: Output file path
            dpi: Resolution
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        self.plt.close(fig)
    
    def save_image(
        self,
        image: np.ndarray,
        output_path: Union[str, Path]
    ) -> None:
        """
        Save a numpy image array.
        
        Args:
            image: Image array
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        pil_image.save(output_path)
    
    def display_results(
        self,
        image: np.ndarray,
        segmentation: np.ndarray,
        result: Dict[str, Any],
        save_path: Optional[Union[str, Path]] = None
    ) -> Optional[Any]:
        """
        Display or save analysis results.
        
        Args:
            image: Original image
            segmentation: Vessel segmentation
            result: Analysis result dictionary
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure if available
        """
        fig = self.create_report_figure(image, segmentation, result)
        
        if fig is not None and save_path:
            self.save_figure(fig, save_path)
        
        return fig
    
    def create_simple_overlay_image(
        self,
        image_path: Union[str, Path],
        segmentation_path: Union[str, Path],
        output_path: Union[str, Path],
        alpha: float = 0.4
    ) -> None:
        """
        Simple function to create and save an overlay image.
        
        Args:
            image_path: Path to original image
            segmentation_path: Path to segmentation
            output_path: Output path for overlay
            alpha: Transparency
        """
        image = np.array(Image.open(image_path).convert('RGB'))
        segmentation = np.array(Image.open(segmentation_path).convert('L'))
        
        overlay = self.overlay_segmentation(image, segmentation, alpha=alpha)
        self.save_image(overlay, output_path)


def create_overlay(
    image: Union[np.ndarray, str, Path],
    segmentation: Union[np.ndarray, str, Path],
    alpha: float = 0.4,
    color: Tuple[int, int, int] = (220, 50, 50)
) -> np.ndarray:
    """
    Convenience function to create a segmentation overlay.
    
    Args:
        image: Image array or path
        segmentation: Segmentation array or path
        alpha: Transparency (0-1)
        color: Overlay color
        
    Returns:
        Overlay image
    """
    if isinstance(image, (str, Path)):
        image = np.array(Image.open(image).convert('RGB'))
    if isinstance(segmentation, (str, Path)):
        segmentation = np.array(Image.open(segmentation).convert('L'))
    
    viz = Visualizer()
    return viz.overlay_segmentation(image, segmentation, alpha, color)


def save_comparison(
    image: Union[np.ndarray, str, Path],
    segmentation: Union[np.ndarray, str, Path],
    output_path: Union[str, Path]
) -> None:
    """
    Convenience function to save a comparison image.
    
    Args:
        image: Image array or path
        segmentation: Segmentation array or path
        output_path: Output file path
    """
    if isinstance(image, (str, Path)):
        image = np.array(Image.open(image).convert('RGB'))
    if isinstance(segmentation, (str, Path)):
        segmentation = np.array(Image.open(segmentation).convert('L'))
    
    viz = Visualizer()
    comparison = viz.create_comparison(image, segmentation)
    viz.save_image(comparison, output_path)
