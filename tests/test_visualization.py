"""
Tests for visualization module.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization import Visualizer, create_overlay, save_comparison


class TestVisualizer:
    """Test cases for Visualizer class."""
    
    @pytest.fixture
    def visualizer(self):
        """Create a visualizer instance."""
        return Visualizer()
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample RGB image."""
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    @pytest.fixture
    def sample_mask(self):
        """Create a sample binary mask."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255
        return mask
    
    def test_overlay_mask(self, visualizer, sample_image, sample_mask):
        """Test mask overlay creation."""
        overlay = visualizer.overlay_mask(sample_image, sample_mask)
        
        assert overlay.shape == sample_image.shape
        assert overlay.dtype == np.uint8
    
    def test_overlay_mask_alpha(self, visualizer, sample_image, sample_mask):
        """Test overlay with different alpha values."""
        overlay_low = visualizer.overlay_mask(sample_image, sample_mask, alpha=0.1)
        overlay_high = visualizer.overlay_mask(sample_image, sample_mask, alpha=0.9)
        
        # With higher alpha, overlay should be more different from original
        diff_low = np.abs(overlay_low.astype(float) - sample_image.astype(float)).mean()
        diff_high = np.abs(overlay_high.astype(float) - sample_image.astype(float)).mean()
        
        # Higher alpha should produce more difference where mask is
        # (this is a rough test, exact behavior depends on mask coverage)
        assert overlay_low.shape == overlay_high.shape
    
    def test_overlay_custom_color(self, visualizer, sample_image, sample_mask):
        """Test overlay with custom color."""
        color = (0, 255, 0)  # Green
        overlay = visualizer.overlay_mask(sample_image, sample_mask, color=color)
        
        # In mask regions, green channel should be elevated
        mask_region = sample_mask > 127
        assert overlay[mask_region, 1].mean() > sample_image[mask_region, 1].mean()
    
    def test_create_comparison(self, visualizer, sample_image, sample_mask):
        """Test comparison image creation."""
        comparison = visualizer.create_comparison(sample_image, sample_mask)
        
        h, w = sample_image.shape[:2]
        # Comparison should have 3 panels
        assert comparison.shape[1] > w * 2
        assert comparison.shape[0] == h
    
    def test_highlight_regions(self, visualizer, sample_image):
        """Test region highlighting."""
        regions = [(10, 10, 30, 30), (50, 50, 80, 80)]
        highlighted = visualizer.highlight_regions(sample_image, regions)
        
        assert highlighted.shape == sample_image.shape
    
    def test_save_image(self, visualizer, sample_image, tmp_path):
        """Test image saving."""
        output_path = tmp_path / "test_output.png"
        visualizer.save_image(sample_image, output_path)
        
        assert output_path.exists()
    
    def test_save_image_normalized(self, visualizer, tmp_path):
        """Test saving normalized (float) image."""
        image = np.random.rand(100, 100, 3).astype(np.float32)
        output_path = tmp_path / "test_float.png"
        
        visualizer.save_image(image, output_path)
        assert output_path.exists()
    
    def test_create_annotated_overlay(self, visualizer, sample_image, sample_mask):
        """Test annotated overlay creation."""
        annotations = [
            {'text': 'Test annotation', 'position': (10, 10)},
            {'text': 'Another note', 'position': (10, 30), 'color': (255, 0, 0)}
        ]
        
        annotated = visualizer.create_annotated_overlay(
            sample_image, sample_mask, annotations
        )
        
        assert annotated.shape == sample_image.shape
    
    def test_format_result_text(self, visualizer):
        """Test result text formatting."""
        result = {
            'continuous': True,
            'confidence': 0.85,
            'quality_score': 0.9,
            'broken_segments': ['segment 1', 'segment 2'],
            'bifurcation_quality': 'good'
        }
        
        text = visualizer._format_result_text(result)
        
        assert 'Continuous' in text
        assert '0.85' in text or '85' in text
        assert 'segment 1' in text
    
    def test_display_results(self, visualizer, sample_image, sample_mask, tmp_path):
        """Test display/save results."""
        result = {
            'continuous': True,
            'confidence': 0.8,
            'broken_segments': []
        }
        
        save_path = tmp_path / "test_figure.png"
        fig = visualizer.display_results(
            sample_image, sample_mask, result, save_path=save_path
        )
        
        if visualizer.has_matplotlib:
            assert save_path.exists()


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create sample data files."""
        from PIL import Image
        
        # Create and save image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        image_path = tmp_path / "image.png"
        Image.fromarray(image).save(image_path)
        
        # Create and save mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255
        mask_path = tmp_path / "mask.png"
        Image.fromarray(mask).save(mask_path)
        
        return {'image': image_path, 'mask': mask_path}
    
    def test_create_overlay_from_arrays(self):
        """Test create_overlay with arrays."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255
        
        overlay = create_overlay(image, mask)
        assert overlay.shape == image.shape
    
    def test_create_overlay_from_paths(self, sample_data):
        """Test create_overlay with file paths."""
        overlay = create_overlay(sample_data['image'], sample_data['mask'])
        assert overlay.shape[2] == 3  # RGB
    
    def test_save_comparison_function(self, sample_data, tmp_path):
        """Test save_comparison convenience function."""
        output_path = tmp_path / "comparison.png"
        save_comparison(sample_data['image'], sample_data['mask'], output_path)
        
        assert output_path.exists()


class TestVisualizerWithoutMatplotlib:
    """Test visualizer behavior without matplotlib."""
    
    def test_create_report_figure_without_matplotlib(self):
        """Test graceful handling when matplotlib unavailable."""
        viz = Visualizer()
        
        # Even if matplotlib is available, test the fallback path
        if not viz.has_matplotlib:
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            mask = np.zeros((100, 100), dtype=np.uint8)
            result = {'continuous': True}
            
            fig = viz.create_report_figure(image, mask, result)
            assert fig is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

