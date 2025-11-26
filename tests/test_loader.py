"""
Tests for segmentation loader.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.segmentation_loader import SegmentationLoader, generate_synthetic_mask
from src.utils import setup_sample_data


class TestSegmentationLoader:
    """Test cases for SegmentationLoader."""
    
    @pytest.fixture
    def loader(self):
        """Create a loader instance."""
        return SegmentationLoader(target_size=(256, 256))
    
    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create sample test data."""
        from PIL import Image
        
        # Create test image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        image_path = tmp_path / "test_image.png"
        Image.fromarray(image).save(image_path)
        
        # Create test mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255
        mask_path = tmp_path / "test_mask.png"
        Image.fromarray(mask).save(mask_path)
        
        return {"image": image_path, "mask": mask_path}
    
    def test_load_image(self, loader, sample_data):
        """Test image loading."""
        image = loader.load_image(sample_data["image"])
        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 3
        assert image.shape[2] == 3  # RGB
    
    def test_load_mask(self, loader, sample_data):
        """Test mask loading."""
        mask = loader.load_mask(sample_data["mask"])
        assert isinstance(mask, np.ndarray)
        assert len(mask.shape) == 2  # Grayscale
        assert set(np.unique(mask)).issubset({0, 255})  # Binary
    
    def test_load_pair(self, loader, sample_data):
        """Test loading image-mask pair."""
        image, mask = loader.load_pair(
            sample_data["image"], 
            sample_data["mask"],
            resize=True
        )
        
        # Check sizes match target
        assert image.shape[:2] == (256, 256)
        assert mask.shape[:2] == (256, 256)
    
    def test_resize_image(self, loader):
        """Test image resizing."""
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        resized = loader.resize_image(image)
        assert resized.shape == (256, 256, 3)
    
    def test_normalize_image(self, loader):
        """Test image normalization."""
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        normalized = loader.normalize_image(image)
        assert normalized.max() == 1.0
        assert normalized.dtype == np.float32
    
    def test_create_composite(self, loader, sample_data):
        """Test composite image creation."""
        image = loader.load_image(sample_data["image"])
        mask = loader.load_mask(sample_data["mask"])
        
        composite = loader.create_composite(image, mask, alpha=0.5)
        assert composite.shape == image.shape
        assert composite.dtype == np.uint8
    
    def test_to_base64(self, loader, sample_data):
        """Test base64 encoding."""
        image = loader.load_image(sample_data["image"])
        b64 = loader.to_base64(image)
        
        assert isinstance(b64, str)
        assert len(b64) > 0
    
    def test_file_not_found(self, loader):
        """Test error handling for missing files."""
        with pytest.raises(FileNotFoundError):
            loader.load_image("nonexistent_file.png")
    
    def test_create_side_by_side(self, loader, sample_data):
        """Test side-by-side comparison creation."""
        image = loader.load_image(sample_data["image"])
        mask = loader.load_mask(sample_data["mask"])
        
        combined = loader.create_side_by_side(image, mask, gap=5)
        
        h, w = image.shape[:2]
        expected_width = w * 2 + 5
        assert combined.shape == (h, expected_width, 3)


class TestSyntheticMask:
    """Test synthetic mask generation."""
    
    def test_otsu_method(self):
        """Test Otsu thresholding."""
        # Create image with clear bimodal distribution
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[25:75, 25:75] = 200
        
        mask = generate_synthetic_mask(image, method="otsu")
        assert mask.shape == (100, 100)
        assert set(np.unique(mask)).issubset({0, 255})
    
    def test_simple_method(self):
        """Test simple thresholding."""
        image = np.full((100, 100), 100, dtype=np.uint8)
        mask = generate_synthetic_mask(image, method="simple")
        assert mask.shape == (100, 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

