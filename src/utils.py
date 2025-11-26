"""
Utility functions for vessel connectivity checker.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import urllib.request
import zipfile
import shutil


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_data_dir() -> Path:
    """Get the data directory."""
    return get_project_root() / "data"


def save_json(data: Dict[str, Any], path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        path: Output path
        indent: JSON indentation
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        path: File path
        
    Returns:
        Loaded data
    """
    with open(path, 'r') as f:
        return json.load(f)


def generate_report_id() -> str:
    """Generate a unique report ID based on timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hash_suffix = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:6]
    return f"report_{timestamp}_{hash_suffix}"


def find_image_pairs(
    directory: Union[str, Path],
    image_patterns: List[str] = None,
    mask_patterns: List[str] = None
) -> List[Dict[str, Path]]:
    """
    Find matching image-mask pairs in a directory.
    
    Args:
        directory: Directory to search
        image_patterns: Patterns for image files
        mask_patterns: Patterns for mask files
        
    Returns:
        List of dicts with 'image' and 'mask' paths
    """
    directory = Path(directory)
    
    if image_patterns is None:
        image_patterns = ['*_image.*', '*_original.*', '*_test.*']
    if mask_patterns is None:
        mask_patterns = ['*_mask.*', '*_manual.*', '*_segmentation.*', '*_1stHO.*']
    
    pairs = []
    
    # Find all image files
    image_files = []
    for pattern in image_patterns:
        image_files.extend(directory.glob(pattern))
    
    # For each image, find matching mask
    for image_path in image_files:
        base_name = image_path.stem.replace('_image', '').replace('_original', '').replace('_test', '')
        
        # Look for matching mask
        for mask_pattern in mask_patterns:
            for mask_path in directory.glob(mask_pattern):
                mask_base = mask_path.stem.replace('_mask', '').replace('_manual', '').replace('_segmentation', '').replace('_1stHO', '')
                if base_name in mask_base or mask_base in base_name:
                    pairs.append({'image': image_path, 'mask': mask_path})
                    break
    
    return pairs


def download_file(url: str, output_path: Union[str, Path], show_progress: bool = True) -> Path:
    """
    Download a file from URL.
    
    Args:
        url: URL to download
        output_path: Output file path
        show_progress: Whether to show progress
        
    Returns:
        Path to downloaded file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if show_progress:
        print(f"Downloading {url}...")
    
    urllib.request.urlretrieve(url, output_path)
    
    if show_progress:
        print(f"Saved to {output_path}")
    
    return output_path


def extract_zip(zip_path: Union[str, Path], extract_to: Union[str, Path]) -> Path:
    """
    Extract a ZIP file.
    
    Args:
        zip_path: Path to ZIP file
        extract_to: Extraction directory
        
    Returns:
        Extraction directory path
    """
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    return extract_to


def setup_sample_data(data_dir: Optional[Union[str, Path]] = None) -> Dict[str, Path]:
    """
    Set up sample data for testing.
    Creates synthetic test images if no real data available.
    
    Args:
        data_dir: Data directory (uses default if not specified)
        
    Returns:
        Dict with paths to sample files
    """
    import numpy as np
    from PIL import Image
    
    if data_dir is None:
        data_dir = get_data_dir() / "samples"
    
    data_dir = ensure_dir(data_dir)
    
    samples = {}
    
    # Create a synthetic vessel-like image
    size = 512
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:] = [40, 30, 30]  # Dark reddish background (like retinal image)
    
    # Add some "vessel-like" structures
    rng = np.random.RandomState(42)
    
    # Draw vessel-like lines
    from PIL import ImageDraw
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)
    
    # Main vessels
    vessel_color = (180, 80, 80)
    
    # Horizontal main vessel
    draw.line([(50, 256), (462, 256)], fill=vessel_color, width=8)
    # Branches
    draw.line([(150, 256), (100, 150)], fill=vessel_color, width=5)
    draw.line([(150, 256), (100, 362)], fill=vessel_color, width=5)
    draw.line([(300, 256), (350, 150)], fill=vessel_color, width=6)
    draw.line([(300, 256), (350, 362)], fill=vessel_color, width=6)
    draw.line([(350, 150), (400, 100)], fill=vessel_color, width=4)
    draw.line([(350, 362), (400, 412)], fill=vessel_color, width=4)
    
    # Add some noise/texture
    image = np.array(pil_img)
    noise = rng.normal(0, 10, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    image_path = data_dir / "synthetic_vessel_image.png"
    Image.fromarray(image).save(image_path)
    samples['image'] = image_path
    
    # Create corresponding mask
    mask = np.zeros((size, size), dtype=np.uint8)
    pil_mask = Image.fromarray(mask)
    draw_mask = ImageDraw.Draw(pil_mask)
    
    # Same vessels but white on black
    mask_color = 255
    draw_mask.line([(50, 256), (462, 256)], fill=mask_color, width=8)
    draw_mask.line([(150, 256), (100, 150)], fill=mask_color, width=5)
    draw_mask.line([(150, 256), (100, 362)], fill=mask_color, width=5)
    draw_mask.line([(300, 256), (350, 150)], fill=mask_color, width=6)
    draw_mask.line([(300, 256), (350, 362)], fill=mask_color, width=6)
    draw_mask.line([(350, 150), (400, 100)], fill=mask_color, width=4)
    draw_mask.line([(350, 362), (400, 412)], fill=mask_color, width=4)
    
    mask_path = data_dir / "synthetic_vessel_mask.png"
    pil_mask.save(mask_path)
    samples['mask'] = mask_path
    
    # Create a "broken" mask for testing discontinuity detection
    mask_broken = np.array(pil_mask)
    # Add gaps
    mask_broken[250:262, 200:230] = 0  # Gap in main vessel
    mask_broken[150:180, 320:350] = 0  # Gap in branch
    
    mask_broken_path = data_dir / "synthetic_vessel_mask_broken.png"
    Image.fromarray(mask_broken).save(mask_broken_path)
    samples['mask_broken'] = mask_broken_path
    
    print(f"Sample data created in {data_dir}")
    print(f"  Image: {samples['image']}")
    print(f"  Mask: {samples['mask']}")
    print(f"  Broken mask: {samples['mask_broken']}")
    
    return samples


def format_result_for_display(result: Dict[str, Any]) -> str:
    """
    Format analysis result for terminal display.
    
    Args:
        result: Analysis result dictionary
        
    Returns:
        Formatted string
    """
    lines = []
    lines.append("=" * 50)
    lines.append("  VESSEL CONNECTIVITY ANALYSIS RESULT")
    lines.append("=" * 50)
    
    if 'continuous' in result:
        status = "✓ CONTINUOUS" if result['continuous'] else "✗ DISCONTINUOUS"
        lines.append(f"\n  Status: {status}")
    
    if 'confidence' in result:
        lines.append(f"  Confidence: {result['confidence']:.1%}")
    
    if 'quality_score' in result and result['quality_score'] is not None:
        lines.append(f"  Quality Score: {result['quality_score']:.1f}/1.0")
    
    if 'broken_segments' in result and result['broken_segments']:
        lines.append(f"\n  Broken Segments ({len(result['broken_segments'])}):")
        for i, seg in enumerate(result['broken_segments'], 1):
            lines.append(f"    {i}. {seg}")
    
    if 'bifurcation_quality' in result and result['bifurcation_quality']:
        lines.append(f"\n  Bifurcation Quality: {result['bifurcation_quality'].upper()}")
    
    if 'anatomically_plausible' in result and result['anatomically_plausible'] is not None:
        anat = "Yes" if result['anatomically_plausible'] else "No"
        lines.append(f"  Anatomically Plausible: {anat}")
    
    lines.append("\n" + "=" * 50)
    
    return "\n".join(lines)


def print_result(result: Dict[str, Any]) -> None:
    """Print formatted analysis result."""
    print(format_result_for_display(result))


class Timer:
    """Simple timer context manager for profiling."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        import time
        self.elapsed = time.time() - self.start_time
        print(f"{self.name}: {self.elapsed:.2f}s")

