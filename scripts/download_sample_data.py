#!/usr/bin/env python3
"""
Download sample data for testing the vessel connectivity checker.

This script:
1. Creates synthetic vessel images for immediate testing
2. Optionally downloads real DRIVE retinal images (if available)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_sample_data, ensure_dir, get_data_dir


def create_synthetic_samples():
    """Create synthetic vessel images for testing."""
    print("=" * 60)
    print("Creating Synthetic Vessel Samples")
    print("=" * 60)
    
    samples = setup_sample_data()
    
    print("\n✓ Created sample files:")
    for name, path in samples.items():
        print(f"  • {name}: {path}")
    
    return samples


def download_drive_sample():
    """
    Attempt to download a sample from DRIVE dataset.
    Note: Full DRIVE dataset requires registration.
    """
    print("\n" + "=" * 60)
    print("DRIVE Dataset Information")
    print("=" * 60)
    
    drive_dir = get_data_dir() / "DRIVE"
    
    print("""
The DRIVE (Digital Retinal Images for Vessel Extraction) dataset 
is a standard benchmark for retinal vessel segmentation.

To use DRIVE data with this tool:

1. Register at: https://drive.grand-challenge.org/
2. Download the dataset
3. Extract to: {drive_dir}
   
   Expected structure:
   {drive_dir}/
   ├── training/
   │   ├── images/
   │   └── 1st_manual/
   └── test/
       ├── images/
       └── 1st_manual/

4. Run the checker:
   python run_checker.py --image data/DRIVE/training/images/21_training.tif \\
                         --mask data/DRIVE/training/1st_manual/21_manual1.gif
""".format(drive_dir=drive_dir))


def create_additional_samples():
    """Create additional test cases with different characteristics."""
    import numpy as np
    from PIL import Image, ImageDraw
    
    print("\n" + "=" * 60)
    print("Creating Additional Test Cases")
    print("=" * 60)
    
    samples_dir = ensure_dir(get_data_dir() / "samples")
    
    # Test case: Dense vessel network
    print("\n📸 Creating dense vessel network sample...")
    size = 512
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:] = [35, 25, 25]  # Dark background
    
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)
    
    # Create dense network
    vessel_color = (160, 70, 70)
    
    # Multiple horizontal vessels
    for y in [150, 256, 362]:
        draw.line([(30, y), (482, y)], fill=vessel_color, width=6)
    
    # Connecting branches
    for x in [100, 200, 300, 400]:
        draw.line([(x, 150), (x, 362)], fill=vessel_color, width=4)
    
    # Diagonal branches
    draw.line([(150, 150), (100, 80)], fill=vessel_color, width=3)
    draw.line([(350, 150), (400, 80)], fill=vessel_color, width=3)
    draw.line([(150, 362), (100, 432)], fill=vessel_color, width=3)
    draw.line([(350, 362), (400, 432)], fill=vessel_color, width=3)
    
    dense_image_path = samples_dir / "dense_network_image.png"
    pil_img.save(dense_image_path)
    print(f"  ✓ {dense_image_path}")
    
    # Create corresponding mask
    mask = np.zeros((size, size), dtype=np.uint8)
    pil_mask = Image.fromarray(mask)
    draw_mask = ImageDraw.Draw(pil_mask)
    
    for y in [150, 256, 362]:
        draw_mask.line([(30, y), (482, y)], fill=255, width=6)
    for x in [100, 200, 300, 400]:
        draw_mask.line([(x, 150), (x, 362)], fill=255, width=4)
    draw_mask.line([(150, 150), (100, 80)], fill=255, width=3)
    draw_mask.line([(350, 150), (400, 80)], fill=255, width=3)
    draw_mask.line([(150, 362), (100, 432)], fill=255, width=3)
    draw_mask.line([(350, 362), (400, 432)], fill=255, width=3)
    
    dense_mask_path = samples_dir / "dense_network_mask.png"
    pil_mask.save(dense_mask_path)
    print(f"  ✓ {dense_mask_path}")
    
    # Test case: Sparse vessels with clear breaks
    print("\n📸 Creating fragmented vessel sample...")
    
    frag_mask = np.array(pil_mask)
    # Add multiple breaks
    frag_mask[145:165, 140:180] = 0  # Break in top vessel
    frag_mask[251:265, 240:280] = 0  # Break in middle vessel
    frag_mask[357:372, 340:380] = 0  # Break in bottom vessel
    frag_mask[180:220, 195:210] = 0  # Break in vertical
    frag_mask[280:320, 295:310] = 0  # Break in vertical
    
    frag_mask_path = samples_dir / "fragmented_network_mask.png"
    Image.fromarray(frag_mask).save(frag_mask_path)
    print(f"  ✓ {frag_mask_path}")
    
    # Test case: Noisy segmentation
    print("\n📸 Creating noisy segmentation sample...")
    
    noisy_mask = np.array(pil_mask)
    # Add random noise blobs
    rng = np.random.RandomState(42)
    for _ in range(20):
        cx, cy = rng.randint(50, size-50, 2)
        r = rng.randint(5, 15)
        y, x = np.ogrid[:size, :size]
        blob = (x - cx)**2 + (y - cy)**2 <= r**2
        noisy_mask[blob] = 255
    
    noisy_mask_path = samples_dir / "noisy_network_mask.png"
    Image.fromarray(noisy_mask).save(noisy_mask_path)
    print(f"  ✓ {noisy_mask_path}")
    
    print("\n✓ Additional test cases created!")
    
    return {
        'dense_image': dense_image_path,
        'dense_mask': dense_mask_path,
        'fragmented_mask': frag_mask_path,
        'noisy_mask': noisy_mask_path,
    }


def main():
    """Main entry point."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║     Vessel Connectivity Checker - Sample Data Setup       ║
╚═══════════════════════════════════════════════════════════╝
""")
    
    # Create synthetic samples
    basic_samples = create_synthetic_samples()
    
    # Create additional test cases
    additional_samples = create_additional_samples()
    
    # Show DRIVE information
    download_drive_sample()
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    
    print("""
You can now run the checker with:

  # Demo with synthetic data
  python run_checker.py --demo --skip-vlm

  # Analyze specific samples
  python run_checker.py \\
      --image data/samples/synthetic_vessel_image.png \\
      --mask data/samples/synthetic_vessel_mask_broken.png

  # With VLM analysis (requires OPENAI_API_KEY)
  export OPENAI_API_KEY='your-key'
  python run_checker.py --demo
""")


if __name__ == "__main__":
    main()

