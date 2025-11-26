"""
Skeleton-based topology metrics for vessel segmentation.

Computes quantitative connectivity metrics from binary segmentations
using skeletonization and graph analysis.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Union, Tuple, Optional
from PIL import Image

# Import skeletonize from scikit-image
try:
    from skimage.morphology import skeletonize
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


def check_dependencies():
    """Check if required dependencies are available."""
    if not SKIMAGE_AVAILABLE:
        raise ImportError(
            "scikit-image is required for skeleton metrics.\n"
            "Install with: pip install scikit-image"
        )


def load_binary_mask(path: Union[str, Path]) -> np.ndarray:
    """
    Load a binary segmentation mask from disk.
    
    Args:
        path: Path to the segmentation image
        
    Returns:
        Binary numpy array (True/False)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Segmentation not found: {path}")
    
    img = Image.open(path)
    if img.mode != 'L':
        img = img.convert('L')
    
    arr = np.array(img)
    # Binarize (threshold at 127)
    return arr > 127


def compute_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """
    Compute the morphological skeleton of a binary mask.
    
    Args:
        binary_mask: Binary numpy array
        
    Returns:
        Skeleton as binary numpy array
    """
    check_dependencies()
    return skeletonize(binary_mask)


def count_neighbor_pixels(skeleton: np.ndarray, row: int, col: int) -> int:
    """
    Count the number of neighboring skeleton pixels (8-connectivity).
    
    Args:
        skeleton: Binary skeleton array
        row, col: Pixel coordinates
        
    Returns:
        Number of neighbors (0-8)
    """
    h, w = skeleton.shape
    count = 0
    
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < h and 0 <= nc < w:
                if skeleton[nr, nc]:
                    count += 1
    
    return count


def find_branch_and_end_points(skeleton: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find branch points (>2 neighbors) and end points (1 neighbor) in skeleton.
    
    Args:
        skeleton: Binary skeleton array
        
    Returns:
        Tuple of (branch_points, end_points) as boolean arrays
    """
    h, w = skeleton.shape
    branch_points = np.zeros_like(skeleton, dtype=bool)
    end_points = np.zeros_like(skeleton, dtype=bool)
    
    # Get skeleton pixel coordinates
    skel_rows, skel_cols = np.where(skeleton)
    
    for row, col in zip(skel_rows, skel_cols):
        neighbors = count_neighbor_pixels(skeleton, row, col)
        
        if neighbors > 2:
            branch_points[row, col] = True
        elif neighbors == 1:
            end_points[row, col] = True
    
    return branch_points, end_points


def compute_topology_metrics(
    binary_mask: np.ndarray,
    skeleton: Optional[np.ndarray] = None
) -> Dict[str, any]:
    """
    Compute comprehensive topology metrics from a binary segmentation.
    
    Args:
        binary_mask: Binary segmentation (True = vessel)
        skeleton: Pre-computed skeleton (optional, will compute if not provided)
        
    Returns:
        Dictionary of topology metrics
    """
    check_dependencies()
    
    # Compute skeleton if not provided
    if skeleton is None:
        skeleton = compute_skeleton(binary_mask)
    
    # Connected components (on the skeleton)
    labeled_skeleton = measure.label(skeleton, connectivity=2)
    num_components = labeled_skeleton.max()
    
    # Also compute connected components on original mask for comparison
    labeled_mask = measure.label(binary_mask, connectivity=2)
    num_mask_components = labeled_mask.max()
    
    # Branch and end points
    branch_points, end_points = find_branch_and_end_points(skeleton)
    num_branch_points = np.sum(branch_points)
    num_end_points = np.sum(end_points)
    
    # Skeleton length (total skeleton pixels)
    skeleton_length = np.sum(skeleton)
    
    # Euler characteristic for the skeleton
    # Using regionprops on the skeleton
    regions = measure.regionprops(labeled_skeleton.astype(int))
    total_euler = sum(r.euler_number for r in regions)
    
    # Vessel density (fraction of image covered by vessels)
    vessel_density = np.sum(binary_mask) / binary_mask.size
    
    # Compute average branch length estimate
    # (skeleton_length - branch_point_area) / (num_end_points + num_branch_points)
    # This is a rough estimate
    if num_end_points + num_branch_points > 0:
        avg_branch_length = skeleton_length / max(1, (num_end_points + num_branch_points) / 2)
    else:
        avg_branch_length = skeleton_length
    
    # Connectivity score: fewer components + more branch points = more connected
    # Normalize to 0-1 range (heuristic)
    if num_components == 0:
        connectivity_score = 0.0
    else:
        # Ideal: 1 component, many branches
        component_penalty = min(1.0, 1.0 / num_components)
        branch_bonus = min(1.0, num_branch_points / 50)  # Normalize by typical branch count
        connectivity_score = 0.7 * component_penalty + 0.3 * branch_bonus
    
    return {
        "connected_components": int(num_components),
        "mask_components": int(num_mask_components),
        "branch_points": int(num_branch_points),
        "end_points": int(num_end_points),
        "skeleton_length_px": int(skeleton_length),
        "euler_number": int(total_euler),
        "vessel_density": float(vessel_density),
        "avg_branch_length_px": float(avg_branch_length),
        "connectivity_score": float(connectivity_score),
        "image_size": list(binary_mask.shape),
    }


def compute_metrics_from_file(path: Union[str, Path]) -> Dict[str, any]:
    """
    Convenience function to compute metrics directly from a file path.
    
    Args:
        path: Path to binary segmentation image
        
    Returns:
        Dictionary of topology metrics
    """
    binary_mask = load_binary_mask(path)
    skeleton = compute_skeleton(binary_mask)
    metrics = compute_topology_metrics(binary_mask, skeleton)
    metrics["source_file"] = str(path)
    return metrics


def get_skeleton_image(binary_mask: np.ndarray) -> np.ndarray:
    """
    Get the skeleton as a displayable image (0-255).
    
    Args:
        binary_mask: Binary segmentation
        
    Returns:
        Skeleton as uint8 image (0 or 255)
    """
    skeleton = compute_skeleton(binary_mask)
    return (skeleton.astype(np.uint8) * 255)


def create_annotated_skeleton(
    binary_mask: np.ndarray,
    show_branch_points: bool = True,
    show_end_points: bool = True
) -> np.ndarray:
    """
    Create an RGB image showing skeleton with annotated branch/end points.
    
    Args:
        binary_mask: Binary segmentation
        show_branch_points: Highlight branch points in red
        show_end_points: Highlight end points in blue
        
    Returns:
        RGB image (H, W, 3) as uint8
    """
    skeleton = compute_skeleton(binary_mask)
    branch_points, end_points = find_branch_and_end_points(skeleton)
    
    # Create RGB image
    rgb = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
    
    # Skeleton in white
    rgb[skeleton] = [255, 255, 255]
    
    # Branch points in red
    if show_branch_points:
        # Dilate branch points slightly for visibility
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                shifted = np.roll(np.roll(branch_points, dr, axis=0), dc, axis=1)
                rgb[shifted] = [255, 50, 50]
    
    # End points in blue
    if show_end_points:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                shifted = np.roll(np.roll(end_points, dr, axis=0), dc, axis=1)
                rgb[shifted] = [50, 100, 255]
    
    return rgb


def format_metrics_for_display(metrics: Dict[str, any]) -> str:
    """
    Format metrics dictionary for terminal display.
    
    Args:
        metrics: Dictionary of computed metrics
        
    Returns:
        Formatted string
    """
    lines = [
        "=" * 55,
        "  SKELETON TOPOLOGY METRICS",
        "=" * 55,
        "",
        f"  Connected Components:  {metrics['connected_components']}",
        f"  Branch Points:         {metrics['branch_points']}",
        f"  End Points:            {metrics['end_points']}",
        f"  Skeleton Length:       {metrics['skeleton_length_px']} px",
        f"  Euler Number:          {metrics['euler_number']}",
        f"  Vessel Density:        {metrics['vessel_density']:.2%}",
        f"  Avg Branch Length:     {metrics['avg_branch_length_px']:.1f} px",
        f"  Connectivity Score:    {metrics['connectivity_score']:.2f}",
        "",
        "=" * 55,
    ]
    return "\n".join(lines)


# CLI for standalone testing
if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Compute skeleton topology metrics")
    parser.add_argument("segmentation", type=str, help="Path to binary segmentation image")
    parser.add_argument("--output", "-o", type=str, help="Save metrics to JSON file")
    parser.add_argument("--save-skeleton", type=str, help="Save annotated skeleton image")
    
    args = parser.parse_args()
    
    print(f"\n📊 Computing skeleton metrics for: {args.segmentation}\n")
    
    # Load and compute
    binary_mask = load_binary_mask(args.segmentation)
    metrics = compute_metrics_from_file(args.segmentation)
    
    # Display
    print(format_metrics_for_display(metrics))
    
    # Save JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n💾 Metrics saved to: {args.output}")
    
    # Save annotated skeleton if requested
    if args.save_skeleton:
        annotated = create_annotated_skeleton(binary_mask)
        Image.fromarray(annotated).save(args.save_skeleton)
        print(f"🖼️  Annotated skeleton saved to: {args.save_skeleton}")

