#!/usr/bin/env python3
"""
Compare Skeleton Metrics vs VLM Estimates.

Computes ground-truth topology metrics from a manual segmentation with skeletonization on top,
then compares with VLM-estimated metrics from a parsed response.

Usage:
    # After getting VLM response for prompt 07:
    python compare_metrics.py --segmentation data/21_manual1.png --response output/result_xxx.json
    
    # Or run skeleton metrics only (no VLM comparison):
    python compare_metrics.py --segmentation data/21_manual1.png --skeleton-only
    
    # Demo with synthetic data:
    python compare_metrics.py --demo
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.skeleton_metrics import (
    compute_metrics_from_file,
    format_metrics_for_display,
    create_annotated_skeleton,
    load_binary_mask,
    SKIMAGE_AVAILABLE
)
from src.utils import save_json, load_json, ensure_dir, generate_report_id


def compute_comparison(skeleton_metrics: dict, vlm_metrics: dict) -> dict:
    """
    Compare skeleton-computed metrics with VLM-estimated metrics.
    
    Args:
        skeleton_metrics: Computed metrics from skeletonization
        vlm_metrics: Estimated metrics from VLM response
        
    Returns:
        Comparison dictionary with deltas and analysis
    """
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "skeleton_metrics": skeleton_metrics,
        "vlm_metrics": {k: v for k, v in vlm_metrics.items() 
                       if k not in ['raw_response', 'parsed_at', 'observations']},
    }
    
    # Compute deltas for numerical metrics
    deltas = {}
    
    # Connected components
    skel_cc = skeleton_metrics.get('connected_components')
    vlm_cc = vlm_metrics.get('connected_components')
    if skel_cc is not None and vlm_cc is not None:
        deltas['connected_components'] = {
            'skeleton': skel_cc,
            'vlm': vlm_cc,
            'difference': vlm_cc - skel_cc,
            'match': skel_cc == vlm_cc,
        }
    
    # Branch points
    skel_bp = skeleton_metrics.get('branch_points')
    vlm_bp = vlm_metrics.get('branch_points')
    if skel_bp is not None and vlm_bp is not None:
        error_pct = abs(vlm_bp - skel_bp) / max(1, skel_bp) * 100
        deltas['branch_points'] = {
            'skeleton': skel_bp,
            'vlm': vlm_bp,
            'difference': vlm_bp - skel_bp,
            'error_percent': round(error_pct, 1),
        }
    
    # End points
    skel_ep = skeleton_metrics.get('end_points')
    vlm_ep = vlm_metrics.get('end_points')
    if skel_ep is not None and vlm_ep is not None:
        error_pct = abs(vlm_ep - skel_ep) / max(1, skel_ep) * 100
        deltas['end_points'] = {
            'skeleton': skel_ep,
            'vlm': vlm_ep,
            'difference': vlm_ep - skel_ep,
            'error_percent': round(error_pct, 1),
        }
    
    # Vessel density
    skel_density = skeleton_metrics.get('vessel_density')
    vlm_density_pct = vlm_metrics.get('vessel_density_pct')
    if skel_density is not None and vlm_density_pct is not None:
        deltas['vessel_density'] = {
            'skeleton': round(skel_density * 100, 1),
            'vlm': round(vlm_density_pct * 100, 1),
            'difference_pct': round((vlm_density_pct - skel_density) * 100, 1),
        }
    
    # Connectivity score
    skel_conn = skeleton_metrics.get('connectivity_score')
    vlm_conn = vlm_metrics.get('connectivity_quality')
    if skel_conn is not None and vlm_conn is not None:
        deltas['connectivity_score'] = {
            'skeleton': round(skel_conn, 2),
            'vlm': round(vlm_conn, 2),
            'difference': round(vlm_conn - skel_conn, 2),
        }
    
    comparison['deltas'] = deltas
    
    # Overall agreement score
    agreement_scores = []
    
    if 'connected_components' in deltas:
        # Exact match for connected components is important
        agreement_scores.append(1.0 if deltas['connected_components']['match'] else 0.0)
    
    if 'branch_points' in deltas:
        # Allow 30% error tolerance
        error = deltas['branch_points']['error_percent']
        agreement_scores.append(max(0, 1.0 - error / 100))
    
    if 'end_points' in deltas:
        error = deltas['end_points']['error_percent']
        agreement_scores.append(max(0, 1.0 - error / 100))
    
    if 'connectivity_score' in deltas:
        diff = abs(deltas['connectivity_score']['difference'])
        agreement_scores.append(max(0, 1.0 - diff * 2))  # 0.5 diff = 0 agreement
    
    if agreement_scores:
        comparison['overall_agreement'] = round(sum(agreement_scores) / len(agreement_scores), 2)
    else:
        comparison['overall_agreement'] = None
    
    return comparison


def print_comparison(comparison: dict):
    """Print formatted comparison results."""
    print("\n" + "=" * 65)
    print("  SKELETON vs VLM TOPOLOGY COMPARISON")
    print("=" * 65)
    
    deltas = comparison.get('deltas', {})
    
    # Header
    print(f"\n  {'Metric':<25} {'Skeleton':>12} {'VLM':>12} {'Δ':>10}")
    print("  " + "-" * 59)
    
    # Connected components
    if 'connected_components' in deltas:
        d = deltas['connected_components']
        match_icon = "✓" if d['match'] else "✗"
        print(f"  {'Connected Components':<25} {d['skeleton']:>12} {d['vlm']:>12} {d['difference']:>+10} {match_icon}")
    
    # Branch points
    if 'branch_points' in deltas:
        d = deltas['branch_points']
        err_str = f"({d['error_percent']:.0f}%)"
        print(f"  {'Branch Points':<25} {d['skeleton']:>12} {d['vlm']:>12} {d['difference']:>+10} {err_str}")
    
    # End points
    if 'end_points' in deltas:
        d = deltas['end_points']
        err_str = f"({d['error_percent']:.0f}%)"
        print(f"  {'End Points':<25} {d['skeleton']:>12} {d['vlm']:>12} {d['difference']:>+10} {err_str}")
    
    # Vessel density
    if 'vessel_density' in deltas:
        d = deltas['vessel_density']
        print(f"  {'Vessel Density (%)':<25} {d['skeleton']:>11.1f}% {d['vlm']:>11.1f}% {d['difference_pct']:>+9.1f}%")
    
    # Connectivity score
    if 'connectivity_score' in deltas:
        d = deltas['connectivity_score']
        print(f"  {'Connectivity Score':<25} {d['skeleton']:>12.2f} {d['vlm']:>12.2f} {d['difference']:>+10.2f}")
    
    print("  " + "-" * 59)
    
    # Overall agreement
    agreement = comparison.get('overall_agreement')
    if agreement is not None:
        if agreement >= 0.8:
            grade = "STRONG"
            icon = "🟢"
        elif agreement >= 0.5:
            grade = "MODERATE"
            icon = "🟡"
        else:
            grade = "WEAK"
            icon = "🔴"
        print(f"\n  {icon} Overall Agreement: {agreement:.0%} ({grade})")
    
    # Observations from VLM
    vlm_metrics = comparison.get('vlm_metrics', {})
    if 'observations' in comparison.get('vlm_metrics', {}):
        # This won't be in vlm_metrics since we filtered it out
        pass
    
    print("\n" + "=" * 65)


def run_demo():
    """Run demo with synthetic data."""
    from src.utils import get_data_dir
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║           TOPOLOGY METRICS COMPARISON - DEMO                  ║
╚═══════════════════════════════════════════════════════════════╝
""")
    
    # Check for sample data
    data_dir = get_data_dir()
    sample_files = [
        data_dir / "21_manual1.png",
        data_dir / "samples" / "synthetic_vessel_mask.png",
    ]
    
    segmentation_path = None
    for path in sample_files:
        if path.exists():
            segmentation_path = path
            break
    
    if segmentation_path is None:
        print("❌ No sample segmentation found.")
        print("   Run: python prepare_for_chatgpt.py --demo")
        print("   to create sample data first.")
        return
    
    print(f"📊 Computing skeleton metrics for: {segmentation_path}\n")
    
    # Compute skeleton metrics
    metrics = compute_metrics_from_file(segmentation_path)
    print(format_metrics_for_display(metrics))
    
    # Save annotated skeleton
    output_dir = ensure_dir(Path(__file__).parent / "output")
    skeleton_path = output_dir / "demo_skeleton.png"
    
    from PIL import Image
    binary_mask = load_binary_mask(segmentation_path)
    annotated = create_annotated_skeleton(binary_mask)
    Image.fromarray(annotated).save(skeleton_path)
    print(f"\n🖼️  Annotated skeleton saved to: {skeleton_path}")
    
    # Save metrics
    metrics_path = output_dir / "demo_skeleton_metrics.json"
    save_json(metrics, metrics_path)
    print(f"💾 Metrics saved to: {metrics_path}")
    
    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 NEXT STEPS:

1. Run: python prepare_for_chatgpt.py --image (desired image) --segmentation (desired segmentation mask)
   select prompt 7
   (This will prepare the image and show the topology metrics prompt)

2. Upload the image to ChatGPT and paste the prompt

3. Paste ChatGPT's response into response.txt

4. Run: python parse_response.py
   (This will parse the topology metrics from the response)

5. Run: python compare_metrics.py --segmentation {seg_path} --response output/result_xxx.json
   (Replace xxx with the actual report ID)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""".format(seg_path=segmentation_path))


def main():
    parser = argparse.ArgumentParser(
        description="Compare skeleton topology metrics with VLM estimates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python compare_metrics.py --demo
    python compare_metrics.py --segmentation data/21_manual1.png --skeleton-only
    python compare_metrics.py --segmentation data/21_manual1.png --response output/result_xxx.json
        """
    )
    
    parser.add_argument('--segmentation', '-s', type=str,
                        help='Path to binary segmentation image')
    parser.add_argument('--response', '-r', type=str,
                        help='Path to parsed VLM response JSON (from parse_response.py)')
    parser.add_argument('--skeleton-only', action='store_true',
                        help='Only compute skeleton metrics (no VLM comparison)')
    parser.add_argument('--output', '-o', type=str,
                        help='Save comparison results to JSON file')
    parser.add_argument('--save-skeleton', action='store_true',
                        help='Save annotated skeleton image')
    parser.add_argument('--demo', action='store_true',
                        help='Run demo with sample data')
    
    args = parser.parse_args()
    
    if not SKIMAGE_AVAILABLE:
        print("❌ scikit-image is required for skeleton metrics.")
        print("   Install with: pip install scikit-image")
        sys.exit(1)
    
    if args.demo:
        run_demo()
        return
    
    if not args.segmentation:
        print("❌ Please provide a segmentation image with --segmentation")
        print("   Or run with --demo for a demo")
        sys.exit(1)
    
    segmentation_path = Path(args.segmentation)
    if not segmentation_path.exists():
        print(f"❌ Segmentation not found: {segmentation_path}")
        sys.exit(1)
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║           TOPOLOGY METRICS COMPARISON                         ║
╚═══════════════════════════════════════════════════════════════╝
""")
    
    # Compute skeleton metrics
    print(f"📊 Computing skeleton metrics for: {segmentation_path}\n")
    skeleton_metrics = compute_metrics_from_file(segmentation_path)
    print(format_metrics_for_display(skeleton_metrics))
    
    # Save annotated skeleton if requested
    if args.save_skeleton:
        output_dir = ensure_dir(Path(__file__).parent / "output")
        skeleton_path = output_dir / f"skeleton_{segmentation_path.stem}.png"
        
        from PIL import Image
        binary_mask = load_binary_mask(segmentation_path)
        annotated = create_annotated_skeleton(binary_mask)
        Image.fromarray(annotated).save(skeleton_path)
        print(f"\n🖼️  Annotated skeleton saved to: {skeleton_path}")
    
    # If skeleton-only, we're done
    if args.skeleton_only or not args.response:
        if not args.skeleton_only and not args.response:
            print("\n💡 To compare with VLM estimates, provide --response path/to/result.json")
        return
    
    # Load VLM response
    response_path = Path(args.response)
    if not response_path.exists():
        print(f"❌ Response file not found: {response_path}")
        sys.exit(1)
    
    print(f"\n📄 Loading VLM response from: {response_path}")
    vlm_metrics = load_json(response_path)
    
    # Check if it's a topology metrics response
    if vlm_metrics.get('prompt_type') != '07_topology_metrics':
        print(f"⚠️  Warning: Response prompt type is '{vlm_metrics.get('prompt_type')}'")
        print("   Expected '07_topology_metrics'. Results may be incomplete.")
    
    # Compute comparison
    comparison = compute_comparison(skeleton_metrics, vlm_metrics)
    
    # Print comparison
    print_comparison(comparison)
    
    # Save comparison if requested
    if args.output:
        save_json(comparison, args.output)
        print(f"\n💾 Comparison saved to: {args.output}")
    else:
        # Auto-save to output directory
        output_dir = ensure_dir(Path(__file__).parent / "output")
        report_id = generate_report_id()
        output_path = output_dir / f"comparison_{report_id}.json"
        save_json(comparison, output_path)
        print(f"\n💾 Comparison saved to: {output_path}")


if __name__ == "__main__":
    main()

