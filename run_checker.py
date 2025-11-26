#!/usr/bin/env python3
"""
Vessel Connectivity Checker - CLI Script (API-based, Optional)

⚠️  NOTE: This script requires an OpenAI API key (costs money).
    For FREE analysis using ChatGPT web interface, use instead:
    
    python prepare_for_chatgpt.py --demo
    python parse_response.py

Usage:
    python run_checker.py --image path/to/img.png --mask path/to/mask.png
    python run_checker.py --image img.png --mask mask.png --output results/
    python run_checker.py --demo  # Run with synthetic sample data
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.segmentation_loader import SegmentationLoader
from src.connectivity_prompts import ConnectivityPrompts
from src.connectivity_checker import ConnectivityChecker
from src.visualization import Visualizer
from src.utils import (
    setup_sample_data, 
    save_json, 
    generate_report_id,
    print_result,
    ensure_dir,
    Timer
)

# API interface is optional - import only when needed
def get_vlm_interface(backend='openai', model='gpt-4o-mini'):
    """Lazily import and create VLM interface."""
    try:
        from src.api_optional.vlm_interface import VLMInterface
        return VLMInterface(backend=backend, model=model)
    except ImportError as e:
        print(f"❌ VLM interface not available: {e}")
        print("\n💡 For FREE analysis without API, use:")
        print("   python prepare_for_chatgpt.py --demo")
        print("   python parse_response.py")
        sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Vessel Connectivity Checker - Evaluate vessel segmentation quality using VLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze a single image-mask pair
    python run_checker.py --image retina.png --mask vessels.png

    # Run with synthetic demo data (no API needed for visualization)
    python run_checker.py --demo --skip-vlm

    # Full analysis with VLM
    python run_checker.py --image img.png --mask mask.png --prompt general_continuity

    # Save results to custom directory  
    python run_checker.py --image img.png --mask mask.png --output results/

Available prompts:
    - general_continuity (default)
    - broken_vessel_detection
    - continuity_score
    - bifurcation_analysis
    - anatomical_sanity
    - segmentation_quality
        """
    )
    
    parser.add_argument(
        '--image', '-i',
        type=str,
        help='Path to the input image (PNG, JPG, TIFF)'
    )
    
    parser.add_argument(
        '--mask', '-m',
        type=str,
        help='Path to the segmentation mask'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='Output directory for results (default: output)'
    )
    
    parser.add_argument(
        '--prompt', '-p',
        type=str,
        default='general_continuity',
        choices=ConnectivityPrompts.list_prompts(),
        help='Prompt template to use (default: general_continuity)'
    )
    
    parser.add_argument(
        '--backend',
        type=str,
        default='openai',
        choices=['openai', 'llava-med'],
        help='VLM backend to use (default: openai)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o-mini',
        help='Model to use (default: gpt-4o-mini for OpenAI)'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run with synthetic sample data'
    )
    
    parser.add_argument(
        '--skip-vlm',
        action='store_true',
        help='Skip VLM analysis (visualization only)'
    )
    
    parser.add_argument(
        '--full-analysis',
        action='store_true',
        help='Run all analysis prompts (not just one)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save output files'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


def run_demo(args):
    """Run demo with synthetic data."""
    print("\n🔬 Running Vessel Connectivity Checker Demo")
    print("=" * 50)
    
    # Create sample data
    print("\n📁 Creating synthetic sample data...")
    samples = setup_sample_data()
    
    image_path = samples['image']
    mask_path = samples['mask_broken'] if not args.skip_vlm else samples['mask']
    
    print(f"\n📷 Image: {image_path}")
    print(f"🎭 Mask: {mask_path}")
    
    # Load images
    loader = SegmentationLoader()
    image, mask = loader.load_pair(image_path, mask_path, resize=False)
    
    print(f"\n📐 Image shape: {image.shape}")
    print(f"📐 Mask shape: {mask.shape}")
    
    # Create visualization
    visualizer = Visualizer()
    output_dir = ensure_dir(args.output)
    
    # Save overlay
    overlay = visualizer.overlay_mask(image, mask)
    overlay_path = output_dir / "demo_overlay.png"
    visualizer.save_image(overlay, overlay_path)
    print(f"\n💾 Saved overlay: {overlay_path}")
    
    # Save comparison
    comparison = visualizer.create_comparison(image, mask)
    comparison_path = output_dir / "demo_comparison.png"
    visualizer.save_image(comparison, comparison_path)
    print(f"💾 Saved comparison: {comparison_path}")
    
    if not args.skip_vlm:
        print("\n🤖 Running VLM analysis...")
        print("   ⚠️  This requires an API key. For FREE analysis, use:")
        print("      python prepare_for_chatgpt.py")
        try:
            vlm = get_vlm_interface(backend=args.backend, model=args.model)
            checker = ConnectivityChecker(vlm)
            
            prompt_template = ConnectivityPrompts.get_prompt(args.prompt)
            
            with Timer("VLM Analysis"):
                result = checker.check_connectivity(
                    image_path, 
                    mask_path,
                    prompt_template.template
                )
            
            # Print result
            print_result(result.to_dict())
            
            # Save report
            if not args.no_save:
                report_path = output_dir / f"demo_report_{generate_report_id()}.json"
                save_json(result.to_dict(), report_path)
                print(f"\n💾 Saved report: {report_path}")
            
            # Save figure
            fig = visualizer.create_report_figure(image, mask, result.to_dict())
            if fig is not None:
                fig_path = output_dir / "demo_analysis.png"
                visualizer.save_figure(fig, fig_path)
                print(f"💾 Saved analysis figure: {fig_path}")
            
            # Print raw VLM response if verbose
            if args.verbose and result.raw_response:
                print("\n📝 Raw VLM Response:")
                print("-" * 40)
                print(result.raw_response)
                print("-" * 40)
                
        except Exception as e:
            print(f"\n⚠️  VLM analysis failed: {e}")
            print("Tip: Make sure OPENAI_API_KEY is set for OpenAI backend")
    else:
        print("\n⏭️  Skipped VLM analysis (--skip-vlm)")
        
        # Create a mock result for visualization
        mock_result = {
            'continuous': True,
            'confidence': 0.0,
            'broken_segments': [],
            'quality_score': None,
            'note': 'VLM analysis skipped'
        }
        
        fig = visualizer.create_report_figure(image, mask, mock_result)
        if fig is not None and not args.no_save:
            fig_path = output_dir / "demo_visualization.png"
            visualizer.save_figure(fig, fig_path)
            print(f"💾 Saved visualization: {fig_path}")
    
    print("\n✅ Demo complete!")
    print(f"📂 Output directory: {output_dir.absolute()}")


def run_analysis(args):
    """Run analysis on provided image and mask."""
    print("\n🔬 Vessel Connectivity Checker")
    print("=" * 50)
    
    # Validate inputs
    image_path = Path(args.image)
    mask_path = Path(args.mask)
    
    if not image_path.exists():
        print(f"❌ Error: Image not found: {image_path}")
        sys.exit(1)
    if not mask_path.exists():
        print(f"❌ Error: Mask not found: {mask_path}")
        sys.exit(1)
    
    print(f"\n📷 Image: {image_path}")
    print(f"🎭 Mask: {mask_path}")
    
    # Load images
    loader = SegmentationLoader()
    image, mask = loader.load_pair(image_path, mask_path, resize=True)
    
    print(f"\n📐 Loaded image: {image.shape}")
    print(f"📐 Loaded mask: {mask.shape}")
    
    # Setup output
    output_dir = ensure_dir(args.output)
    report_id = generate_report_id()
    
    # Create visualizer
    visualizer = Visualizer()
    
    # Save overlay
    overlay = visualizer.overlay_mask(image, mask)
    if not args.no_save:
        overlay_path = output_dir / f"{report_id}_overlay.png"
        visualizer.save_image(overlay, overlay_path)
        print(f"\n💾 Saved overlay: {overlay_path}")
    
    # Run VLM analysis
    if not args.skip_vlm:
        print("\n🤖 Initializing VLM...")
        print("   ⚠️  This requires an API key. For FREE analysis, use:")
        print("      python prepare_for_chatgpt.py")
        
        try:
            vlm = get_vlm_interface(backend=args.backend, model=args.model)
            checker = ConnectivityChecker(vlm)
            
            if args.full_analysis:
                print("📊 Running full analysis (multiple prompts)...")
                with Timer("Full Analysis"):
                    results = checker.run_full_analysis(image_path, mask_path)
                
                # Generate combined report
                report = checker.generate_report(results)
                
                print("\n" + "=" * 50)
                print("  FULL ANALYSIS RESULTS")
                print("=" * 50)
                
                for name, result in results.items():
                    print(f"\n--- {name.upper()} ---")
                    print_result(result.to_dict())
                
                print("\n--- SUMMARY ---")
                print(f"Overall Continuous: {report['summary']['overall_continuous']}")
                print(f"Total Broken Segments: {report['summary']['total_broken_segments']}")
                print(f"Average Confidence: {report['summary']['average_confidence']:.2f}")
                
                if not args.no_save:
                    report_path = output_dir / f"{report_id}_full_report.json"
                    save_json(report, report_path)
                    print(f"\n💾 Saved full report: {report_path}")
                    
            else:
                prompt_template = ConnectivityPrompts.get_prompt(args.prompt)
                print(f"📝 Using prompt: {args.prompt}")
                
                with Timer("VLM Analysis"):
                    result = checker.check_connectivity(
                        image_path,
                        mask_path,
                        prompt_template.template
                    )
                
                # Print result
                print_result(result.to_dict())
                
                # Save report
                if not args.no_save:
                    report_path = output_dir / f"{report_id}_report.json"
                    save_json(result.to_dict(), report_path)
                    print(f"\n💾 Saved report: {report_path}")
                
                # Save figure
                fig = visualizer.create_report_figure(image, mask, result.to_dict())
                if fig is not None and not args.no_save:
                    fig_path = output_dir / f"{report_id}_analysis.png"
                    visualizer.save_figure(fig, fig_path)
                    print(f"💾 Saved analysis figure: {fig_path}")
                
                # Print raw response if verbose
                if args.verbose and result.raw_response:
                    print("\n📝 Raw VLM Response:")
                    print("-" * 40)
                    print(result.raw_response)
                    print("-" * 40)
                    
        except ValueError as e:
            if "API key" in str(e):
                print(f"\n❌ Error: {e}")
                print("\nTo fix this:")
                print("  export OPENAI_API_KEY='your-api-key'")
                print("\nOr run with --skip-vlm for visualization only")
            else:
                raise
        except Exception as e:
            print(f"\n❌ VLM analysis failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    else:
        print("\n⏭️  Skipped VLM analysis (--skip-vlm)")
        
        # Save comparison visualization
        comparison = visualizer.create_comparison(image, mask)
        if not args.no_save:
            comparison_path = output_dir / f"{report_id}_comparison.png"
            visualizer.save_image(comparison, comparison_path)
            print(f"💾 Saved comparison: {comparison_path}")
    
    print("\n✅ Analysis complete!")
    print(f"📂 Output directory: {Path(args.output).absolute()}")


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.demo:
        run_demo(args)
    elif args.image and args.mask:
        run_analysis(args)
    else:
        print("❌ Error: Either --demo or both --image and --mask are required")
        print("\nUsage examples:")
        print("  python run_checker.py --demo")
        print("  python run_checker.py --image img.png --mask mask.png")
        print("\nRun with --help for more options")
        sys.exit(1)


if __name__ == "__main__":
    main()

