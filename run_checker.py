#!/usr/bin/env python3
"""
Vessel Connectivity Checker - API-based CLI Script

⚠️  This script requires an OpenAI API key (costs money).
    
For FREE analysis using ChatGPT web interface, use instead:
    python prepare_for_chatgpt.py --demo
    python parse_response.py

Usage:
    python run_checker.py --demo --skip-vlm    # Visualization only (free)
    python run_checker.py --demo               # Full analysis (needs API key)
    python run_checker.py --image img.png --segmentation seg.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.segmentation_loader import SegmentationLoader
from src.connectivity_prompts import ConnectivityPrompts
from src.connectivity_checker import ConnectivityChecker
from src.visualization import Visualizer
from src.utils import setup_sample_data, save_json, generate_report_id, print_result, ensure_dir


def get_vlm_interface(backend='openai', model='gpt-4o-mini'):
    """Lazily import VLM interface (only when needed)."""
    try:
        from src.api_optional.vlm_interface import VLMInterface
        return VLMInterface(backend=backend, model=model)
    except ImportError as e:
        print(f"❌ VLM interface not available: {e}")
        print("\n💡 For FREE analysis, use:")
        print("   python prepare_for_chatgpt.py --demo")
        print("   python parse_response.py")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="API-based Vessel Connectivity Checker",
        epilog="For FREE analysis, use prepare_for_chatgpt.py instead."
    )
    
    parser.add_argument('--image', '-i', type=str, help='Path to input image')
    parser.add_argument('--segmentation', '-s', type=str, help='Path to vessel segmentation')
    parser.add_argument('--mask', '-m', type=str, help='(Deprecated) Alias for --segmentation')
    parser.add_argument('--output', '-o', type=str, default='output', help='Output directory')
    parser.add_argument('--demo', action='store_true', help='Use synthetic sample data')
    parser.add_argument('--skip-vlm', action='store_true', help='Skip VLM (visualization only)')
    parser.add_argument('--prompt', '-p', type=str, default='general_continuity',
                        choices=ConnectivityPrompts.list_prompts())
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--verbose', '-v', action='store_true')
    
    args = parser.parse_args()
    
    # Handle deprecated --mask argument
    segmentation_arg = args.segmentation or args.mask
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║     Vessel Connectivity Checker (API Version)                 ║
╚═══════════════════════════════════════════════════════════════╝
""")
    
    # Get image paths
    if args.demo:
        print("📁 Creating sample data...")
        samples = setup_sample_data()
        image_path = samples['image']
        segmentation_path = samples['segmentation_broken']
    elif args.image and segmentation_arg:
        image_path = Path(args.image)
        segmentation_path = Path(segmentation_arg)
        if not image_path.exists() or not segmentation_path.exists():
            print("❌ Image or segmentation file not found")
            sys.exit(1)
    else:
        print("❌ Use --demo or provide --image and --segmentation")
        print("\n💡 For FREE analysis without API:")
        print("   python prepare_for_chatgpt.py --demo")
        sys.exit(1)
    
    print(f"📷 Image: {image_path}")
    print(f"🔬 Segmentation: {segmentation_path}")
    
    # Load and visualize
    loader = SegmentationLoader()
    image, segmentation = loader.load_pair(image_path, segmentation_path)
    
    visualizer = Visualizer()
    output_dir = ensure_dir(args.output)
    
    # Save comparison image
    comparison = visualizer.create_comparison(image, segmentation)
    comparison_path = output_dir / "comparison.png"
    visualizer.save_image(comparison, comparison_path)
    print(f"💾 Saved: {comparison_path}")
    
    if args.skip_vlm:
        print("\n⏭️ Skipped VLM analysis (--skip-vlm)")
        print("\n💡 For actual analysis, either:")
        print("   - Remove --skip-vlm (requires API key)")
        print("   - Use: python prepare_for_chatgpt.py --demo")
    else:
        print("\n🤖 Running VLM analysis...")
        print("   ⚠️ This requires OPENAI_API_KEY")
        
        try:
            vlm = get_vlm_interface(model=args.model)
            checker = ConnectivityChecker(vlm)
            
            prompt = ConnectivityPrompts.get_prompt(args.prompt)
            result = checker.check_connectivity(image_path, segmentation_path, prompt.template)
            
            print_result(result.to_dict())
            
            # Save report
            report_path = output_dir / f"report_{generate_report_id()}.json"
            save_json(result.to_dict(), report_path)
            print(f"💾 Saved: {report_path}")
            
            if args.verbose and result.raw_response:
                print(f"\n📝 Raw response:\n{result.raw_response[:500]}...")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            print("\n💡 For FREE analysis, use:")
            print("   python prepare_for_chatgpt.py --demo")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
