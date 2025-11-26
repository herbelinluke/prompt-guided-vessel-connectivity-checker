#!/usr/bin/env python3
"""
Prepare images for ChatGPT Web Interface.

This script creates a composite image and lets you interactively select a prompt.
No API keys needed - works with the free ChatGPT web interface!

Usage:
    python prepare_for_chatgpt.py --image path/to/image.png --mask path/to/mask.png
    python prepare_for_chatgpt.py --demo  # Use synthetic sample data
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.segmentation_loader import SegmentationLoader
from src.visualization import Visualizer
from src.utils import setup_sample_data, ensure_dir


def get_prompt_dir() -> Path:
    """Get the prompts directory."""
    return Path(__file__).parent / "prompts"


def load_prompt_info(prompt_file: Path) -> dict:
    """Load prompt metadata from a prompt file."""
    content = prompt_file.read_text()
    lines = content.split('\n')
    
    info = {
        'file': prompt_file,
        'name': prompt_file.stem,
        'title': '',
        'description': '',
        'content': ''
    }
    
    # Only look at the header section (before the dashed line)
    in_header = True
    for line in lines:
        if line.startswith('---'):
            in_header = False
            continue
        
        if in_header:
            if line.startswith('PROMPT:'):
                info['title'] = line.replace('PROMPT:', '').strip()
            elif line.startswith('DESCRIPTION:'):
                info['description'] = line.replace('DESCRIPTION:', '').strip()
    
    # Extract the copyable part (after the dashed line)
    if '-' * 10 in content:
        parts = content.split('-' * 80)
        if len(parts) > 1:
            info['content'] = parts[-1].strip()
        else:
            # Try shorter dashed line
            for i, line in enumerate(lines):
                if line.startswith('---'):
                    info['content'] = '\n'.join(lines[i+1:]).strip()
                    break
    
    if not info['content']:
        info['content'] = content
    
    return info


def get_all_prompts() -> list:
    """Get all available prompts with their info."""
    prompt_dir = get_prompt_dir()
    prompts = []
    
    for f in sorted(prompt_dir.glob("*.txt")):
        if f.stem.lower() != "readme":
            prompts.append(load_prompt_info(f))
    
    return prompts


def print_prompt_list(prompts: list):
    """Print a formatted list of available prompts."""
    print("\n" + "=" * 60)
    print("  AVAILABLE PROMPTS")
    print("=" * 60)
    
    for i, p in enumerate(prompts, 1):
        print(f"\n  [{i}] {p['title']}")
        print(f"      {p['description']}")
    
    print("\n" + "=" * 60)


def interactive_prompt_selection(prompts: list) -> dict:
    """Interactively let user select a prompt."""
    
    print("\n" + "=" * 60)
    print("  SELECT A PROMPT")
    print("=" * 60)
    print("\n  Commands:")
    print("    1-6    Select prompt by number")
    print("    l      List all prompts with descriptions")
    print("    q      Quit")
    print()
    
    # Show brief list
    for i, p in enumerate(prompts, 1):
        # Truncate description to fit on one line
        desc = p['description'][:45] + "..." if len(p['description']) > 45 else p['description']
        print(f"  [{i}] {p['title'][:25]:<25} - {desc}")
    
    print()
    
    while True:
        try:
            choice = input("  Enter choice (or 'l' for details): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Cancelled.")
            sys.exit(0)
        
        if choice == 'q' or choice == 'quit':
            print("\n  Cancelled.")
            sys.exit(0)
        
        if choice == 'l' or choice == 'list':
            print_prompt_list(prompts)
            continue
        
        try:
            idx = int(choice)
            if 1 <= idx <= len(prompts):
                selected = prompts[idx - 1]
                print(f"\n  ✓ Selected: {selected['title']}")
                return selected
            else:
                print(f"  ⚠ Please enter a number between 1 and {len(prompts)}")
        except ValueError:
            print("  ⚠ Invalid input. Enter a number, 'l' for list, or 'q' to quit.")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare images for ChatGPT Web Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Prepare an image-mask pair for ChatGPT
    python prepare_for_chatgpt.py --image retina.png --mask vessels.png
    
    # Use synthetic demo data
    python prepare_for_chatgpt.py --demo

Workflow:
    1. Run this script to create composite image
    2. Select a prompt interactively (or press 'l' to see descriptions)
    3. Upload the image to ChatGPT web interface
    4. Copy the prompt shown and paste it
    5. Copy ChatGPT's response to response.txt
    6. Run: python parse_response.py
        """
    )
    
    parser.add_argument('--image', '-i', type=str, help='Path to input image')
    parser.add_argument('--mask', '-m', type=str, help='Path to segmentation mask')
    parser.add_argument('--output', '-o', type=str, default='output', help='Output directory')
    parser.add_argument('--demo', action='store_true', help='Use synthetic sample data')
    parser.add_argument('--list-prompts', action='store_true', help='List available prompts and exit')
    
    args = parser.parse_args()
    
    # Load all prompts
    prompts = get_all_prompts()
    
    if not prompts:
        print("❌ No prompts found in prompts/ directory")
        sys.exit(1)
    
    if args.list_prompts:
        print_prompt_list(prompts)
        return
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║           VESSEL CONNECTIVITY CHECKER                         ║
║           Prepare for ChatGPT Web Interface                   ║
╚═══════════════════════════════════════════════════════════════╝
""")
    
    # Get image and mask paths
    if args.demo:
        print("📁 Creating synthetic sample data...")
        samples = setup_sample_data()
        image_path = samples['image']
        mask_path = samples['mask_broken']
        print(f"   Image: {image_path}")
        print(f"   Mask: {mask_path}")
    elif args.image and args.mask:
        image_path = Path(args.image)
        mask_path = Path(args.mask)
        if not image_path.exists():
            print(f"❌ Error: Image not found: {image_path}")
            sys.exit(1)
        if not mask_path.exists():
            print(f"❌ Error: Mask not found: {mask_path}")
            sys.exit(1)
    else:
        print("❌ Error: Provide --image and --mask, or use --demo")
        parser.print_help()
        sys.exit(1)
    
    # Load images
    print("\n📷 Loading images...")
    loader = SegmentationLoader()
    image, mask = loader.load_pair(image_path, mask_path, resize=True)
    print(f"   Image shape: {image.shape}")
    print(f"   Mask shape: {mask.shape}")
    
    # Create visualizations
    print("\n🎨 Creating composite image...")
    visualizer = Visualizer()
    output_dir = ensure_dir(args.output)
    
    # Create side-by-side comparison (this is what we'll upload to ChatGPT)
    comparison = visualizer.create_comparison(image, mask)
    comparison_path = output_dir / "chatgpt_upload.png"
    visualizer.save_image(comparison, comparison_path)
    
    # Also save individual images for reference
    overlay = visualizer.overlay_mask(image, mask)
    visualizer.save_image(overlay, output_dir / "overlay.png")
    
    print(f"\n✅ Composite image saved to:")
    print(f"   📁 {comparison_path.absolute()}")
    
    # Interactive prompt selection
    selected_prompt = interactive_prompt_selection(prompts)
    
    # Display the prompt
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║                    NEXT STEPS                                 ║
╚═══════════════════════════════════════════════════════════════╝

1. 🌐 Go to ChatGPT: https://chat.openai.com

2. 📎 Click the attachment icon and upload:
   {comparison_path.absolute()}

3. 📋 Copy and paste this prompt:

════════════════════════════════════════════════════════════════
{selected_prompt['content']}
════════════════════════════════════════════════════════════════

4. 💬 After ChatGPT responds, copy its response

5. 💾 Paste the response into: response.txt

6. 🔍 Run the parser:
   python parse_response.py --prompt {selected_prompt['name']}

""")
    
    # Save the prompt and selected prompt name for the parser
    prompt_output = output_dir / "prompt_to_copy.txt"
    prompt_output.write_text(selected_prompt['content'])
    
    # Save which prompt was selected (for the parser to use)
    prompt_meta = output_dir / "last_prompt.txt"
    prompt_meta.write_text(selected_prompt['name'])
    
    print(f"💡 Prompt saved to: {prompt_output}")
    print(f"💡 Prompt type saved for parser: {prompt_meta}")



if __name__ == "__main__":
    main()
