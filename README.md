# 🔬 Vessel Connectivity Checker

A tool for evaluating vessel segmentation quality using Vision-Language Models.

**No API keys required!** Works with the free ChatGPT web interface.

## Quick Start (Manual Workflow)

```bash
# 1. Prepare your image for ChatGPT
python prepare_for_chatgpt.py --image your_image.png --mask your_mask.png

# 2. Go to chat.openai.com, upload the generated image, paste the prompt

# 3. Copy ChatGPT's response to responses/response.txt

# 4. Parse the response
python parse_response.py
```

That's it! You get structured analysis without paying for API access.

## Project Structure

```
vessel-connectivity-checker/
├── prepare_for_chatgpt.py   # Step 1: Create composite image + get prompt
├── parse_response.py        # Step 3: Parse ChatGPT's response
├── prompts/                  # Copy-paste prompts for ChatGPT
│   ├── 01_general_continuity.txt
│   ├── 02_broken_vessel_detection.txt
│   ├── 03_continuity_score.txt
│   ├── 04_bifurcation_analysis.txt
│   ├── 05_anatomical_sanity.txt
│   └── 06_segmentation_quality.txt
├── responses/                # Paste ChatGPT responses here
│   └── response.txt
├── output/                   # Generated images and reports
├── src/
│   ├── segmentation_loader.py  # Load images & masks
│   ├── connectivity_checker.py # Parse responses
│   ├── connectivity_prompts.py # Prompt definitions
│   ├── visualization.py        # Image overlays
│   └── api_optional/           # [Optional] API-based automation
│       └── vlm_interface.py    # OpenAI/LLaVA integration
├── notebooks/               # Jupyter notebooks
├── tests/                   # Unit tests
└── requirements.txt
```

## Detailed Workflow

### Step 1: Prepare Images

```bash
# With your own images
python prepare_for_chatgpt.py --image retina.png --mask vessels.png

# Or use demo data
python prepare_for_chatgpt.py --demo

# Choose a specific analysis prompt
python prepare_for_chatgpt.py --demo --prompt 02_broken_vessel_detection
```

This creates:
- `output/chatgpt_upload.png` - Composite image to upload to ChatGPT
- `output/prompt_to_copy.txt` - The prompt text to paste

### Step 2: Use ChatGPT Web Interface

1. Go to [chat.openai.com](https://chat.openai.com)
2. Click the 📎 attachment icon
3. Upload `output/chatgpt_upload.png`
4. Copy the prompt from terminal (or `output/prompt_to_copy.txt`)
5. Paste and send

### Step 3: Parse Response

1. Copy ChatGPT's response
2. Paste it into `responses/response.txt`
3. Run the parser:

```bash
python parse_response.py
```

You'll get structured output like:

```
==================================================
  VESSEL CONNECTIVITY ANALYSIS RESULT
==================================================

  Status: ✗ DISCONTINUOUS
  Confidence: 85%
  Quality Score: 0.6/1.0

  Broken Segments (2):
    1. Central main vessel - gap in middle section
    2. Upper branch - disconnection near branch point

==================================================
```

## Available Prompts

| File | Purpose |
|------|---------|
| `01_general_continuity.txt` | Basic continuity check (start here!) |
| `02_broken_vessel_detection.txt` | Find and locate broken segments |
| `03_continuity_score.txt` | Get a numerical score |
| `04_bifurcation_analysis.txt` | Analyze branching points |
| `05_anatomical_sanity.txt` | Check anatomical plausibility |
| `06_segmentation_quality.txt` | Comprehensive quality assessment |

List available prompts:
```bash
python prepare_for_chatgpt.py --list-prompts
```

## Output Format

The parser generates JSON output:

```json
{
    "continuous": false,
    "broken_segments": [
        "upper-left branch - gap near bifurcation",
        "main artery midsection - disconnection"
    ],
    "confidence": 0.85,
    "quality_score": 0.6,
    "bifurcation_quality": "good",
    "anatomically_plausible": true
}
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# That's it! No API keys needed for manual workflow
```

## Using Real Data

### DRIVE Retinal Dataset

1. Download from [DRIVE Challenge](https://drive.grand-challenge.org/)
2. Run:
   ```bash
   python prepare_for_chatgpt.py \
       --image data/DRIVE/images/image.tif \
       --mask data/DRIVE/masks/mask.gif
   ```

---

## [Optional] API-Based Automation

If you DO want automated API access (requires OpenAI API key and costs money):

```python
# API code is in src/api_optional/
from src.api_optional.vlm_interface import VLMInterface

# Set your API key
import os
os.environ['OPENAI_API_KEY'] = 'sk-...'

# Use the API interface
vlm = VLMInterface(backend='openai', model='gpt-4o-mini')
response = vlm.ask(image_path, prompt, mask_path)
```

Or use the legacy CLI (also requires API key):
```bash
export OPENAI_API_KEY='sk-...'
python run_checker.py --image img.png --mask mask.png
```

---

## Why This Tool?

Medical image segmentation often produces topologically incorrect results—vessels may appear fragmented or disconnected. Traditional metrics (Dice, IoU) don't capture these structural errors.

This tool uses VLMs to:
- **Understand spatial relationships** in medical images
- **Apply domain knowledge** about vascular anatomy  
- **Provide interpretable feedback** in natural language
- **Generate structured data** for further analysis

## License

MIT License
