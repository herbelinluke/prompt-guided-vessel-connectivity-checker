# 🔬 Vessel Connectivity Checker

Evaluate vessel segmentation quality using Vision-Language Models or LLMs.

**No API keys required!** Uses free ChatGPT web interface.

NOTE: Cursor and Opus 4.5 used heaviliy in the creation of this repository. Also only the web-based workflow has been manually run and tested so the api workflow may have issues in it.

## Quick Start

```bash
# 1. Prepare image for ChatGPT
python prepare_for_chatgpt.py --demo

# 2. Upload output/chatgpt_upload.png to chat.openai.com
# 3. Paste the prompt, get response

# 4. Paste response to response.txt (in root directory)

# 5. Parse the response
python parse_response.py
```

## Using Your Own Images

```bash
python prepare_for_chatgpt.py --image retina.png --segmentation vessels.png
```

## Project Structure

```
vessel-connectivity-checker/
├── prepare_for_chatgpt.py   # Create composite image + select prompt
├── parse_response.py        # Parse ChatGPT's response to JSON
├── response.txt             # ← Paste ChatGPT response here
├── compare_metrics.py       # compare ChatGPT Topology analysis with Skeletonize calculations
├── prompts/                 # Copy-paste prompts for ChatGPT
│   ├── 01_general_continuity.txt
│   ├── 02_broken_vessel_detection.txt
│   ├── 03_continuity_score.txt
│   ├── 04_bifurcation_analysis.txt
│   ├── 05_anatomical_sanity.txt
│   ├── 06_segmentation_quality.txt
    └── 07_topology_metrics.txt
├── responses/               # Archived responses (saved when generating json reports)
│   └── response_report_<...>.txt
├── output/                  # Generated images and reports
│   ├── chatgpt_upload.png
│   └── result_<id>.json
├── src/
│   ├── segmentation_loader.py
│   ├── connectivity_checker.py
│   ├── connectivity_prompts.py
│   ├── skeleton_metrics.py
│   ├── visualization.py
│   └── api_optional/        # [Optional] API automation
│       └── vlm_interface.py
├── tests/
├── run_checker.py           # [Optional] API-based CLI
└── requirements.txt
```

## Available Prompts

| # | Prompt | Description |
|---|--------|-------------|
| 1 | General Continuity | Quick yes/no check with quality score |
| 2 | Broken Vessel Detection | Find specific break locations |
| 3 | Continuity Score | Numerical 0-1 score for comparison |
| 4 | Bifurcation Analysis | Check branching point quality |
| 5 | Anatomical Sanity | Verify biological realism |
| 6 | Segmentation Quality | Comprehensive 5-dimension scores |
| 7 | Topology Metrics | Estimate metrics for comparison with computed values |

## Output Format

The JSON result references the archived response:

```json
{
    "continuous": false,
    "broken_segments": [
        "Gap in upper-left branch",
        "Break in main artery midsection"
    ],
    "confidence": 0.85,
    "quality_score": 0.6,
    "response_file": "responses/response_report_20251126_142204_abc123.txt"
}
```

## Skeleton Metrics Comparison

Compare LLM topology estimates with skeletonized computed values:

```bash
# 1. Compute skeleton metrics (no VLM needed)
python compare_metrics.py --demo

# 2. Get LLM topology estimates
python prepare_for_chatgpt.py --prompt 07
# Upload image, paste prompt, get response, save to response.txt

# 3. Parse and compare
python parse_response.py
python compare_metrics.py --segmentation data/21_manual1.png --response output/result_xxx.json
```

This computes skeleton-based topology metrics (branch points, connected components, etc.)
and compares them with LLM estimates to validate how well LLMs understand a vessel topology segmentation.

## Installation

```bash
pip install -r requirements.txt
```

Core: numpy, Pillow, matplotlib
Skeleton metrics: scikit-image (optional, only for compare_metrics.py)

## [Optional] API Automation

If you have an OpenAI API key:

```bash
export OPENAI_API_KEY='sk-...'
python run_checker.py --demo
```
