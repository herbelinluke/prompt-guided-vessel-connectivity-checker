"""
Vessel Connectivity Checker - A VLM-based tool for evaluating vessel segmentation quality.

This tool is designed to work with the FREE ChatGPT web interface.
No API keys required for the default workflow!

MANUAL WORKFLOW (Default - No API needed):
    1. python prepare_for_chatgpt.py --image img.png --mask mask.png
    2. Upload image to ChatGPT, copy-paste the prompt
    3. Copy ChatGPT's response to responses/response.txt
    4. python parse_response.py

API WORKFLOW (Optional - Requires API key):
    See src/api_optional/ for VLMInterface if you want automated API access.
"""

from .segmentation_loader import SegmentationLoader
from .connectivity_prompts import ConnectivityPrompts
from .connectivity_checker import ConnectivityChecker
from .visualization import Visualizer

__version__ = "0.1.0"
__all__ = [
    "SegmentationLoader",
    "ConnectivityPrompts",
    "ConnectivityChecker",
    "Visualizer",
]

# Note: VLMInterface moved to src/api_optional/
# Import it explicitly if you want API-based analysis:
#   from src.api_optional.vlm_interface import VLMInterface
