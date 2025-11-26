"""
Vessel Connectivity Checker - Evaluate vessel segmentation quality using VLMs.

TWO WORKFLOWS AVAILABLE:

1. MANUAL (FREE) - Uses ChatGPT web interface:
   python prepare_for_chatgpt.py --demo
   # Upload image to ChatGPT, copy response
   python parse_response.py

2. API (PAID) - Uses OpenAI API:
   export OPENAI_API_KEY='sk-...'
   python run_checker.py --demo
"""

from .segmentation_loader import SegmentationLoader
from .connectivity_checker import ConnectivityChecker
from .visualization import Visualizer

__version__ = "0.1.0"
__all__ = [
    "SegmentationLoader",
    "ConnectivityChecker",
    "Visualizer",
]
