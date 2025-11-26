"""
Optional API Integrations for Vessel Connectivity Checker.

This module contains code for programmatic VLM API access.
Only use this if you have API keys and want automated analysis.

For manual workflow (free, using ChatGPT web interface), 
use the main scripts instead:
  - prepare_for_chatgpt.py  (prepare images and prompts)
  - parse_response.py       (parse ChatGPT's response)

To use the API-based approach:
    from src.api_optional.vlm_interface import VLMInterface
    
    vlm = VLMInterface(backend='openai', model='gpt-4o-mini')
    response = vlm.ask(image, prompt, mask)
"""

# Only import if user explicitly needs it
# from .vlm_interface import VLMInterface

