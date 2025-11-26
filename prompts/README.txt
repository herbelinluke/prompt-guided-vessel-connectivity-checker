================================================================================
                     PROMPT FILES FOR CHATGPT WEB INTERFACE
================================================================================

HOW TO USE THESE PROMPTS:

1. Run the preparation script:
   
   python prepare_for_chatgpt.py --demo
   
   Or with your own images:
   
   python prepare_for_chatgpt.py --image your_image.png --mask your_mask.png

2. The script will:
   - Create a composite image in output/chatgpt_upload.png
   - Let you interactively select a prompt (press 'l' to see descriptions)
   - Display the prompt text to copy

3. Go to ChatGPT (https://chat.openai.com)

4. Click the attachment icon and upload output/chatgpt_upload.png

5. Paste the prompt and send

6. Copy ChatGPT's response and save it to responses/response.txt

7. Run the parser:
   
   python parse_response.py

================================================================================
                              AVAILABLE PROMPTS
================================================================================

[1] 01_general_continuity.txt
    Quick yes/no check for vessel continuity with overall quality score
    START HERE for basic analysis

[2] 02_broken_vessel_detection.txt
    Find and list specific locations of all breaks with severity ratings
    Use when you need to know WHERE the problems are

[3] 03_continuity_score.txt
    Get a single 0-1 numerical score for comparing multiple segmentations
    Good for benchmarking different algorithms

[4] 04_bifurcation_analysis.txt
    Evaluate quality of vessel branching points (Y-shaped splits)
    Specialized check for branch point issues

[5] 05_anatomical_sanity.txt
    Check if segmentation looks biologically realistic (no artifacts)
    Catches weird loops, floating blobs, unnatural patterns

[6] 06_segmentation_quality.txt
    Comprehensive evaluation with 5 separate dimension scores
    Use for detailed final reports

================================================================================
