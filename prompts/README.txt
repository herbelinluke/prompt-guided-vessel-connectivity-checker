================================================================================
                     PROMPT FILES FOR CHATGPT WEB INTERFACE
================================================================================

HOW TO USE THESE PROMPTS:

1. Run the preparation script to create your composite image:
   
   python prepare_for_chatgpt.py --image your_image.png --mask your_mask.png

2. This will create a composite image in output/ and tell you which prompt to use.

3. Go to ChatGPT (https://chat.openai.com)

4. Click the attachment icon and upload the composite image

5. Open one of these prompt files:
   - 01_general_continuity.txt     (start here for basic analysis)
   - 02_broken_vessel_detection.txt (find specific broken segments)
   - 03_continuity_score.txt       (get a numerical score)
   - 04_bifurcation_analysis.txt   (check branching points)
   - 05_anatomical_sanity.txt      (verify anatomical plausibility)
   - 06_segmentation_quality.txt   (comprehensive quality check)

6. Copy the text below the dashed line and paste into ChatGPT

7. Copy ChatGPT's response and save it to responses/response.txt

8. Run the parser:
   
   python parse_response.py

This will extract structured data from ChatGPT's response!

================================================================================

