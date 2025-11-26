"""
Connectivity Prompts - Prompt templates for vessel connectivity analysis.

NOTE: This module is primarily used by the API-based workflow (run_checker.py).
For the manual ChatGPT workflow, use the text files in prompts/ directory instead.

The text files are easier to copy-paste and can be edited without code changes.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class PromptTemplate:
    """A structured prompt template with metadata."""
    name: str
    category: str
    template: str
    description: str
    expected_keywords: List[str]
    
    def format(self, **kwargs) -> str:
        """Format the template with provided values."""
        return self.template.format(**kwargs)


class ConnectivityPrompts:
    """
    Library of prompts for vessel connectivity analysis.
    
    NOTE: These prompts are used by the API-based workflow.
    For manual ChatGPT workflow, see prompts/*.txt files.
    """
    
    # System prompt for API-based VLM context
    SYSTEM_PROMPT = """You are an expert medical image analyst specializing in vascular imaging 
and vessel segmentation quality assessment. You analyze retinal fundus images, angiograms, 
and vessel segmentation masks with precision.

When analyzing vessel segmentations, you should:
1. Evaluate the continuity and connectivity of vessel structures
2. Identify any breaks, gaps, or discontinuities in the vessel tree
3. Assess the quality of bifurcations (vessel branching points)
4. Note any artifacts or segmentation errors
5. Provide specific spatial locations when describing issues

Always structure your responses clearly and be specific about locations using 
anatomical or spatial references (e.g., "upper-left quadrant", "near the optic disc", 
"main arterial branch", "peripheral vessels")."""

    GENERAL_CONTINUITY = PromptTemplate(
        name="general_continuity",
        category="continuity",
        template="""Analyze this vessel segmentation for connectivity.

You are shown:
1. The original retinal/vessel image
2. The binary segmentation mask (white = vessels, black = background)
3. An overlay showing the segmentation on the original image

Please assess:
- Does the vessel segmentation appear continuous throughout?
- Are there any visible breaks or gaps in the vessel structures?
- Rate the overall connectivity quality from 1-10

Provide your analysis in this format:
CONTINUITY: [continuous/discontinuous]
BREAKS_FOUND: [yes/no]
QUALITY_SCORE: [1-10]
DESCRIPTION: [detailed analysis]""",
        description="Quick yes/no check for vessel continuity with overall quality score",
        expected_keywords=["continuous", "discontinuous", "breaks", "gaps", "intact", "connected"]
    )
    
    BROKEN_VESSEL_DETECTION = PromptTemplate(
        name="broken_vessel_detection", 
        category="continuity",
        template="""Examine this vessel segmentation and identify any broken or disconnected segments.

You are shown:
1. The original retinal/vessel image
2. The binary segmentation mask (white = vessels, black = background)
3. An overlay showing the segmentation on the original image

For each broken segment you find, describe:
- The location (e.g., "upper-left quadrant", "near center")
- The type of vessel (major artery, minor vessel, capillary)
- The severity (minor gap, major disconnection)

Format your response as:
BROKEN_SEGMENTS:
1. [location] - [vessel type] - [severity]
2. [location] - [vessel type] - [severity]
...

TOTAL_BREAKS: [number]
SUMMARY: [brief overall assessment]""",
        description="Find and list specific locations of all breaks with severity ratings",
        expected_keywords=["broken", "disconnected", "gap", "missing", "interrupted", "fragmented"]
    )
    
    SEGMENTATION_QUALITY = PromptTemplate(
        name="segmentation_quality",
        category="quality",
        template="""Provide an overall quality assessment of this vessel segmentation.

You are shown:
1. The original retinal/vessel image
2. The binary segmentation mask (white = vessels, black = background)
3. An overlay showing the segmentation on the original image

Evaluate:
1. Completeness - Are all visible vessels captured?
2. Accuracy - Does the mask align with actual vessel boundaries?
3. Connectivity - Are vessels properly connected?
4. Noise - Are there false positive segments?
5. Gaps - Are there missing vessel sections?

Score each from 1-10:
COMPLETENESS: [1-10]
ACCURACY: [1-10]
CONNECTIVITY: [1-10]
NOISE_LEVEL: [1-10, where 10 = no noise]
GAP_SEVERITY: [1-10, where 10 = no gaps]

OVERALL_QUALITY: [1-10]
SUMMARY: [brief explanation]""",
        description="Comprehensive segmentation quality assessment",
        expected_keywords=["quality", "complete", "accurate", "connected", "noise", "gaps"]
    )
    
    @classmethod
    def get_all_prompts(cls) -> Dict[str, PromptTemplate]:
        """Get all available prompts."""
        return {
            "general_continuity": cls.GENERAL_CONTINUITY,
            "broken_vessel_detection": cls.BROKEN_VESSEL_DETECTION,
            "segmentation_quality": cls.SEGMENTATION_QUALITY,
        }
    
    @classmethod
    def get_prompt(cls, name: str) -> PromptTemplate:
        """Get a specific prompt by name."""
        prompts = cls.get_all_prompts()
        if name not in prompts:
            raise ValueError(f"Unknown prompt: {name}. Available: {list(prompts.keys())}")
        return prompts[name]
    
    @classmethod
    def list_prompts(cls) -> List[str]:
        """List all available prompt names."""
        return list(cls.get_all_prompts().keys())
    
    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the system prompt for VLM context."""
        return cls.SYSTEM_PROMPT
