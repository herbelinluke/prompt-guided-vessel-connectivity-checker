"""
Connectivity Prompts - Prompt templates for vessel connectivity analysis.
Designed to mirror topological grammar concepts for medical VLM evaluation.
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
    Organized by analysis type and designed for VLM evaluation.
    """
    
    # System prompt for medical image analysis context
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

    # ==================== CONTINUITY PROMPTS ====================
    
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
        description="General continuity assessment of vessel segmentation",
        expected_keywords=["continuous", "discontinuous", "breaks", "gaps", "intact", "connected"]
    )
    
    BROKEN_VESSEL_DETECTION = PromptTemplate(
        name="broken_vessel_detection", 
        category="continuity",
        template="""Examine this vessel segmentation and identify any broken or disconnected segments.

Images provided:
1. Original vessel image
2. Binary segmentation mask
3. Overlay visualization

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
        description="Detect and localize broken vessel segments",
        expected_keywords=["broken", "disconnected", "gap", "missing", "interrupted", "fragmented"]
    )
    
    CONTINUITY_SCORE = PromptTemplate(
        name="continuity_score",
        category="continuity", 
        template="""Rate the continuity of this vessel segmentation on a scale of 0 to 1.

Consider:
- Are major vessels fully connected from origin to periphery?
- Are there any unexpected breaks in vessel paths?
- Do vessels maintain continuity through crossing points?

Respond with:
CONTINUITY_SCORE: [0.0 to 1.0]
CONFIDENCE: [0.0 to 1.0]
REASONING: [brief explanation]""",
        description="Numerical continuity scoring",
        expected_keywords=["score", "continuity", "connected", "complete"]
    )
    
    # ==================== BIFURCATION PROMPTS ====================
    
    BIFURCATION_ANALYSIS = PromptTemplate(
        name="bifurcation_analysis",
        category="bifurcation",
        template="""Analyze the vessel bifurcations (branching points) in this segmentation.

A good bifurcation in a vessel segmentation should:
- Show smooth transitions at branch points
- Maintain vessel continuity through the branch
- Have consistent vessel width relative to child branches

Examine the segmentation and report:
BIFURCATION_QUALITY: [good/fair/poor]
ISSUES_FOUND: [list any problems at branch points]
LOCATIONS: [where issues occur, if any]
ANATOMICAL_PLAUSIBILITY: [yes/no - do bifurcations look natural?]""",
        description="Evaluate quality of vessel bifurcations/branching",
        expected_keywords=["bifurcation", "branch", "split", "junction", "fork", "diverge"]
    )
    
    BIFURCATION_CONTINUITY = PromptTemplate(
        name="bifurcation_continuity",
        category="bifurcation",
        template="""Check if vessel continuity is maintained through all branching points.

At each visible bifurcation (where one vessel splits into two), verify:
1. The parent vessel connects to both child branches
2. There are no gaps at the junction point
3. The segmentation doesn't artificially merge or split vessels

Report:
BIFURCATIONS_EXAMINED: [count]
CONTINUOUS_BIFURCATIONS: [count]
PROBLEMATIC_BIFURCATIONS: [list locations and issues]
OVERALL_ASSESSMENT: [summary]""",
        description="Verify continuity through vessel bifurcations",
        expected_keywords=["bifurcation", "continuous", "junction", "connected", "branch"]
    )
    
    # ==================== ANATOMICAL PROMPTS ====================
    
    ANATOMICAL_SANITY = PromptTemplate(
        name="anatomical_sanity",
        category="anatomical",
        template="""Evaluate if this vessel segmentation is anatomically plausible.

Check for:
1. Does the vessel tree structure look natural for retinal/vascular anatomy?
2. Are vessels properly connected in a tree-like hierarchy?
3. Are there any floating segments not connected to the main tree?
4. Are there unrealistic loops or crossings?

Provide your assessment:
ANATOMICALLY_PLAUSIBLE: [yes/no]
STRUCTURE_TYPE: [tree-like/network/fragmented]
ISOLATED_SEGMENTS: [none/few/many]
ARTIFACTS_DETECTED: [list any obvious segmentation artifacts]
CONFIDENCE: [0.0 to 1.0]""",
        description="Check anatomical plausibility of vessel structure",
        expected_keywords=["anatomical", "plausible", "natural", "tree", "structure", "realistic"]
    )
    
    VESSEL_HIERARCHY = PromptTemplate(
        name="vessel_hierarchy",
        category="anatomical",
        template="""Analyze the hierarchical structure of the vessel tree.

In healthy vascular anatomy:
- Major vessels branch into smaller vessels
- Vessel diameter decreases towards periphery
- The structure follows a tree-like pattern

Assess this segmentation:
HIERARCHY_PRESERVED: [yes/partially/no]
MAJOR_VESSELS_PRESENT: [yes/no]
MINOR_VESSELS_CONNECTED: [yes/no]
PERIPHERAL_CONTINUITY: [good/fair/poor]
NOTES: [additional observations]""",
        description="Evaluate vessel tree hierarchy",
        expected_keywords=["hierarchy", "major", "minor", "peripheral", "tree", "branch"]
    )
    
    # ==================== QUALITY PROMPTS ====================
    
    SEGMENTATION_QUALITY = PromptTemplate(
        name="segmentation_quality",
        category="quality",
        template="""Provide an overall quality assessment of this vessel segmentation.

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
    
    COMPARISON_PROMPT = PromptTemplate(
        name="comparison",
        category="quality",
        template="""Compare the segmentation mask to the original image.

Looking at both the original vessel image and the segmentation:
1. Does the mask capture the main vessels visible in the image?
2. Are there visible vessels that are missing from the mask?
3. Are there mask regions that don't correspond to actual vessels?

Report:
COVERAGE: [complete/partial/poor]
MISSED_VESSELS: [none/few/many] - describe locations
FALSE_POSITIVES: [none/few/many] - describe locations
ALIGNMENT: [good/fair/poor]
RECOMMENDATIONS: [how could the segmentation be improved?]""",
        description="Compare segmentation to original image",
        expected_keywords=["match", "miss", "false", "align", "coverage"]
    )
    
    # ==================== HELPER METHODS ====================
    
    @classmethod
    def get_all_prompts(cls) -> Dict[str, PromptTemplate]:
        """Get all available prompts."""
        return {
            "general_continuity": cls.GENERAL_CONTINUITY,
            "broken_vessel_detection": cls.BROKEN_VESSEL_DETECTION,
            "continuity_score": cls.CONTINUITY_SCORE,
            "bifurcation_analysis": cls.BIFURCATION_ANALYSIS,
            "bifurcation_continuity": cls.BIFURCATION_CONTINUITY,
            "anatomical_sanity": cls.ANATOMICAL_SANITY,
            "vessel_hierarchy": cls.VESSEL_HIERARCHY,
            "segmentation_quality": cls.SEGMENTATION_QUALITY,
            "comparison": cls.COMPARISON_PROMPT,
        }
    
    @classmethod
    def get_by_category(cls, category: str) -> Dict[str, PromptTemplate]:
        """Get prompts by category."""
        all_prompts = cls.get_all_prompts()
        return {
            name: prompt 
            for name, prompt in all_prompts.items() 
            if prompt.category == category
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
    
    @classmethod
    def create_custom_prompt(
        cls,
        name: str,
        category: str,
        template: str,
        description: str,
        expected_keywords: Optional[List[str]] = None
    ) -> PromptTemplate:
        """Create a custom prompt template."""
        return PromptTemplate(
            name=name,
            category=category,
            template=template,
            description=description,
            expected_keywords=expected_keywords or []
        )


# Convenience functions for quick access
def get_continuity_prompt() -> str:
    """Get the general continuity check prompt."""
    return ConnectivityPrompts.GENERAL_CONTINUITY.template


def get_broken_segments_prompt() -> str:
    """Get the broken segments detection prompt."""
    return ConnectivityPrompts.BROKEN_VESSEL_DETECTION.template


def get_bifurcation_prompt() -> str:
    """Get the bifurcation analysis prompt."""
    return ConnectivityPrompts.BIFURCATION_ANALYSIS.template


def get_quality_prompt() -> str:
    """Get the overall quality assessment prompt."""
    return ConnectivityPrompts.SEGMENTATION_QUALITY.template

