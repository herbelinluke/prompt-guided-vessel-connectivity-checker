"""
Connectivity Checker - Parse VLM outputs and generate structured connectivity reports.
"""

import re
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime


@dataclass
class ConnectivityResult:
    """Structured result from connectivity analysis."""
    continuous: bool
    broken_segments: List[str]
    confidence: float
    quality_score: Optional[float] = None
    bifurcation_quality: Optional[str] = None
    anatomically_plausible: Optional[bool] = None
    raw_response: Optional[str] = None
    analysis_type: str = "general"
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class ResponseParser:
    """
    Parse VLM text responses into structured data.
    Uses keyword patterns and regex to extract information.
    """
    
    # Keywords indicating continuity
    CONTINUOUS_KEYWORDS = [
        "continuous", "intact", "connected", "complete", "unbroken",
        "no breaks", "no gaps", "fully connected", "well-connected"
    ]
    
    # Keywords indicating discontinuity
    DISCONTINUOUS_KEYWORDS = [
        "discontinuous", "broken", "disconnected", "gap", "interrupted",
        "fragmented", "missing", "incomplete", "severed", "cut off"
    ]
    
    # Keywords for confidence/certainty
    HIGH_CONFIDENCE_KEYWORDS = [
        "clearly", "definitely", "certainly", "obviously", "absolutely",
        "without doubt", "evident", "apparent"
    ]
    
    LOW_CONFIDENCE_KEYWORDS = [
        "possibly", "might", "may", "uncertain", "unclear", "appears to",
        "seems", "potentially", "could be"
    ]
    
    @classmethod
    def extract_boolean(cls, text: str, positive_keywords: List[str], negative_keywords: List[str]) -> Optional[bool]:
        """
        Extract a boolean value based on keyword presence.
        """
        text_lower = text.lower()
        
        positive_count = sum(1 for kw in positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in text_lower)
        
        if positive_count > negative_count:
            return True
        elif negative_count > positive_count:
            return False
        return None
    
    @classmethod
    def extract_continuity(cls, text: str) -> bool:
        """Extract continuity status from VLM response."""
        patterns = [
            r"CONTINUITY:\s*(continuous|discontinuous)",
            r"continuity[:\s]+(is\s+)?(continuous|discontinuous|intact|broken)",
            r"vessels?\s+(are|appear)\s+(continuous|discontinuous|connected|disconnected)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(match.lastindex).lower()
                return value in ["continuous", "intact", "connected"]
        
        result = cls.extract_boolean(text, cls.CONTINUOUS_KEYWORDS, cls.DISCONTINUOUS_KEYWORDS)
        return result if result is not None else True
    
    @classmethod
    def extract_broken_segments(cls, text: str) -> List[str]:
        """Extract list of broken/disconnected segment locations."""
        segments = []
        
        # Pattern for numbered lists
        numbered_pattern = r"\d+\.\s*([^.\n]+(?:gap|break|disconnect|missing|broken)[^.\n]*)"
        segments.extend(re.findall(numbered_pattern, text, re.IGNORECASE))
        
        # Pattern for location descriptions
        location_patterns = [
            r"(?:break|gap|disconnect(?:ion)?|missing\s+segment)\s+(?:in|at|near)\s+(?:the\s+)?([^.,\n]+)",
            r"([^.,\n]*(?:upper|lower|left|right|central|peripheral)[^.,\n]*(?:break|gap|missing)[^.,\n]*)",
            r"BROKEN_SEGMENTS?:\s*\n((?:\d+\.\s*[^\n]+\n?)+)",
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, str) and len(match.strip()) > 0:
                    cleaned = match.strip()
                    if "\n" in cleaned:
                        items = [item.strip() for item in cleaned.split("\n") if item.strip()]
                        segments.extend(items)
                    else:
                        segments.append(cleaned)
        
        # Remove duplicates and clean up
        seen = set()
        unique_segments = []
        for seg in segments:
            seg = re.sub(r"^\d+\.\s*", "", seg).strip()
            if seg and seg.lower() not in seen:
                seen.add(seg.lower())
                unique_segments.append(seg)
        
        return unique_segments
    
    @classmethod
    def extract_confidence(cls, text: str) -> float:
        """Extract confidence score from VLM response."""
        patterns = [
            r"CONFIDENCE:\s*([0-9.]+)",
            r"confidence[:\s]+([0-9.]+)",
            r"([0-9.]+)\s*(?:confidence|certain)",
            r"(\d+)%\s*confiden",
            r"am\s+(\d+)%",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    if value > 1:
                        value = value / 100
                    return min(1.0, max(0.0, value))
                except ValueError:
                    continue
        
        text_lower = text.lower()
        high_count = sum(1 for kw in cls.HIGH_CONFIDENCE_KEYWORDS if kw in text_lower)
        low_count = sum(1 for kw in cls.LOW_CONFIDENCE_KEYWORDS if kw in text_lower)
        
        base_confidence = 0.7
        confidence = base_confidence + (high_count * 0.1) - (low_count * 0.15)
        return min(1.0, max(0.0, confidence))
    
    @classmethod
    def extract_quality_score(cls, text: str) -> Optional[float]:
        """Extract quality/continuity score from VLM response."""
        patterns = [
            r"QUALITY_SCORE:\s*([0-9.]+)",
            r"CONTINUITY_SCORE:\s*([0-9.]+)", 
            r"OVERALL_QUALITY:\s*([0-9.]+)",
            r"score[:\s]+([0-9.]+)\s*(?:out of\s*10|/10)?",
            r"([0-9.]+)\s*/\s*10",
            r"rating[:\s]+([0-9.]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    if value > 1:
                        value = value / 10
                    return min(1.0, max(0.0, value))
                except ValueError:
                    continue
        
        return None
    
    @classmethod
    def extract_bifurcation_quality(cls, text: str) -> Optional[str]:
        """Extract bifurcation quality assessment."""
        patterns = [
            r"BIFURCATION_QUALITY:\s*(good|fair|poor|excellent)",
            r"bifurcation(?:s)?\s+(?:are|appear|quality[:\s]+)\s*(good|fair|poor|excellent)",
            r"bifurcation\s+quality\s+(?:is\s+)?(good|fair|poor|excellent)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).lower()
        
        return None
    
    @classmethod
    def extract_anatomical_plausibility(cls, text: str) -> Optional[bool]:
        """Extract anatomical plausibility assessment."""
        patterns = [
            r"ANATOMICALLY_PLAUSIBLE:\s*(yes|no)",
            r"anatomically\s+(plausible|implausible|realistic|unrealistic)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).lower()
                return value in ["yes", "plausible", "realistic"]
        
        return None


class ConnectivityChecker:
    """
    Main class for checking vessel connectivity using VLM analysis.
    Orchestrates VLM queries and parses responses.
    """
    
    def __init__(self, vlm_interface=None):
        """
        Initialize the connectivity checker.
        
        Args:
            vlm_interface: VLMInterface instance for querying VLM
        """
        self.vlm = vlm_interface
        self.parser = ResponseParser()
    
    def analyze(
        self,
        vlm_response: str,
        analysis_type: str = "general"
    ) -> ConnectivityResult:
        """
        Parse a VLM response into a structured connectivity result.
        
        Args:
            vlm_response: Raw text response from VLM
            analysis_type: Type of analysis that was performed
            
        Returns:
            ConnectivityResult with parsed data
        """
        return ConnectivityResult(
            continuous=self.parser.extract_continuity(vlm_response),
            broken_segments=self.parser.extract_broken_segments(vlm_response),
            confidence=self.parser.extract_confidence(vlm_response),
            quality_score=self.parser.extract_quality_score(vlm_response),
            bifurcation_quality=self.parser.extract_bifurcation_quality(vlm_response),
            anatomically_plausible=self.parser.extract_anatomical_plausibility(vlm_response),
            raw_response=vlm_response,
            analysis_type=analysis_type,
            timestamp=datetime.now().isoformat()
        )
    
    def check_connectivity(
        self,
        image_path: Union[str, Path],
        segmentation_path: Union[str, Path],
        prompt: Optional[str] = None
    ) -> ConnectivityResult:
        """
        Full connectivity check: query VLM and parse response.
        
        Args:
            image_path: Path to the original image
            segmentation_path: Path to the vessel segmentation
            prompt: Custom prompt (uses default if not provided)
            
        Returns:
            ConnectivityResult with analysis
        """
        if self.vlm is None:
            raise ValueError("VLM interface not configured. Pass vlm_interface to constructor.")
        
        from .connectivity_prompts import ConnectivityPrompts
        
        if prompt is None:
            prompt = ConnectivityPrompts.GENERAL_CONTINUITY.template
        
        # Query VLM
        system_prompt = ConnectivityPrompts.get_system_prompt()
        response = self.vlm.ask(
            image_path, 
            prompt, 
            segmentation_path,
            system_prompt=system_prompt
        )
        
        # Parse and return result
        return self.analyze(response, analysis_type="general")
    
    def run_full_analysis(
        self,
        image_path: Union[str, Path],
        segmentation_path: Union[str, Path]
    ) -> Dict[str, ConnectivityResult]:
        """
        Run multiple analysis prompts and combine results.
        
        Args:
            image_path: Path to the original image
            segmentation_path: Path to the vessel segmentation
            
        Returns:
            Dictionary of analysis type -> ConnectivityResult
        """
        if self.vlm is None:
            raise ValueError("VLM interface not configured.")
        
        from .connectivity_prompts import ConnectivityPrompts
        
        results = {}
        prompts_to_run = [
            ("continuity", ConnectivityPrompts.GENERAL_CONTINUITY),
            ("broken_segments", ConnectivityPrompts.BROKEN_VESSEL_DETECTION),
            ("quality", ConnectivityPrompts.SEGMENTATION_QUALITY),
        ]
        
        system_prompt = ConnectivityPrompts.get_system_prompt()
        
        for analysis_type, prompt_template in prompts_to_run:
            response = self.vlm.ask(
                image_path,
                prompt_template.template,
                segmentation_path,
                system_prompt=system_prompt
            )
            results[analysis_type] = self.analyze(response, analysis_type=analysis_type)
        
        return results
    
    def generate_report(
        self,
        result: Union[ConnectivityResult, Dict[str, ConnectivityResult]],
        output_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive report from analysis results.
        
        Args:
            result: Single result or dict of results
            output_path: Optional path to save JSON report
            
        Returns:
            Report dictionary
        """
        if isinstance(result, ConnectivityResult):
            results = {"primary": result}
        else:
            results = result
        
        report = {
            "summary": {
                "overall_continuous": all(r.continuous for r in results.values()),
                "total_broken_segments": sum(len(r.broken_segments) for r in results.values()),
                "average_confidence": sum(r.confidence for r in results.values()) / len(results),
            },
            "analyses": {name: r.to_dict() for name, r in results.items()},
            "generated_at": datetime.now().isoformat(),
        }
        
        quality_scores = [r.quality_score for r in results.values() if r.quality_score is not None]
        if quality_scores:
            report["summary"]["average_quality_score"] = sum(quality_scores) / len(quality_scores)
        
        all_segments = []
        for r in results.values():
            all_segments.extend(r.broken_segments)
        report["summary"]["all_broken_segments"] = list(set(all_segments))
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report


def parse_vlm_response(response: str) -> ConnectivityResult:
    """
    Convenience function to parse a VLM response.
    
    Args:
        response: Raw VLM text response
        
    Returns:
        Parsed ConnectivityResult
    """
    checker = ConnectivityChecker()
    return checker.analyze(response)


def check_keywords(response: str) -> Dict[str, bool]:
    """
    Quick keyword check for connectivity indicators.
    
    Args:
        response: VLM text response
        
    Returns:
        Dict of keyword categories and their presence
    """
    response_lower = response.lower()
    
    return {
        "mentions_continuous": any(kw in response_lower for kw in ResponseParser.CONTINUOUS_KEYWORDS),
        "mentions_discontinuous": any(kw in response_lower for kw in ResponseParser.DISCONTINUOUS_KEYWORDS),
        "high_confidence_language": any(kw in response_lower for kw in ResponseParser.HIGH_CONFIDENCE_KEYWORDS),
        "low_confidence_language": any(kw in response_lower for kw in ResponseParser.LOW_CONFIDENCE_KEYWORDS),
        "mentions_bifurcation": any(kw in response_lower for kw in ["bifurcation", "branch", "fork", "junction"]),
        "mentions_location": any(kw in response_lower for kw in ["upper", "lower", "left", "right", "center", "peripheral"]),
    }
