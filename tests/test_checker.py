"""
Tests for connectivity checker and response parsing.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.connectivity_checker import (
    ConnectivityChecker,
    ConnectivityResult,
    ResponseParser,
    parse_vlm_response,
    check_keywords
)


class TestResponseParser:
    """Test cases for VLM response parsing."""
    
    def test_extract_continuity_positive(self):
        """Test extraction of positive continuity."""
        responses = [
            "The vessels appear continuous throughout the image.",
            "CONTINUITY: continuous",
            "The segmentation is intact with no visible breaks.",
            "Vessels are well-connected from origin to periphery."
        ]
        
        for response in responses:
            result = ResponseParser.extract_continuity(response)
            assert result is True, f"Failed for: {response}"
    
    def test_extract_continuity_negative(self):
        """Test extraction of negative continuity."""
        responses = [
            "The vessels are discontinuous in several areas.",
            "CONTINUITY: discontinuous",
            "There are multiple gaps and broken segments.",
            "The segmentation shows fragmented vessel structures."
        ]
        
        for response in responses:
            result = ResponseParser.extract_continuity(response)
            assert result is False, f"Failed for: {response}"
    
    def test_extract_broken_segments(self):
        """Test extraction of broken segment descriptions."""
        response = """
        BROKEN_SEGMENTS:
        1. Upper-left quadrant - minor vessel - moderate gap
        2. Central region - main artery - major disconnection
        
        There is also a break near the peripheral vessels.
        """
        
        segments = ResponseParser.extract_broken_segments(response)
        assert len(segments) >= 2
        assert any("upper" in s.lower() for s in segments)
    
    def test_extract_confidence_explicit(self):
        """Test extraction of explicit confidence values."""
        responses = [
            ("CONFIDENCE: 0.85", 0.85),
            ("confidence: 72", 0.72),
            ("I am 90% confident", 0.9),
        ]
        
        for response, expected in responses:
            result = ResponseParser.extract_confidence(response)
            assert abs(result - expected) < 0.1, f"Failed for: {response}"
    
    def test_extract_confidence_implicit(self):
        """Test confidence estimation from language."""
        high_conf = "This is clearly and definitely a continuous vessel."
        low_conf = "This might possibly be continuous, but I'm uncertain."
        
        high_result = ResponseParser.extract_confidence(high_conf)
        low_result = ResponseParser.extract_confidence(low_conf)
        
        assert high_result > low_result
    
    def test_extract_quality_score(self):
        """Test extraction of quality scores."""
        responses = [
            ("QUALITY_SCORE: 8", 0.8),
            ("OVERALL_QUALITY: 7.5", 0.75),
            ("I rate this 9/10", 0.9),
        ]
        
        for response, expected in responses:
            result = ResponseParser.extract_quality_score(response)
            assert result is not None
            assert abs(result - expected) < 0.15, f"Failed for: {response}"
    
    def test_extract_bifurcation_quality(self):
        """Test extraction of bifurcation quality."""
        responses = [
            ("BIFURCATION_QUALITY: good", "good"),
            ("The bifurcation quality is poor", "poor"),
            ("Bifurcations appear fair overall", "fair"),
        ]
        
        for response, expected in responses:
            result = ResponseParser.extract_bifurcation_quality(response)
            assert result == expected, f"Failed for: {response}"
    
    def test_extract_anatomical_plausibility(self):
        """Test extraction of anatomical plausibility."""
        responses = [
            ("ANATOMICALLY_PLAUSIBLE: yes", True),
            ("ANATOMICALLY_PLAUSIBLE: no", False),
            ("This is anatomically realistic", True),
        ]
        
        for response, expected in responses:
            result = ResponseParser.extract_anatomical_plausibility(response)
            assert result == expected, f"Failed for: {response}"


class TestConnectivityResult:
    """Test cases for ConnectivityResult dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ConnectivityResult(
            continuous=True,
            broken_segments=["segment1"],
            confidence=0.85
        )
        
        d = result.to_dict()
        assert d['continuous'] is True
        assert d['broken_segments'] == ["segment1"]
        assert d['confidence'] == 0.85
    
    def test_to_json(self):
        """Test conversion to JSON."""
        result = ConnectivityResult(
            continuous=False,
            broken_segments=[],
            confidence=0.5
        )
        
        json_str = result.to_json()
        assert '"continuous": false' in json_str
        assert '"confidence": 0.5' in json_str


class TestConnectivityChecker:
    """Test cases for ConnectivityChecker."""
    
    @pytest.fixture
    def checker(self):
        """Create a checker instance without VLM."""
        return ConnectivityChecker()
    
    def test_analyze_continuous_response(self, checker):
        """Test analysis of a continuous vessel response."""
        response = """
        CONTINUITY: continuous
        BREAKS_FOUND: no
        QUALITY_SCORE: 9
        CONFIDENCE: 0.92
        
        The vessel segmentation appears intact and continuous throughout.
        No significant breaks or gaps were detected in the vessel tree.
        """
        
        result = checker.analyze(response)
        
        assert result.continuous is True
        assert len(result.broken_segments) == 0
        assert result.confidence > 0.8
        assert result.quality_score is not None
    
    def test_analyze_discontinuous_response(self, checker):
        """Test analysis of a discontinuous vessel response."""
        response = """
        CONTINUITY: discontinuous
        BREAKS_FOUND: yes
        QUALITY_SCORE: 4
        CONFIDENCE: 0.75
        
        BROKEN_SEGMENTS:
        1. Upper-left branch - gap of approximately 5 pixels
        2. Main artery midsection - complete disconnection
        
        The segmentation shows significant fragmentation.
        """
        
        result = checker.analyze(response)
        
        assert result.continuous is False
        assert len(result.broken_segments) >= 1
        assert result.quality_score < 0.6
    
    def test_generate_report(self, checker):
        """Test report generation."""
        result = ConnectivityResult(
            continuous=True,
            broken_segments=[],
            confidence=0.9,
            quality_score=0.8
        )
        
        report = checker.generate_report(result)
        
        assert 'summary' in report
        assert 'analyses' in report
        assert report['summary']['overall_continuous'] is True


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_parse_vlm_response(self):
        """Test the parse_vlm_response function."""
        response = "The vessels are continuous with high confidence: 0.95"
        result = parse_vlm_response(response)
        
        assert isinstance(result, ConnectivityResult)
        assert result.continuous is True
    
    def test_check_keywords(self):
        """Test keyword checking function."""
        response = "The vessel shows a gap in the upper-left region with good bifurcation."
        
        keywords = check_keywords(response)
        
        assert keywords['mentions_discontinuous'] is True  # "gap"
        assert keywords['mentions_bifurcation'] is True
        assert keywords['mentions_location'] is True  # "upper", "left"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

