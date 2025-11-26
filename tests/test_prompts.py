"""
Tests for connectivity prompts (API workflow).

Note: These tests are for the API-based workflow.
For manual ChatGPT workflow, prompts are in prompts/*.txt files.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.connectivity_prompts import ConnectivityPrompts, PromptTemplate


class TestPromptTemplate:
    """Test cases for PromptTemplate."""
    
    def test_format(self):
        """Test template formatting."""
        template = PromptTemplate(
            name="test",
            category="test",
            template="Analyze the {region} for {feature}.",
            description="Test template",
            expected_keywords=["region", "feature"]
        )
        
        formatted = template.format(region="upper-left", feature="gaps")
        assert "upper-left" in formatted
        assert "gaps" in formatted


class TestConnectivityPrompts:
    """Test cases for ConnectivityPrompts class."""
    
    def test_get_all_prompts(self):
        """Test getting all prompts."""
        prompts = ConnectivityPrompts.get_all_prompts()
        
        assert isinstance(prompts, dict)
        assert len(prompts) >= 2
        assert "general_continuity" in prompts
        assert "broken_vessel_detection" in prompts
    
    def test_get_prompt(self):
        """Test getting specific prompt."""
        prompt = ConnectivityPrompts.get_prompt("general_continuity")
        
        assert isinstance(prompt, PromptTemplate)
        assert prompt.name == "general_continuity"
        assert len(prompt.template) > 0
    
    def test_get_prompt_invalid(self):
        """Test error on invalid prompt name."""
        with pytest.raises(ValueError):
            ConnectivityPrompts.get_prompt("nonexistent_prompt")
    
    def test_list_prompts(self):
        """Test listing prompt names."""
        names = ConnectivityPrompts.list_prompts()
        
        assert isinstance(names, list)
        assert "general_continuity" in names
    
    def test_get_system_prompt(self):
        """Test getting system prompt."""
        system_prompt = ConnectivityPrompts.get_system_prompt()
        
        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 0
        assert "medical" in system_prompt.lower() or "vessel" in system_prompt.lower()


class TestPromptContent:
    """Test prompt content quality."""
    
    def test_continuity_prompt_has_instructions(self):
        """Test that continuity prompt has clear instructions."""
        prompt = ConnectivityPrompts.GENERAL_CONTINUITY
        
        assert "continuous" in prompt.template.lower() or "continuity" in prompt.template.lower()
        assert "break" in prompt.template.lower() or "gap" in prompt.template.lower()
    
    def test_broken_detection_prompt_structure(self):
        """Test broken vessel detection prompt structure."""
        prompt = ConnectivityPrompts.BROKEN_VESSEL_DETECTION
        
        assert "broken" in prompt.template.lower() or "disconnect" in prompt.template.lower()
        assert "location" in prompt.template.lower() or "where" in prompt.template.lower()
    
    def test_quality_prompt_has_scores(self):
        """Test quality prompt asks for scores."""
        prompt = ConnectivityPrompts.SEGMENTATION_QUALITY
        
        assert "score" in prompt.template.lower() or "1-10" in prompt.template
    
    def test_all_prompts_have_expected_keywords(self):
        """Test that all prompts have expected keywords defined."""
        prompts = ConnectivityPrompts.get_all_prompts()
        
        for name, prompt in prompts.items():
            assert hasattr(prompt, 'expected_keywords')
            assert isinstance(prompt.expected_keywords, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
