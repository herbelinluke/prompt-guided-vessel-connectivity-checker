"""
Tests for connectivity prompts.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.connectivity_prompts import (
    ConnectivityPrompts,
    PromptTemplate,
    get_continuity_prompt,
    get_broken_segments_prompt,
    get_bifurcation_prompt,
    get_quality_prompt
)


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
        assert len(prompts) >= 5
        assert "general_continuity" in prompts
        assert "broken_vessel_detection" in prompts
    
    def test_get_by_category(self):
        """Test filtering prompts by category."""
        continuity_prompts = ConnectivityPrompts.get_by_category("continuity")
        
        assert len(continuity_prompts) >= 2
        for prompt in continuity_prompts.values():
            assert prompt.category == "continuity"
    
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
        assert "bifurcation_analysis" in names
    
    def test_get_system_prompt(self):
        """Test getting system prompt."""
        system_prompt = ConnectivityPrompts.get_system_prompt()
        
        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 0
        assert "medical" in system_prompt.lower() or "vessel" in system_prompt.lower()
    
    def test_create_custom_prompt(self):
        """Test creating custom prompt."""
        custom = ConnectivityPrompts.create_custom_prompt(
            name="custom_test",
            category="custom",
            template="Custom prompt for {task}.",
            description="A custom test prompt",
            expected_keywords=["custom", "test"]
        )
        
        assert custom.name == "custom_test"
        assert custom.category == "custom"
        assert "custom" in custom.expected_keywords


class TestPromptContent:
    """Test prompt content quality."""
    
    def test_continuity_prompt_has_instructions(self):
        """Test that continuity prompt has clear instructions."""
        prompt = ConnectivityPrompts.GENERAL_CONTINUITY
        
        # Should ask about continuity
        assert "continuous" in prompt.template.lower() or "continuity" in prompt.template.lower()
        # Should mention breaks or gaps
        assert "break" in prompt.template.lower() or "gap" in prompt.template.lower()
    
    def test_broken_detection_prompt_structure(self):
        """Test broken vessel detection prompt structure."""
        prompt = ConnectivityPrompts.BROKEN_VESSEL_DETECTION
        
        # Should ask about broken/disconnected segments
        assert "broken" in prompt.template.lower() or "disconnect" in prompt.template.lower()
        # Should ask for locations
        assert "location" in prompt.template.lower() or "where" in prompt.template.lower()
    
    def test_bifurcation_prompt_content(self):
        """Test bifurcation prompt content."""
        prompt = ConnectivityPrompts.BIFURCATION_ANALYSIS
        
        # Should mention bifurcation
        assert "bifurcation" in prompt.template.lower()
        # Should mention branching
        assert "branch" in prompt.template.lower()
    
    def test_anatomical_prompt_content(self):
        """Test anatomical sanity prompt content."""
        prompt = ConnectivityPrompts.ANATOMICAL_SANITY
        
        # Should mention anatomical
        assert "anatomic" in prompt.template.lower()
        # Should ask about plausibility
        assert "plausible" in prompt.template.lower() or "natural" in prompt.template.lower()
    
    def test_all_prompts_have_expected_keywords(self):
        """Test that all prompts have expected keywords defined."""
        prompts = ConnectivityPrompts.get_all_prompts()
        
        for name, prompt in prompts.items():
            assert hasattr(prompt, 'expected_keywords')
            assert isinstance(prompt.expected_keywords, list)


class TestConvenienceFunctions:
    """Test prompt convenience functions."""
    
    def test_get_continuity_prompt(self):
        """Test get_continuity_prompt function."""
        prompt = get_continuity_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
    
    def test_get_broken_segments_prompt(self):
        """Test get_broken_segments_prompt function."""
        prompt = get_broken_segments_prompt()
        assert isinstance(prompt, str)
        assert "broken" in prompt.lower() or "segment" in prompt.lower()
    
    def test_get_bifurcation_prompt(self):
        """Test get_bifurcation_prompt function."""
        prompt = get_bifurcation_prompt()
        assert isinstance(prompt, str)
        assert "bifurcation" in prompt.lower()
    
    def test_get_quality_prompt(self):
        """Test get_quality_prompt function."""
        prompt = get_quality_prompt()
        assert isinstance(prompt, str)
        assert "quality" in prompt.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

