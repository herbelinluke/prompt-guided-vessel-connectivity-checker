#!/usr/bin/env python3
"""
Parse ChatGPT Response.

Reads a response from responses/response.txt (or specified file)
and extracts structured connectivity data.

Usage:
    python parse_response.py
    python parse_response.py --file path/to/response.txt
    python parse_response.py --prompt 01_general_continuity
"""

import argparse
import sys
import re
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import save_json, ensure_dir, generate_report_id


class SmartResponseParser:
    """
    Smarter parser that extracts information more thoroughly from ChatGPT responses.
    """
    
    # Location keywords for detecting where breaks are mentioned
    LOCATION_PATTERNS = [
        r"upper[- ]?left",
        r"upper[- ]?right", 
        r"lower[- ]?left",
        r"lower[- ]?right",
        r"top[- ]?left",
        r"top[- ]?right",
        r"bottom[- ]?left",
        r"bottom[- ]?right",
        r"center|central|middle",
        r"peripheral|periphery",
        r"main\s+(?:artery|vessel|branch)",
        r"(?:upper|lower|left|right)\s+(?:branch|quadrant|region|area|section)",
        r"(?:major|minor)\s+(?:vessel|branch|artery)",
        r"midsection",
        r"near\s+(?:the\s+)?(?:center|edge|periphery|optic\s+disc)",
    ]
    
    # Keywords indicating a break/gap
    BREAK_KEYWORDS = [
        "break", "broken", "gap", "disconnect", "discontinu", "fragment",
        "missing", "absent", "interrupted", "severed", "cut", "incomplete"
    ]
    
    def __init__(self, prompt_type: str = "general"):
        self.prompt_type = prompt_type
    
    def extract_all_broken_segments(self, text: str) -> list:
        """
        Extract ALL mentions of broken segments from the entire response.
        This searches through the whole text, not just structured fields.
        """
        segments = []
        text_lower = text.lower()
        
        # Method 1: Look for numbered lists (1. xxx, 2. xxx)
        numbered_pattern = r"^\s*(\d+)[.)]\s*(.+)$"
        in_broken_section = False
        
        for line in text.split('\n'):
            line_stripped = line.strip()
            
            # Check if we're entering a broken segments section
            if any(kw in line_stripped.lower() for kw in ['broken_segments', 'broken segments', 'breaks found', 'disconnections']):
                in_broken_section = True
                continue
            
            # Check for numbered items
            match = re.match(numbered_pattern, line_stripped)
            if match:
                item_text = match.group(2).strip()
                # Only add if it looks like a break description
                if any(kw in item_text.lower() for kw in self.BREAK_KEYWORDS) or in_broken_section:
                    if len(item_text) > 5:  # Avoid very short items
                        segments.append(item_text)
            elif in_broken_section and line_stripped and not line_stripped.startswith(('TOTAL', 'SUMMARY', '=')):
                # Also capture non-numbered items in the broken section
                if line_stripped.startswith('-') or line_stripped.startswith('•'):
                    item = line_stripped.lstrip('-•').strip()
                    if len(item) > 5:
                        segments.append(item)
        
        # Method 2: Find sentences mentioning breaks with locations
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            # Check if sentence mentions a break
            has_break = any(kw in sentence.lower() for kw in self.BREAK_KEYWORDS)
            if not has_break:
                continue
            
            # Check if it mentions a location
            for loc_pattern in self.LOCATION_PATTERNS:
                match = re.search(loc_pattern, sentence, re.IGNORECASE)
                if match:
                    # Extract a meaningful segment description
                    # Try to get the clause containing the location and break
                    segment_desc = self._extract_segment_description(sentence, match.group())
                    if segment_desc and segment_desc not in segments:
                        segments.append(segment_desc)
                    break
        
        # Method 3: Look for explicit "DESCRIPTION:" field and parse it
        desc_match = re.search(r'DESCRIPTION:\s*(.+?)(?=\n[A-Z_]+:|$)', text, re.DOTALL | re.IGNORECASE)
        if desc_match:
            desc_text = desc_match.group(1)
            # Parse the description for break mentions
            desc_segments = self._parse_description_for_breaks(desc_text)
            for seg in desc_segments:
                if seg not in segments:
                    segments.append(seg)
        
        # Clean up and deduplicate
        segments = self._deduplicate_segments(segments)
        
        return segments
    
    def _extract_segment_description(self, sentence: str, location: str) -> str:
        """Extract a concise description of a broken segment from a sentence."""
        # Try to get the part of the sentence around the location mention
        sentence = sentence.strip()
        
        # If sentence is short enough, use it as-is
        if len(sentence) < 100:
            return sentence
        
        # Otherwise, try to extract just the relevant clause
        # Look for the location and surrounding context
        loc_idx = sentence.lower().find(location.lower())
        if loc_idx >= 0:
            # Get ~60 chars before and after the location
            start = max(0, loc_idx - 60)
            end = min(len(sentence), loc_idx + len(location) + 60)
            
            # Expand to word boundaries
            while start > 0 and sentence[start] not in ' ,.;':
                start -= 1
            while end < len(sentence) and sentence[end] not in ' ,.;':
                end += 1
            
            excerpt = sentence[start:end].strip(' ,.')
            if excerpt:
                return excerpt
        
        return sentence[:100] + "..." if len(sentence) > 100 else sentence
    
    def _parse_description_for_breaks(self, description: str) -> list:
        """Parse a DESCRIPTION field for break mentions."""
        segments = []
        
        # Split by common delimiters
        for delimiter in [';', ',', ' and ', ' also ']:
            if delimiter in description:
                parts = description.split(delimiter)
                for part in parts:
                    part = part.strip()
                    if any(kw in part.lower() for kw in self.BREAK_KEYWORDS):
                        for loc_pattern in self.LOCATION_PATTERNS:
                            if re.search(loc_pattern, part, re.IGNORECASE):
                                if len(part) > 10 and len(part) < 150:
                                    segments.append(part)
                                break
        
        return segments
    
    def _deduplicate_segments(self, segments: list) -> list:
        """Remove duplicate or overlapping segment descriptions."""
        if not segments:
            return []
        
        unique = []
        seen_normalized = set()
        
        for seg in segments:
            # Normalize for comparison
            normalized = re.sub(r'[^\w\s]', '', seg.lower())
            normalized = ' '.join(normalized.split())
            
            # Check for duplicates or near-duplicates
            is_duplicate = False
            for seen in seen_normalized:
                # If one is a substring of the other, skip
                if normalized in seen or seen in normalized:
                    is_duplicate = True
                    break
                # If they share most words, skip
                words1 = set(normalized.split())
                words2 = set(seen.split())
                if len(words1 & words2) > min(len(words1), len(words2)) * 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(seg)
                seen_normalized.add(normalized)
        
        return unique
    
    def extract_continuity(self, text: str) -> bool:
        """Extract continuity status."""
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
        
        # Count keywords as fallback
        text_lower = text.lower()
        continuous_count = sum(1 for kw in ["continuous", "intact", "connected", "no breaks", "no gaps"] if kw in text_lower)
        discontinuous_count = sum(1 for kw in self.BREAK_KEYWORDS if kw in text_lower)
        
        return continuous_count > discontinuous_count
    
    def extract_confidence(self, text: str) -> float:
        """Extract confidence score."""
        patterns = [
            r"CONFIDENCE:\s*([0-9.]+)",
            r"confidence[:\s]+([0-9.]+)",
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
        
        return 0.75  # Default
    
    def extract_quality_score(self, text: str) -> float:
        """Extract quality score."""
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
    
    def extract_bifurcation_quality(self, text: str) -> str:
        """Extract bifurcation quality."""
        patterns = [
            r"BIFURCATION_QUALITY:\s*(good|fair|poor|excellent)",
            r"bifurcation\s+quality\s+(?:is\s+)?(good|fair|poor|excellent)",
            r"bifurcation(?:s)?\s+(?:are|appear|look)\s+(good|fair|poor|excellent)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).lower()
        
        return None
    
    def extract_anatomical_plausibility(self, text: str) -> bool:
        """Extract anatomical plausibility."""
        patterns = [
            r"ANATOMICALLY_PLAUSIBLE:\s*(yes|no)",
            r"ANATOMICAL_PLAUSIBILITY:\s*(yes|no)",
            r"anatomically\s+(plausible|implausible|realistic|unrealistic)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).lower()
                return value in ["yes", "plausible", "realistic"]
        
        return None
    
    def extract_dimension_scores(self, text: str) -> dict:
        """Extract individual dimension scores (for prompt 06)."""
        dimensions = {}
        
        score_patterns = [
            ("completeness", r"COMPLETENESS:\s*([0-9.]+)"),
            ("accuracy", r"ACCURACY:\s*([0-9.]+)"),
            ("connectivity", r"CONNECTIVITY:\s*([0-9.]+)"),
            ("noise_level", r"NOISE_LEVEL:\s*([0-9.]+)"),
            ("gap_severity", r"GAP_SEVERITY:\s*([0-9.]+)"),
        ]
        
        for name, pattern in score_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    dimensions[name] = value / 10 if value > 1 else value
                except ValueError:
                    pass
        
        return dimensions if dimensions else None
    
    def extract_total_breaks(self, text: str) -> int:
        """Extract the total number of breaks mentioned."""
        patterns = [
            r"TOTAL_BREAKS:\s*(\d+)",
            r"(\d+)\s+(?:breaks?|gaps?|disconnection)",
            r"found\s+(\d+)\s+(?:breaks?|gaps?)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return None
    
    def parse(self, text: str) -> dict:
        """Parse the full response and return structured data."""
        result = {
            "continuous": self.extract_continuity(text),
            "broken_segments": self.extract_all_broken_segments(text),
            "confidence": self.extract_confidence(text),
            "quality_score": self.extract_quality_score(text),
            "bifurcation_quality": self.extract_bifurcation_quality(text),
            "anatomically_plausible": self.extract_anatomical_plausibility(text),
            "total_breaks": self.extract_total_breaks(text),
            "dimension_scores": self.extract_dimension_scores(text),
            "prompt_type": self.prompt_type,
            "raw_response": text,
            "parsed_at": datetime.now().isoformat(),
        }
        
        # If no broken segments found but marked discontinuous, add a note
        if not result["continuous"] and not result["broken_segments"]:
            result["broken_segments"] = ["(breaks mentioned but specific locations not parsed)"]
        
        # If total_breaks is set, validate against found segments
        if result["total_breaks"] and len(result["broken_segments"]) < result["total_breaks"]:
            result["_note"] = f"Found {len(result['broken_segments'])} segments but response mentions {result['total_breaks']} breaks"
        
        return result


def print_result(result: dict):
    """Print formatted result."""
    print("=" * 60)
    print("  VESSEL CONNECTIVITY ANALYSIS RESULT")
    print("=" * 60)
    
    status = "✓ CONTINUOUS" if result['continuous'] else "✗ DISCONTINUOUS"
    print(f"\n  Status: {status}")
    print(f"  Confidence: {result['confidence']:.0%}")
    
    if result['quality_score'] is not None:
        print(f"  Quality Score: {result['quality_score']:.1f}/1.0")
    
    if result['broken_segments']:
        print(f"\n  Broken Segments ({len(result['broken_segments'])}):")
        for i, seg in enumerate(result['broken_segments'], 1):
            # Wrap long segments
            if len(seg) > 55:
                print(f"    {i}. {seg[:55]}")
                print(f"       {seg[55:]}")
            else:
                print(f"    {i}. {seg}")
    
    if result['bifurcation_quality']:
        print(f"\n  Bifurcation Quality: {result['bifurcation_quality'].upper()}")
    
    if result['anatomically_plausible'] is not None:
        anat = "Yes" if result['anatomically_plausible'] else "No"
        print(f"  Anatomically Plausible: {anat}")
    
    if result['dimension_scores']:
        print("\n  Dimension Scores:")
        for dim, score in result['dimension_scores'].items():
            print(f"    {dim.replace('_', ' ').title()}: {score:.1f}/1.0")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Parse ChatGPT's response into structured data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--file', '-f', type=str, 
                        default='responses/response.txt',
                        help='Path to response file')
    parser.add_argument('--text', '-t', type=str,
                        help='Parse response text directly')
    parser.add_argument('--prompt', '-p', type=str,
                        help='Prompt type used (e.g., 01_general_continuity)')
    parser.add_argument('--output', '-o', type=str,
                        help='Save JSON output to file')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show raw response')
    
    args = parser.parse_args()
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║           VESSEL CONNECTIVITY CHECKER                         ║
║           Parse ChatGPT Response                              ║
╚═══════════════════════════════════════════════════════════════╝
""")
    
    # Try to auto-detect prompt type from last_prompt.txt
    prompt_type = args.prompt
    if not prompt_type:
        last_prompt_file = Path(__file__).parent / "output" / "last_prompt.txt"
        if last_prompt_file.exists():
            prompt_type = last_prompt_file.read_text().strip()
            print(f"📋 Auto-detected prompt type: {prompt_type}")
    
    if not prompt_type:
        prompt_type = "general"
    
    # Get response text
    if args.text:
        response_text = args.text
        source = "command line"
    else:
        response_file = Path(args.file)
        if not response_file.exists():
            print(f"❌ Response file not found: {response_file}")
            print("\nCreate this file and paste ChatGPT's response into it.")
            sys.exit(1)
        
        response_text = response_file.read_text()
        source = str(response_file)
        
        # Remove comment lines
        lines = [line for line in response_text.split('\n') if not line.strip().startswith('#')]
        response_text = '\n'.join(lines).strip()
        
        if not response_text:
            print(f"❌ Response file is empty: {response_file}")
            sys.exit(1)
    
    print(f"📄 Reading response from: {source}")
    print(f"   Length: {len(response_text)} characters")
    
    # Parse with the smarter parser
    print("\n🔍 Parsing response...")
    parser_instance = SmartResponseParser(prompt_type=prompt_type)
    result = parser_instance.parse(response_text)
    
    # Print formatted result
    print_result(result)
    
    # Show raw response if verbose
    if args.verbose:
        print("\n📝 Raw Response:")
        print("-" * 60)
        print(response_text[:800])
        if len(response_text) > 800:
            print(f"... ({len(response_text) - 800} more characters)")
        print("-" * 60)
    
    # Save output
    output_path = args.output
    if not output_path:
        output_dir = ensure_dir(Path(__file__).parent / "output")
        output_path = output_dir / f"result_{generate_report_id()}.json"
    
    # Remove raw_response from saved JSON (too large)
    save_result = {k: v for k, v in result.items() if k != 'raw_response'}
    save_result['source_file'] = source
    
    save_json(save_result, output_path)
    print(f"\n💾 Result saved to: {output_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    
    if result['continuous']:
        print("✅ Vessel segmentation appears CONTINUOUS")
    else:
        print("⚠️  Vessel segmentation has DISCONTINUITIES")
    
    if result['broken_segments'] and result['broken_segments'][0] != "(breaks mentioned but specific locations not parsed)":
        print(f"\n🔴 Found {len(result['broken_segments'])} broken segment(s)")
    
    if '_note' in result:
        print(f"\n📌 Note: {result['_note']}")


if __name__ == "__main__":
    main()
