"""Format_Json checker for MulDimIF constraints.

Ported from: https://github.com/Junjie-Ye/MulDimIF (Apache 2.0 License)
"""

import json
import re


class Format_Json:
    """Check JSON format constraints in response text."""

    def check(self, constraint: str, response: str) -> bool:
        """Check JSON-related constraints."""
        constraint_lower = constraint.lower()

        if "depth" in constraint_lower or "nest" in constraint_lower or "level" in constraint_lower:
            return self._check_json_depth(constraint, response)
        elif "valid" in constraint_lower or "format" in constraint_lower:
            return self._check_valid_json(response)
        else:
            return self._check_valid_json(response)

    def _extract_json(self, text: str) -> str:
        """Extract JSON string from text (handles code blocks)."""
        # Try to find JSON in code blocks first
        match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text)
        if match:
            return match.group(1).strip()
        # Try to find raw JSON
        for start_char in ['{', '[']:
            idx = text.find(start_char)
            if idx != -1:
                return text[idx:].strip()
        return text.strip()

    def _check_valid_json(self, response: str) -> bool:
        """Check if response contains valid JSON."""
        json_str = self._extract_json(response)
        try:
            json.loads(json_str)
            return True
        except (json.JSONDecodeError, ValueError):
            return False

    def _get_depth(self, obj: object, current: int = 1) -> int:
        """Get the nesting depth of a JSON object."""
        if isinstance(obj, dict):
            if not obj:
                return current
            return max(self._get_depth(v, current + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current
            return max(self._get_depth(v, current + 1) for v in obj)
        return current

    def _check_json_depth(self, constraint: str, response: str) -> bool:
        """Check JSON nesting depth constraints."""
        json_str = self._extract_json(response)
        try:
            obj = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            return False

        depth = self._get_depth(obj)

        # Extract required depth
        count_match = re.search(r'(\d+)', constraint)
        if not count_match:
            return True
        required = int(count_match.group(1))

        constraint_lower = constraint.lower()
        if "at least" in constraint_lower:
            return depth >= required
        elif "at most" in constraint_lower or "no more" in constraint_lower:
            return depth <= required
        elif "exactly" in constraint_lower:
            return depth == required
        else:
            return depth >= required
