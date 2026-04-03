"""Length_Paragraphs checker for MulDimIF constraints.

Ported from: https://github.com/Junjie-Ye/MulDimIF (Apache 2.0 License)
"""

import re


class Length_Paragraphs:
    """Check paragraph count constraints in response text."""

    def check(self, constraint: str, response: str) -> bool:
        """Check paragraph count constraints."""
        paragraphs = self._count_paragraphs(response)
        constraint_lower = constraint.lower()

        # Extract required count
        count_match = re.search(r'(\d+)', constraint)
        if not count_match:
            return True

        required = int(count_match.group(1))

        if "at least" in constraint_lower or "more than" in constraint_lower or "no less" in constraint_lower:
            return paragraphs >= required
        elif "at most" in constraint_lower or "no more" in constraint_lower or "fewer than" in constraint_lower:
            return paragraphs <= required
        elif "exactly" in constraint_lower:
            return paragraphs == required
        elif "between" in constraint_lower:
            # Try to extract range: "between X and Y"
            range_match = re.search(r'between\s+(\d+)\s+and\s+(\d+)', constraint_lower)
            if range_match:
                low = int(range_match.group(1))
                high = int(range_match.group(2))
                return low <= paragraphs <= high
            return paragraphs >= required
        else:
            return paragraphs >= required

    def _count_paragraphs(self, text: str) -> int:
        """Count paragraphs in text (separated by blank lines)."""
        # Split by double newline
        parts = re.split(r'\n\s*\n', text.strip())
        # Filter out empty parts
        paragraphs = [p.strip() for p in parts if p.strip()]
        return len(paragraphs)
