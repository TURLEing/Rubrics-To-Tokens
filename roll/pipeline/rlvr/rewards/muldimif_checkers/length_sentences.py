"""Length_Sentences checker for MulDimIF constraints.

Ported from: https://github.com/Junjie-Ye/MulDimIF (Apache 2.0 License)
"""

import re


class Length_Sentences:
    """Check sentence count constraints in response text."""

    def check(self, constraint: str, response: str) -> bool:
        """Check sentence count constraints."""
        sentences = self._count_sentences(response)
        constraint_lower = constraint.lower()

        # Extract required count
        count_match = re.search(r'(\d+)', constraint)
        if not count_match:
            return True

        required = int(count_match.group(1))

        if "at least" in constraint_lower or "more than" in constraint_lower or "no less" in constraint_lower:
            return sentences >= required
        elif "at most" in constraint_lower or "no more" in constraint_lower or "fewer than" in constraint_lower:
            return sentences <= required
        elif "exactly" in constraint_lower:
            return sentences == required
        elif "between" in constraint_lower:
            range_match = re.search(r'between\s+(\d+)\s+and\s+(\d+)', constraint_lower)
            if range_match:
                low = int(range_match.group(1))
                high = int(range_match.group(2))
                return low <= sentences <= high
            return sentences >= required
        else:
            return sentences >= required

    def _count_sentences(self, text: str) -> int:
        """Count sentences in text."""
        # Split on sentence-ending punctuation followed by space or end of string
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return len([s for s in sentences if s.strip()])
