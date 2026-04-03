"""Length_Words checker for MulDimIF constraints.

Ported from: https://github.com/Junjie-Ye/MulDimIF (Apache 2.0 License)
"""

import re


class Length_Words:
    """Check word count constraints in response text."""

    def check(self, constraint: str, response: str) -> bool:
        """Check word count constraints."""
        constraint_lower = constraint.lower()

        # Check for per-element constraints (e.g., "each sentence should have X words")
        if "each" in constraint_lower or "every" in constraint_lower:
            return self._check_per_element(constraint, response)

        word_count = len(response.split())

        # Extract required count
        count_match = re.search(r'(\d+)', constraint)
        if not count_match:
            return True

        required = int(count_match.group(1))

        if "at least" in constraint_lower or "more than" in constraint_lower or "no less" in constraint_lower:
            return word_count >= required
        elif "at most" in constraint_lower or "no more" in constraint_lower or "fewer than" in constraint_lower:
            return word_count <= required
        elif "exactly" in constraint_lower:
            return word_count == required
        elif "between" in constraint_lower:
            range_match = re.search(r'between\s+(\d+)\s+and\s+(\d+)', constraint_lower)
            if range_match:
                low = int(range_match.group(1))
                high = int(range_match.group(2))
                return low <= word_count <= high
            return word_count >= required
        else:
            return word_count >= required

    def _check_per_element(self, constraint: str, response: str) -> bool:
        """Check word count per element (sentence, paragraph, bullet point)."""
        constraint_lower = constraint.lower()
        count_match = re.search(r'(\d+)', constraint)
        if not count_match:
            return True
        required = int(count_match.group(1))

        if "sentence" in constraint_lower:
            elements = re.split(r'(?<=[.!?])\s+', response.strip())
        elif "paragraph" in constraint_lower:
            elements = re.split(r'\n\s*\n', response.strip())
        elif "bullet" in constraint_lower or "item" in constraint_lower or "point" in constraint_lower:
            elements = re.findall(r'^\s*[-*+]\s+(.+)', response, re.MULTILINE)
            if not elements:
                elements = re.findall(r'^\s*\d+[.)]\s+(.+)', response, re.MULTILINE)
        else:
            elements = re.split(r'(?<=[.!?])\s+', response.strip())

        elements = [e.strip() for e in elements if e.strip()]
        if not elements:
            return False

        for elem in elements:
            wc = len(elem.split())
            if "at least" in constraint_lower or "more than" in constraint_lower:
                if wc < required:
                    return False
            elif "at most" in constraint_lower or "no more" in constraint_lower:
                if wc > required:
                    return False
            elif "exactly" in constraint_lower:
                if wc != required:
                    return False
        return True
