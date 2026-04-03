"""Content_Keywords checker for MulDimIF constraints.

Ported from: https://github.com/Junjie-Ye/MulDimIF (Apache 2.0 License)
"""

import re
from typing import List


class Content_Keywords:
    """Check keyword-related constraints in response text."""

    def check(self, constraint: str, response: str) -> bool:
        """Dispatch to the appropriate keyword check based on constraint text."""
        constraint_lower = constraint.lower()

        if "include" in constraint_lower and "keyword" in constraint_lower:
            return self._check_include_keywords(constraint, response)
        elif "exclude" in constraint_lower and "keyword" in constraint_lower:
            return self._check_exclude_keywords(constraint, response)
        elif "frequency" in constraint_lower:
            return self._check_keyword_frequency(constraint, response)
        elif "at least" in constraint_lower:
            return self._check_include_keywords(constraint, response)
        elif "not" in constraint_lower and ("contain" in constraint_lower or "use" in constraint_lower):
            return self._check_exclude_keywords(constraint, response)
        elif "contain" in constraint_lower or "use" in constraint_lower or "mention" in constraint_lower:
            return self._check_include_keywords(constraint, response)
        else:
            return self._check_include_keywords(constraint, response)

    def _extract_keywords(self, constraint: str) -> List[str]:
        """Extract quoted keywords from constraint text."""
        keywords = re.findall(r'"([^"]+)"', constraint)
        if not keywords:
            keywords = re.findall(r"'([^']+)'", constraint)
        return keywords

    def _check_include_keywords(self, constraint: str, response: str) -> bool:
        """Check that all specified keywords appear in the response."""
        keywords = self._extract_keywords(constraint)
        if not keywords:
            return True
        response_lower = response.lower()
        return all(kw.lower() in response_lower for kw in keywords)

    def _check_exclude_keywords(self, constraint: str, response: str) -> bool:
        """Check that none of the specified keywords appear in the response."""
        keywords = self._extract_keywords(constraint)
        if not keywords:
            return True
        response_lower = response.lower()
        return all(kw.lower() not in response_lower for kw in keywords)

    def _check_keyword_frequency(self, constraint: str, response: str) -> bool:
        """Check keyword frequency constraints (e.g., 'at least 3 times')."""
        keywords = self._extract_keywords(constraint)
        if not keywords:
            return True

        # Extract required count
        count_match = re.search(r'(?:at least|exactly|at most|no more than)\s+(\d+)', constraint.lower())
        if not count_match:
            return self._check_include_keywords(constraint, response)

        required = int(count_match.group(1))
        response_lower = response.lower()
        constraint_lower = constraint.lower()

        for kw in keywords:
            actual = len(re.findall(re.escape(kw.lower()), response_lower))
            if "at least" in constraint_lower:
                if actual < required:
                    return False
            elif "exactly" in constraint_lower:
                if actual != required:
                    return False
            elif "at most" in constraint_lower or "no more than" in constraint_lower:
                if actual > required:
                    return False
        return True
