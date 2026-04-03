"""Language_English checker for MulDimIF constraints.

Ported from: https://github.com/Junjie-Ye/MulDimIF (Apache 2.0 License)
"""

import re


class Language_English:
    """Check English language constraints (case, etc.) in response text."""

    def check(self, constraint: str, response: str) -> bool:
        """Dispatch to the appropriate language check."""
        constraint_lower = constraint.lower()

        if "uppercase" in constraint_lower or "upper case" in constraint_lower or "capital" in constraint_lower:
            return self._check_uppercase(constraint, response)
        elif "lowercase" in constraint_lower or "lower case" in constraint_lower:
            return self._check_lowercase(constraint, response)
        elif "title case" in constraint_lower or "title-case" in constraint_lower:
            return self._check_title_case(response)
        else:
            return True

    def _get_alpha_text(self, text: str) -> str:
        """Extract only alphabetic characters from text for case checking."""
        return re.sub(r'[^a-zA-Z]', '', text)

    def _check_uppercase(self, constraint: str, response: str) -> bool:
        """Check if response is in uppercase."""
        alpha = self._get_alpha_text(response)
        if not alpha:
            return False
        constraint_lower = constraint.lower()

        if "all" in constraint_lower or "entire" in constraint_lower:
            return alpha == alpha.upper()
        else:
            return alpha == alpha.upper()

    def _check_lowercase(self, constraint: str, response: str) -> bool:
        """Check if response is in lowercase."""
        alpha = self._get_alpha_text(response)
        if not alpha:
            return False
        constraint_lower = constraint.lower()

        if "all" in constraint_lower or "entire" in constraint_lower:
            return alpha == alpha.lower()
        else:
            return alpha == alpha.lower()

    def _check_title_case(self, response: str) -> bool:
        """Check if response is in title case."""
        minor_words = {'a', 'an', 'the', 'and', 'but', 'or', 'nor', 'for', 'yet', 'so',
                       'in', 'on', 'at', 'to', 'by', 'of', 'up', 'as', 'is', 'it'}
        words = response.split()
        if not words:
            return False

        for i, word in enumerate(words):
            clean = word.strip('.,!?;:\"\'-()[]{}')
            if not clean or not clean[0].isalpha():
                continue
            if i == 0:
                if clean[0].islower():
                    return False
            elif clean.lower() not in minor_words:
                if clean[0].islower():
                    return False
        return True
