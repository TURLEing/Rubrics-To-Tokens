"""Content_Punctuation checker for MulDimIF constraints.

Ported from: https://github.com/Junjie-Ye/MulDimIF (Apache 2.0 License)
"""

import re


class Content_Punctuation:
    """Check punctuation-related constraints in response text."""

    def check(self, constraint: str, response: str) -> bool:
        """Dispatch to the appropriate punctuation check."""
        constraint_lower = constraint.lower()

        if "end" in constraint_lower and ("sentence" in constraint_lower or "." in constraint or "!" in constraint):
            return self._check_ending_punctuation(constraint, response)
        elif "exclamation" in constraint_lower:
            return self._check_punctuation_presence(constraint, response, "!")
        elif "question mark" in constraint_lower:
            return self._check_punctuation_presence(constraint, response, "?")
        elif "comma" in constraint_lower:
            return self._check_punctuation_presence(constraint, response, ",")
        elif "semicolon" in constraint_lower:
            return self._check_punctuation_presence(constraint, response, ";")
        elif "colon" in constraint_lower:
            return self._check_punctuation_presence(constraint, response, ":")
        elif "no" in constraint_lower and "punctuation" in constraint_lower:
            return not bool(re.search(r'[.,!?;:]', response))
        else:
            return self._check_ending_punctuation(constraint, response)

    def _split_sentences(self, text: str) -> list:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _check_ending_punctuation(self, constraint: str, response: str) -> bool:
        """Check that sentences end with specific punctuation."""
        # Extract the required punctuation
        punct = "."
        if "exclamation" in constraint.lower() or "!" in constraint:
            punct = "!"
        elif "question" in constraint.lower() or "?" in constraint:
            punct = "?"

        sentences = self._split_sentences(response)
        if not sentences:
            return False

        if "every" in constraint.lower() or "each" in constraint.lower() or "all" in constraint.lower():
            return all(s.rstrip()[-1] == punct for s in sentences if s.rstrip())
        else:
            # Just check the last sentence
            return sentences[-1].rstrip()[-1] == punct if sentences[-1].rstrip() else False

    def _check_punctuation_presence(self, constraint: str, response: str, punct: str) -> bool:
        """Check presence/absence of a specific punctuation mark."""
        constraint_lower = constraint.lower()
        has_punct = punct in response

        if "no" in constraint_lower or "without" in constraint_lower or "not" in constraint_lower:
            return not has_punct
        else:
            # Check for count constraints
            count_match = re.search(r'(?:at least|exactly|at most)\s+(\d+)', constraint_lower)
            if count_match:
                required = int(count_match.group(1))
                actual = response.count(punct)
                if "at least" in constraint_lower:
                    return actual >= required
                elif "exactly" in constraint_lower:
                    return actual == required
                elif "at most" in constraint_lower:
                    return actual <= required
            return has_punct
