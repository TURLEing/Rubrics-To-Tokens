"""Content_Others (Identifiers) checker for MulDimIF constraints.

Ported from: https://github.com/Junjie-Ye/MulDimIF (Apache 2.0 License)
"""

import re


class Content_Others:
    """Check content identifier constraints (start/end markers, emojis, blurbs, etc.)."""

    def check(self, constraint: str, response: str) -> bool:
        """Dispatch to the appropriate content check."""
        constraint_lower = constraint.lower()

        if "start with" in constraint_lower or "begin with" in constraint_lower:
            return self._check_starts_with(constraint, response)
        elif "end with" in constraint_lower:
            return self._check_ends_with(constraint, response)
        elif "emoji" in constraint_lower:
            return self._check_emoji(constraint, response)
        elif "contain" in constraint_lower or "include" in constraint_lower:
            return self._check_contains(constraint, response)
        else:
            return self._check_contains(constraint, response)

    def _extract_quoted(self, text: str) -> list:
        """Extract quoted strings from text."""
        quoted = re.findall(r'"([^"]+)"', text)
        if not quoted:
            quoted = re.findall(r"'([^']+)'", text)
        return quoted

    def _check_starts_with(self, constraint: str, response: str) -> bool:
        """Check if the response starts with a specific string."""
        quoted = self._extract_quoted(constraint)
        if quoted:
            return response.strip().startswith(quoted[0])
        return True

    def _check_ends_with(self, constraint: str, response: str) -> bool:
        """Check if the response ends with a specific string."""
        quoted = self._extract_quoted(constraint)
        if quoted:
            return response.strip().endswith(quoted[0])
        return True

    def _check_emoji(self, constraint: str, response: str) -> bool:
        """Check emoji-related constraints."""
        emoji_pattern = re.compile(
            r'['
            '\U0001F600-\U0001F64F'
            '\U0001F300-\U0001F5FF'
            '\U0001F680-\U0001F6FF'
            '\U0001F900-\U0001F9FF'
            '\U0001FA00-\U0001FA6F'
            '\U0001FA70-\U0001FAFF'
            '\U00002702-\U000027B0'
            '\U000024C2-\U0001F251'
            ']+', flags=re.UNICODE
        )
        constraint_lower = constraint.lower()
        has_emoji = bool(emoji_pattern.search(response))

        if "no" in constraint_lower or "without" in constraint_lower or "not" in constraint_lower:
            return not has_emoji
        return has_emoji

    def _check_contains(self, constraint: str, response: str) -> bool:
        """Check if response contains a specific string."""
        quoted = self._extract_quoted(constraint)
        if not quoted:
            return True
        response_lower = response.lower()
        constraint_lower = constraint.lower()
        if "not" in constraint_lower or "no" in constraint_lower:
            return all(q.lower() not in response_lower for q in quoted)
        return all(q.lower() in response_lower for q in quoted)
