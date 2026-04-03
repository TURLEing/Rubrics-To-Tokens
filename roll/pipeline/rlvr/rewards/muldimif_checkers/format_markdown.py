"""Format_Markdown checker for MulDimIF constraints.

Ported from: https://github.com/Junjie-Ye/MulDimIF (Apache 2.0 License)
"""

import re


class Format_Markdown:
    """Check Markdown formatting constraints in response text."""

    def check(self, constraint: str, response: str) -> bool:
        """Dispatch to the appropriate markdown check."""
        constraint_lower = constraint.lower()

        if "heading" in constraint_lower or "header" in constraint_lower:
            return self._check_headings(constraint, response)
        elif "bold" in constraint_lower:
            return self._check_bold(constraint, response)
        elif "bullet" in constraint_lower or "list" in constraint_lower:
            return self._check_bullet_points(constraint, response)
        elif "block quote" in constraint_lower or "blockquote" in constraint_lower or "> " in constraint:
            return self._check_block_quotes(constraint, response)
        elif "italic" in constraint_lower:
            return self._check_italic(constraint, response)
        elif "code" in constraint_lower:
            return self._check_code(constraint, response)
        else:
            return True

    def _check_headings(self, constraint: str, response: str) -> bool:
        """Check heading-level constraints."""
        # Find all headings
        headings = re.findall(r'^(#{1,6})\s+', response, re.MULTILINE)
        constraint_lower = constraint.lower()

        # Check for specific level requirement
        level_match = re.search(r'level\s+(\d+)', constraint_lower)
        if level_match:
            required_level = int(level_match.group(1))
            level_headings = [h for h in headings if len(h) == required_level]
            if "at least" in constraint_lower:
                count_match = re.search(r'at least\s+(\d+)', constraint_lower)
                if count_match:
                    return len(level_headings) >= int(count_match.group(1))
            return len(level_headings) > 0

        # Check heading count
        count_match = re.search(r'(\d+)', constraint)
        if count_match:
            required = int(count_match.group(1))
            if "at least" in constraint_lower:
                return len(headings) >= required
            elif "at most" in constraint_lower or "no more" in constraint_lower:
                return len(headings) <= required
            elif "exactly" in constraint_lower:
                return len(headings) == required
            else:
                return len(headings) >= required

        return len(headings) > 0

    def _check_bold(self, constraint: str, response: str) -> bool:
        """Check bold text constraints."""
        bold_matches = re.findall(r'\*\*[^*]+\*\*|__[^_]+__', response)
        constraint_lower = constraint.lower()

        count_match = re.search(r'(\d+)', constraint)
        if count_match:
            required = int(count_match.group(1))
            if "at least" in constraint_lower:
                return len(bold_matches) >= required
            elif "at most" in constraint_lower:
                return len(bold_matches) <= required
            elif "exactly" in constraint_lower:
                return len(bold_matches) == required

        if "no" in constraint_lower or "without" in constraint_lower:
            return len(bold_matches) == 0
        return len(bold_matches) > 0

    def _check_bullet_points(self, constraint: str, response: str) -> bool:
        """Check bullet point constraints."""
        bullets = re.findall(r'^\s*[-*+]\s+', response, re.MULTILINE)
        numbered = re.findall(r'^\s*\d+[.)]\s+', response, re.MULTILINE)
        total = len(bullets) + len(numbered)
        constraint_lower = constraint.lower()

        count_match = re.search(r'(\d+)', constraint)
        if count_match:
            required = int(count_match.group(1))
            if "at least" in constraint_lower:
                return total >= required
            elif "at most" in constraint_lower:
                return total <= required
            elif "exactly" in constraint_lower:
                return total == required
            else:
                return total >= required

        return total > 0

    def _check_block_quotes(self, constraint: str, response: str) -> bool:
        """Check block quote constraints."""
        quotes = re.findall(r'^\s*>\s+', response, re.MULTILINE)
        return len(quotes) > 0

    def _check_italic(self, constraint: str, response: str) -> bool:
        """Check italic text constraints."""
        italic_matches = re.findall(r'(?<!\*)\*(?!\*)[^*]+\*(?!\*)|(?<!_)_(?!_)[^_]+_(?!_)', response)
        constraint_lower = constraint.lower()
        if "no" in constraint_lower or "without" in constraint_lower:
            return len(italic_matches) == 0
        return len(italic_matches) > 0

    def _check_code(self, constraint: str, response: str) -> bool:
        """Check code block/inline code constraints."""
        code_blocks = re.findall(r'```[\s\S]*?```', response)
        inline_code = re.findall(r'`[^`]+`', response)
        constraint_lower = constraint.lower()
        if "block" in constraint_lower:
            return len(code_blocks) > 0
        return len(code_blocks) + len(inline_code) > 0
