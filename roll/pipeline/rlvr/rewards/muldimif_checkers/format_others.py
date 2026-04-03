"""Format_Others (XML, APA, references, etc.) checker for MulDimIF constraints.

Ported from: https://github.com/Junjie-Ye/MulDimIF (Apache 2.0 License)
"""

import re


class Format_Others:
    """Check miscellaneous format constraints (XML, APA, references, table rows)."""

    def check(self, constraint: str, response: str) -> bool:
        """Dispatch to the appropriate format check."""
        constraint_lower = constraint.lower()

        if "xml" in constraint_lower or "tag" in constraint_lower:
            return self._check_xml(constraint, response)
        elif "apa" in constraint_lower:
            return self._check_apa(constraint, response)
        elif "reference" in constraint_lower or "citation" in constraint_lower:
            return self._check_references(constraint, response)
        elif "number" in constraint_lower and "list" in constraint_lower:
            return self._check_numbered_list(constraint, response)
        else:
            return True

    def _check_xml(self, constraint: str, response: str) -> bool:
        """Check XML-related constraints (tags, attributes)."""
        constraint_lower = constraint.lower()

        # Extract required tag names
        quoted = re.findall(r'"([^"]+)"', constraint)
        if not quoted:
            quoted = re.findall(r"'([^']+)'", constraint)

        if quoted:
            for tag in quoted:
                # Check for opening and closing tags
                if not re.search(rf'<{re.escape(tag)}[\s>]', response):
                    return False
            return True

        # General XML check - at least one XML tag
        if "attribute" in constraint_lower:
            return bool(re.search(r'<\w+\s+\w+\s*=\s*["\'][^"\']*["\']', response))
        return bool(re.search(r'<\w+[\s>].*</\w+>', response, re.DOTALL))

    def _check_apa(self, constraint: str, response: str) -> bool:
        """Check APA format constraints."""
        # Basic APA citation pattern: (Author, Year) or Author (Year)
        return bool(re.search(r'\(\w+,?\s*\d{4}\)|\w+\s*\(\d{4}\)', response))

    def _check_references(self, constraint: str, response: str) -> bool:
        """Check reference/citation constraints."""
        constraint_lower = constraint.lower()

        # Count reference patterns
        ref_patterns = [
            r'\[\d+\]',  # [1], [2]
            r'\(\w+,?\s*\d{4}\)',  # (Author, Year)
        ]
        total_refs = 0
        for pattern in ref_patterns:
            total_refs += len(re.findall(pattern, response))

        count_match = re.search(r'(\d+)', constraint)
        if count_match:
            required = int(count_match.group(1))
            if "at least" in constraint_lower:
                return total_refs >= required
            elif "at most" in constraint_lower:
                return total_refs <= required
            elif "exactly" in constraint_lower:
                return total_refs == required
            return total_refs >= required

        return total_refs > 0

    def _check_numbered_list(self, constraint: str, response: str) -> bool:
        """Check numbered list constraints."""
        items = re.findall(r'^\s*\d+[.)]\s+', response, re.MULTILINE)
        constraint_lower = constraint.lower()

        count_match = re.search(r'(\d+)', constraint)
        if count_match:
            required = int(count_match.group(1))
            if "at least" in constraint_lower:
                return len(items) >= required
            elif "exactly" in constraint_lower:
                return len(items) == required
            return len(items) >= required

        return len(items) > 0
