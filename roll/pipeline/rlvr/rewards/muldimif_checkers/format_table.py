"""Format_Table checker for MulDimIF constraints.

Ported from: https://github.com/Junjie-Ye/MulDimIF (Apache 2.0 License)
"""

import re


class Format_Table:
    """Check table format constraints in response text."""

    def check(self, constraint: str, response: str) -> bool:
        """Check table-related constraints."""
        constraint_lower = constraint.lower()

        if "row" in constraint_lower:
            return self._check_table_rows(constraint, response)
        elif "column" in constraint_lower:
            return self._check_table_columns(constraint, response)
        elif "table" in constraint_lower:
            return self._check_has_table(response)
        else:
            return self._check_has_table(response)

    def _parse_table_rows(self, response: str) -> list:
        """Parse markdown table rows from response."""
        lines = response.split('\n')
        table_rows = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('|') and stripped.endswith('|'):
                # Skip separator rows (|---|---|)
                if re.match(r'^\|[\s\-:|]+\|$', stripped):
                    continue
                table_rows.append(stripped)
        return table_rows

    def _check_has_table(self, response: str) -> bool:
        """Check if response contains a markdown table."""
        return len(self._parse_table_rows(response)) > 0

    def _check_table_rows(self, constraint: str, response: str) -> bool:
        """Check table row count constraints."""
        rows = self._parse_table_rows(response)
        constraint_lower = constraint.lower()

        # Extract the required count
        count_match = re.search(r'(\d+)', constraint)
        if not count_match:
            return len(rows) > 0

        required = int(count_match.group(1))

        if "at least" in constraint_lower:
            return len(rows) >= required
        elif "at most" in constraint_lower or "no more" in constraint_lower:
            return len(rows) <= required
        elif "exactly" in constraint_lower:
            return len(rows) == required
        else:
            return len(rows) >= required

    def _check_table_columns(self, constraint: str, response: str) -> bool:
        """Check table column count constraints."""
        rows = self._parse_table_rows(response)
        if not rows:
            return False

        # Count columns from the first data row
        first_row = rows[0]
        cols = [c.strip() for c in first_row.strip('|').split('|')]
        num_cols = len(cols)

        constraint_lower = constraint.lower()
        count_match = re.search(r'(\d+)', constraint)
        if not count_match:
            return num_cols > 0

        required = int(count_match.group(1))

        if "at least" in constraint_lower:
            return num_cols >= required
        elif "at most" in constraint_lower or "no more" in constraint_lower:
            return num_cols <= required
        elif "exactly" in constraint_lower:
            return num_cols == required
        else:
            return num_cols >= required
