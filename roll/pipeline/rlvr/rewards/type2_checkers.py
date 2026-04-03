"""Code-based checkers for type2 (Chat Arena) instruction following constraints."""

import re
from typing import Any, Dict, Optional


# Common English pronouns
PRONOUNS = {
    'i', 'me', 'my', 'mine', 'myself',
    'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself',
    'we', 'us', 'our', 'ours', 'ourselves',
    'they', 'them', 'their', 'theirs', 'themselves',
    'who', 'whom', 'whose', 'which', 'that',
    'this', 'these', 'those',
}

# Coordinating conjunctions (FANBOYS)
COORDINATING_CONJUNCTIONS = {'for', 'and', 'nor', 'but', 'or', 'yet', 'so'}

# Common English stop words
STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
    'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
    'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'through', 'during', 'before', 'after', 'above',
    'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
    't', 'can', 'will', 'just', 'don', 'should', 'now',
}

# Emoji regex pattern
EMOJI_PATTERN = re.compile(
    r'['
    '\U0001F600-\U0001F64F'
    '\U0001F300-\U0001F5FF'
    '\U0001F680-\U0001F6FF'
    '\U0001F900-\U0001F9FF'
    '\U0001FA00-\U0001FA6F'
    '\U0001FA70-\U0001FAFF'
    '\U00002702-\U000027B0'
    '\U000024C2-\U0001F251'
    '\U00010000-\U0010FFFF'
    ']+', flags=re.UNICODE
)


def _split_sentences(text: str) -> list:
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _get_words(text: str) -> list:
    """Extract words from text."""
    return re.findall(r'\b\w+\b', text.lower())


def _is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def check_pronouns(text: str, kwargs: Dict[str, Any]) -> bool:
    """Check: at least N pronouns in response."""
    n = kwargs.get('N', 0)
    words = _get_words(text)
    count = sum(1 for w in words if w in PRONOUNS)
    return count >= n


def check_numbers(text: str, kwargs: Dict[str, Any]) -> bool:
    """Check: exactly N numbers in response."""
    n = kwargs.get('N', 0)
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    return len(numbers) == n


def check_unique_word_count(text: str, kwargs: Dict[str, Any]) -> bool:
    """Check: at least N unique words."""
    n = kwargs.get('N', 0)
    words = _get_words(text)
    return len(set(words)) >= n


def check_word_count_range(text: str, kwargs: Dict[str, Any]) -> bool:
    """Check: word count between min_words and max_words."""
    min_words = kwargs.get('min_words', 0)
    max_words = kwargs.get('max_words', float('inf'))
    word_count = len(text.split())
    return min_words <= word_count <= max_words


def check_conjunctions(text: str, kwargs: Dict[str, Any]) -> bool:
    """Check: at least N different coordinating conjunctions."""
    n = kwargs.get('small_n', 0)
    words = _get_words(text)
    found = set(w for w in words if w in COORDINATING_CONJUNCTIONS)
    return len(found) >= n


def check_keywords_multiple(text: str, kwargs: Dict[str, Any]) -> bool:
    """Check: keyword1 once, keyword2 twice, keyword3 three times, keyword4 five times, keyword5 seven times."""
    required_counts = {
        'keyword1': 1, 'keyword2': 2, 'keyword3': 3,
        'keyword4': 5, 'keyword5': 7,
    }
    text_lower = text.lower()
    for key, required in required_counts.items():
        keyword = kwargs.get(key)
        if keyword is None:
            continue
        pattern = re.compile(r'\b' + re.escape(keyword.lower()) + r'\b')
        actual = len(pattern.findall(text_lower))
        if actual < required:
            return False
    return True


def check_emoji(text: str, kwargs: Dict[str, Any]) -> bool:
    """Check: emoji at end of every sentence."""
    sentences = _split_sentences(text)
    if not sentences:
        return False
    for sent in sentences:
        sent = sent.rstrip()
        if not sent:
            continue
        # Check if last character(s) contain emoji
        last_chars = sent[-5:] if len(sent) >= 5 else sent
        if not EMOJI_PATTERN.search(last_chars):
            return False
    return True


def check_line_indent(text: str, kwargs: Dict[str, Any]) -> bool:
    """Check: incrementally indenting each new line (staircase pattern)."""
    lines = text.split('\n')
    non_empty = [line for line in lines if line.strip()]
    if len(non_empty) < 2:
        return False
    prev_indent = len(non_empty[0]) - len(non_empty[0].lstrip())
    for line in non_empty[1:]:
        curr_indent = len(line) - len(line.lstrip())
        if curr_indent <= prev_indent:
            return False
        prev_indent = curr_indent
    return True


def check_output_template(text: str, kwargs: Dict[str, Any]) -> bool:
    """Check: response follows template 'My Answer: ... My Conclusion: ... Future Outlook: ...'."""
    has_answer = bool(re.search(r'My Answer:', text, re.IGNORECASE))
    has_conclusion = bool(re.search(r'My Conclusion:', text, re.IGNORECASE))
    has_outlook = bool(re.search(r'Future Outlook:', text, re.IGNORECASE))
    return has_answer and has_conclusion and has_outlook


def check_title_case(text: str, kwargs: Dict[str, Any]) -> bool:
    """Check: entire response in title case (first letter of every major word capitalized)."""
    # Minor words that don't need capitalization in title case
    minor_words = {'a', 'an', 'the', 'and', 'but', 'or', 'nor', 'for', 'yet', 'so',
                   'in', 'on', 'at', 'to', 'by', 'of', 'up', 'as', 'is', 'it'}
    words = text.split()
    if not words:
        return False
    for i, word in enumerate(words):
        # Strip punctuation for checking
        clean = word.strip('.,!?;:"\'-()[]{}')
        if not clean:
            continue
        # First word must be capitalized
        if i == 0:
            if clean[0].islower():
                return False
        elif clean.lower() not in minor_words:
            if clean[0].islower():
                return False
    return True


def check_sub_bullets(text: str, kwargs: Dict[str, Any]) -> bool:
    """Check: bullet points with * and sub-bullets with - for each."""
    lines = text.split('\n')
    bullet_lines = []
    current_bullet = None
    has_sub = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('* ') or stripped.startswith('*\t'):
            if current_bullet is not None and not has_sub:
                return False
            current_bullet = stripped
            has_sub = False
        elif stripped.startswith('- ') or stripped.startswith('-\t'):
            if current_bullet is not None:
                has_sub = True
            bullet_lines.append(stripped)

    # Check last bullet
    if current_bullet is not None and not has_sub:
        return False
    # Must have at least one bullet
    return current_bullet is not None


def check_stop_words(text: str, kwargs: Dict[str, Any]) -> bool:
    """Check: stop words no more than N% of total words."""
    percentage = kwargs.get('percentage', 100)
    words = _get_words(text)
    if not words:
        return True
    stop_count = sum(1 for w in words if w in STOP_WORDS)
    actual_pct = (stop_count / len(words)) * 100
    return actual_pct <= percentage


def check_sentence_keyword(text: str, kwargs: Dict[str, Any]) -> bool:
    """Check: keyword appears in the N-th sentence."""
    n = kwargs.get('N', 1)
    word = kwargs.get('word', '')
    sentences = _split_sentences(text)
    if n > len(sentences) or n < 1:
        return False
    target_sentence = sentences[n - 1].lower()
    return word.lower() in target_sentence


def check_last_first(text: str, kwargs: Dict[str, Any]) -> bool:
    """Check: last word of each sentence becomes first word of next sentence."""
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return len(sentences) == 1
    for i in range(len(sentences) - 1):
        words_current = re.findall(r'\b\w+\b', sentences[i])
        words_next = re.findall(r'\b\w+\b', sentences[i + 1])
        if not words_current or not words_next:
            return False
        if words_current[-1].lower() != words_next[0].lower():
            return False
    return True


def check_no_consecutive(text: str, kwargs: Dict[str, Any]) -> bool:
    """Check: no two consecutive words share the same first letter."""
    words = re.findall(r'\b[a-zA-Z]\w*\b', text)
    for i in range(len(words) - 1):
        if words[i][0].lower() == words[i + 1][0].lower():
            return False
    return True


def check_prime_lengths(text: str, kwargs: Dict[str, Any]) -> bool:
    """Check: all words have lengths that are prime numbers."""
    words = re.findall(r'\b\w+\b', text)
    if not words:
        return False
    return all(_is_prime(len(w)) for w in words)


def check_consonants(text: str, kwargs: Dict[str, Any]) -> bool:
    """Check: each word has at least one consonant cluster (2+ consonants together)."""
    consonant_cluster = re.compile(r'[bcdfghjklmnpqrstvwxyz]{2,}', re.IGNORECASE)
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    if not words:
        return False
    return all(consonant_cluster.search(w) for w in words)


# Mapping from type2 instruction_id to checker function.
# Functions that return None mean "not implemented, fall back to LLM".
TYPE2_CHECKERS = {
    'count:pronouns': check_pronouns,
    'count:numbers': check_numbers,
    'count:unique_word_count': check_unique_word_count,
    'count:word_count_range': check_word_count_range,
    'count:conjunctions': check_conjunctions,
    'count:keywords_multiple': check_keywords_multiple,
    'format:emoji': check_emoji,
    'format:line_indent': check_line_indent,
    'format:output_template': check_output_template,
    'format:title_case': check_title_case,
    'format:sub-bullets': check_sub_bullets,
    'ratio:stop_words': check_stop_words,
    'sentence:keyword': check_sentence_keyword,
    'words:last_first': check_last_first,
    'words:no_consecutive': check_no_consecutive,
    'words:prime_lengths': check_prime_lengths,
    'words:consonants': check_consonants,
    # These fall back to LLM (not in this map):
    # 'format:quotes' - complex nested quote detection
    # 'ratio:sentence_balance' - "balanced" is vague
    # 'ratio:sentence_type' - needs sentence type classification
    # 'words:start_verb' - needs POS tagging
}
