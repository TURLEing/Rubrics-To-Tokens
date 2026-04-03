"""MulDimIF constraint checkers for type3 evaluation.

Ported from: https://github.com/Junjie-Ye/MulDimIF (Apache 2.0 License)
"""

from .content_keywords import Content_Keywords
from .content_punctuation import Content_Punctuation
from .content_others import Content_Others
from .format_json import Format_Json
from .format_markdown import Format_Markdown
from .format_others import Format_Others
from .format_table import Format_Table
from .language_english import Language_English
from .length_paragraphs import Length_Paragraphs
from .length_sentences import Length_Sentences
from .length_words import Length_Words

# Mapping from constraint "{Category}_{Subcategory}" to checker instance.
# Based on the MulDimIF evaluation.py class_mapping.
# Language_Chinese is excluded (requires zhconv dependency); those entries fall back to LLM.
CONSTRAINT_CHECKER_MAP = {
    "Content_Keywords": Content_Keywords(),
    "Content_Punctuation": Content_Punctuation(),
    "Content_Identifiers": Content_Others(),
    "Format_Json": Format_Json(),
    "Format_Markdown": Format_Markdown(),
    "Format_Table": Format_Table(),
    "Format_XML": Format_Others(),
    "Language_English": Language_English(),
    "Length_Paragraphs": Length_Paragraphs(),
    "Length_Sentences": Length_Sentences(),
    "Length_Words": Length_Words(),
}
