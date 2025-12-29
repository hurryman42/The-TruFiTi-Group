import re

import language_tool_python.utils
from spellchecker import SpellChecker
import language_tool_python as lang_tool

class ReviewAdjuster():
    def __init__(self):
        self.spell = SpellChecker(distance=1)
        self.sensitivity = 0.9
        self.word_pattern = re.compile(r'\b[a-zA-Z]+\b')
        self.tool = lang_tool.LanguageTool('en-US')
        self.tool.enabled_rules_only = True
        self.tool.enabled_categories = {"GRAMMAR", "COLLOCATIONS", "PUNCTUATION", "TYPOGRAPHY"}


    def is_spelling_adequate(self, review_text: str) -> bool:
        words = set(self.word_pattern.findall(review_text))
        misspelled = self.spell.unknown(words)

        if len(misspelled) == len(words):
            return False
        elif 1.0 - (len(misspelled) / len(words)) >= self.sensitivity:
            return True
        else:
            return False


    def is_grammar_adequate(self, review_text: str) -> bool:
        matches = self.tool.check(review_text)
        if len(matches) < 1:
            return True
        else:
            return False


    def is_text_adequate(self, review_text: str) -> bool:
        return self.is_spelling_adequate(review_text) and self.is_grammar_adequate(review_text)


    def replace_spelling_errors(self, review_text: str):
        words = set(self.word_pattern.findall(review_text))
        misspelled = self.spell.unknown(words)

        corrections = {}
        for word in misspelled:
            correction = self.spell.correction(word)
            if correction and correction != word:
                corrections[word] = correction

        if not corrections:
            return review_text

        pattern = re.compile(
            r'\b(' + '|'.join(re.escape(w) for w in corrections) + r')\b',
            re.IGNORECASE
        )
        result = pattern.sub(lambda m: corrections.get(m.group(), m.group()), review_text)

        # print("Misspelled words:", misspelled)

        return result


    def replace_grammar_errors(self, review_text: str):
        matches = self.tool.check(review_text)
        return language_tool_python.utils.correct(review_text, matches)


    def adjust_review(self, review_text: str):
        intermediate_result = self.replace_spelling_errors(review_text)
        return self.replace_grammar_errors(intermediate_result)