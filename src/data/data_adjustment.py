import re
from spellchecker import SpellChecker

class ReviewAdjuster():
    def __init__(self):
        self.spell = SpellChecker()
        self.word_pattern = re.compile(r'\b[a-zA-Z]+\b')

    def is_spelling_adequate(self, review_text : str) -> bool:
        words = set(self.word_pattern.findall(review_text))
        misspelled = self.spell.unknown(words)

        if len(misspelled) == len(words):
            return False
        elif 1.0 - (len(misspelled)/len(words)) >= 0.9:
            return True
        else:
            return False

    def replace_spelling_errors(self, review_text: str):
        words = set(self.word_pattern.findall(review_text))
        misspelled = self.spell.unknown(words)

        corrections = {}
        for word in misspelled:
            correction = self.spell.correction(word)
            if correction and correction != word:
                corrections[word] = correction

        if not corrections:
            return len(misspelled), 0, review_text

        pattern = re.compile(
            r'\b(' + '|'.join(re.escape(w) for w in corrections) + r')\b',
            re.IGNORECASE
        )
        result = pattern.sub(lambda m: corrections.get(m.group(), m.group()), review_text)

        # print("Misspelled words:", misspelled)

        return len(misspelled), len(corrections), result


    def adjust_review(self, review_text : str):
        return self.replace_spelling_errors(review_text)