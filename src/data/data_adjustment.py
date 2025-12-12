from spellchecker import SpellChecker

class ReviewAdjuster():
    def __init__(self):
        self.spell = SpellChecker()

    def count_spelling_errors(self, review_text : str) -> int:
        for char in [".", ",", "?", "\"", "\'"]:
            review_text = review_text.replace(char, "")
        review_text_words = review_text.split()
        possible_misspelled_words = self.spell.unknown(review_text_words)
        return len(possible_misspelled_words)

    def replace_spelling_errors(self, review_text: str) -> str:
        for char in [".", ",", "?", "\"", "\'"]:
            review_text = review_text.replace(char, "")
        review_text_words = review_text.split()
        possible_misspelled_words = self.spell.unknown(review_text_words)
        misspelled_words = list(dict.fromkeys(possible_misspelled_words))

        for word in misspelled_words:
            replacement = self.spell.correction(word)

            if replacement is None:
                continue

            review_text = review_text.replace(word, replacement)

        print("Misspelled words:", misspelled_words)
        return review_text

    def adjust_review(self, review_text : str) -> str:
        return self.replace_spelling_errors(review_text)