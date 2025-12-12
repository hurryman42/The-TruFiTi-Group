from spellchecker import SpellChecker

class ReviewAdjuster():
    def __init__(self):
        self.spell = SpellChecker()

    def replace_spelling_errors(self, review_text: str):
        for char in [".", ",", ":", "!", "?", '"', "'", "’", "(", ")", "-"]:
            review_text = review_text.replace(char, "")
        review_text_words = review_text.split()
        possible_misspelled_words = self.spell.unknown(review_text_words)
        misspelled_words = list(dict.fromkeys(possible_misspelled_words))
        num_fixes = 0

        for word in misspelled_words:
            replacement = self.spell.correction(word)

            if replacement is None:
                continue

            review_text = review_text.replace(word, replacement)
            num_fixes += 1

        print("Misspelled words:", misspelled_words)
        return len(misspelled_words), num_fixes, review_text

    def adjust_review(self, review_text : str):
        return self.replace_spelling_errors(review_text)