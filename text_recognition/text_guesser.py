import difflib
import re
from typing import Any

import jamotools

from constants import KOR_ANSWER_LIST, KOR_OCR_SIMILARITY


def guess_text(text: str) -> Any:
    splitted_text = jamotools.split_syllables(re.sub(r"\s+", "", text))
    best_ratio = 0
    result = text
    for answer_text in KOR_ANSWER_LIST:
        splitted_answer = jamotools.split_syllables(answer_text)
        seq = difflib.SequenceMatcher(None, splitted_text, splitted_answer)
        if seq.ratio() > KOR_OCR_SIMILARITY and best_ratio < seq.ratio():
            best_ratio = seq.ratio()
            result = answer_text
    return result


if __name__ == "__main__":
    guess_text(text="총내용량")
