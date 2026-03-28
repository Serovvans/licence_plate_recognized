# src/ocr/postprocessing.py

import re

LETTER_TO_DIGIT = {'O': '0', 'I': '1', 'S': '5', 'B': '8', 'Z': '2'}
DIGIT_TO_LETTER = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '2': 'Z'}

CJK_RANGE = r'\u4e00-\u9fff'


def clean_text(text: str) -> str:
    return re.sub(fr'[^A-Za-z0-9{CJK_RANGE}]', '', text).upper()


def find_letter_digit_split(text: str) -> int:
    """
    Буквенная зона HK номера — максимум 2 символа.
    В первых двух позициях цифры-похожие-на-буквы (0,1,2,5,8)
    тоже считаем частью буквенной зоны.
    """
    split = 0
    for i, c in enumerate(text):
        if i >= 2:
            break
        if c.isalpha() or c in DIGIT_TO_LETTER:
            split = i + 1
        else:
            break
    return split


def correct_plate(text: str) -> str:
    if not text:
        return text

    # Материковый номер — иероглиф в начале, не трогаем
    if re.match(fr'^[{CJK_RANGE}]', text):
        return text

    split = find_letter_digit_split(text)

    result = []
    for i, c in enumerate(text):
        if i < split:
            result.append(DIGIT_TO_LETTER.get(c, c))  # цифры → буквы
        else:
            result.append(LETTER_TO_DIGIT.get(c, c))  # буквы → цифры
    return ''.join(result)


def postprocess(text: str) -> str:
    text = clean_text(text)
    text = correct_plate(text)
    return text