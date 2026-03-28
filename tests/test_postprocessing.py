from src.ocr.postprocessing import postprocess

def test_clean():
    assert postprocess("AB-12 34!") == "AB1234"

def test_letter_zone():
    # O в буквенной зоне → должна остаться O (не 0)
    assert postprocess("0B1234").startswith("OB")

def test_digit_zone():
    # O в цифровой зоне → должна стать 0
    assert postprocess("ABIO34") == "AB1034"

def test_cjk_untouched():
    # Иероглиф в начале — материковый номер, не трогаем
    assert postprocess("粤A12345").startswith("粤")