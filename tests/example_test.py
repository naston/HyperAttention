def is_even(number):
    return number % 2 == 0


def test_is_even():
    assert is_even(4) is True
    assert is_even(7) is False