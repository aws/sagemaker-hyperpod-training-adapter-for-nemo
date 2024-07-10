def func(x):
    return x + 1


def test_pass_answer():
    assert func(3) == 4


def test_fail_answer():
    assert func(3) == 5