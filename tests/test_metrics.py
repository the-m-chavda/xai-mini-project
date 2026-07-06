from xai_miniproject.metrics import accuracy, macro_f1


def test_accuracy() -> None:
    assert accuracy([0, 1, 1], [0, 0, 1]) == 2 / 3


def test_macro_f1_handles_missing_predictions() -> None:
    score = macro_f1([0, 1, 1], [0, 0, 0], labels=[0, 1])
    assert round(score, 4) == 0.25
