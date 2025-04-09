import pytest

from dragon_baseline import DragonBaseline
from dragon_baseline.nlp_algorithm import ProblemType


class MockTaskTarget:
    def __init__(self, problem_type, prediction_name):
        self.problem_type = problem_type
        self.prediction_name = prediction_name

class MockTask:
    def __init__(self, target):
        self.target = target


@pytest.mark.parametrize(
    "predictions, expected_output, problem_type",
    [
        ([{"prediction": ["A", "B", 1, True]}], [{"prediction": ["A", "B", "1", "True"]}], ProblemType.SINGLE_LABEL_NER),
        ([{"prediction":[["A"], ["B", 1], [True]] }], [{"prediction": [["A"], ["B", "1"], ["True"]]}], ProblemType.MULTI_LABEL_NER),
        ([{"prediction": True}], [{"prediction": 1.0}], ProblemType.SINGLE_LABEL_BINARY_CLASSIFICATION),
        ([{"prediction": [True, 1, 0.9, False]}], [{"prediction": [1.0, 1.0, 0.9, 0.0]}], ProblemType.MULTI_LABEL_BINARY_CLASSIFICATION),
        ([{"prediction": 1}], [{"prediction": "1"}], ProblemType.SINGLE_LABEL_MULTI_CLASS_CLASSIFICATION),
        ([{"prediction": [1, "A", True]}], [{"prediction": ["1", "A", "True"]}], ProblemType.MULTI_LABEL_MULTI_CLASS_CLASSIFICATION),
        ([{"prediction": "0.123"}], [{"prediction": 0.123}], ProblemType.SINGLE_LABEL_REGRESSION),
        ([{"prediction": ["0.123", 0.123, False]}], [{"prediction": [0.123, 0.123, 0]}], ProblemType.MULTI_LABEL_REGRESSION),
        ([{"prediction": 1}], [{"prediction": "1"}], ProblemType.SINGLE_LABEL_TEXT),
    ]
)
def test_casting(predictions, expected_output, problem_type):
    algorithm = DragonBaseline()
    algorithm.task = MockTask(MockTaskTarget(problem_type, "prediction"))
    output = algorithm.cast_predictions(predictions)
    assert output == expected_output


if __name__ == "__main__":
    pytest.main()
