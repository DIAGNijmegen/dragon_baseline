#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch
from sklearn.base import TransformerMixin


class ProblemType(Enum):
    """Problem type of the task"""
    SINGLE_LABEL_REGRESSION = "single_label_regression"
    MULTI_LABEL_REGRESSION = "multi_label_regression"
    SINGLE_LABEL_BINARY_CLASSIFICATION = "single_label_binary_classification"
    MULTI_LABEL_BINARY_CLASSIFICATION = "multi_label_binary_classification"
    SINGLE_LABEL_MULTI_CLASS_CLASSIFICATION = "single_label_multi_class_classification"
    MULTI_LABEL_MULTI_CLASS_CLASSIFICATION = "multi_label_multi_class_classification"
    SINGLE_LABEL_NER = "named_entity_recognition"
    MULTI_LABEL_NER = "multi_label_named_entity_recognition"
    SINGLE_LABEL_TEXT = "text"


def string_to_ProblemType(type_str):
    """
    Convert a string to a corresponding Enum member if possible.

    Args:
        type_str (str): The string representation of the ProblemType.

    Returns:
        ProblemType: The corresponding Enum member.

    Raises:
        ValueError: If the type_str does not correspond to a valid ProblemType.
    """
    for problem_type in ProblemType:
        if problem_type.value == type_str:
            return problem_type

    raise ValueError(f"{type_str} does not correspond to a valid ProblemType")


@dataclass
class LabelDetails:
    """
    Details about a label.

    Attributes:
        problem_type (str): The type of problem associated with the label.
        label_name (str): The name of the label.
        prediction_name (str): The name of the prediction associated with the label.
        values (List[Union[str, int, float]], optional): The list of values associated with the label. Set after calling `NLPAlgorithm.analyze()`.
        mean (float, optional): The mean value of the label. Defaults to None. Set after calling `NLPAlgorithm.analyze()`.
        std (float, optional): The standard deviation of the label. Defaults to None. Set after calling `NLPAlgorithm.analyze()`.
        min (float, optional): The minimum value of the label. Defaults to None. Set after calling `NLPAlgorithm.analyze()`.
        max (float, optional): The maximum value of the label. Defaults to None. Set after calling `NLPAlgorithm.analyze()`.
        skew (float, optional): The skewness of the label. Defaults to None. Set after calling `NLPAlgorithm.analyze()`.
    """

    problem_type: str
    label_name: str
    prediction_name: str
    values: List[Union[str, int, float]] = None
    mean: float = None
    std: float = None
    min: float = None
    max: float = None
    skew: float = None

    @classmethod
    def from_label_name(cls, name: str):
        """Create a `LabelDetails` object from a label name as {problem_type}_target`.

        Args:
            name (str): The label name.

        Returns:
            LabelDetails: The created `LabelDetails` object.

        Raises:
            ValueError: If the label name does not end with '_target'.
        """
        if not name.endswith("_target"):
            raise ValueError(f"Expected label name to end with '_target', but got '{name}'")
        prediction_name = name[:-len("_target")]
        problem_type = string_to_ProblemType(prediction_name)

        return cls(
            problem_type=problem_type,
            label_name=name,
            prediction_name=prediction_name,
        )


@dataclass
class TaskDetails:
    """Details about the task.

    Attributes:
        version (str): Version of the task details format.
        jobid (int): Unique identifier for this job.
        task_name (str): Name of the task.
        input_name (str): Name of algorithm input ("text" for a single input string, "text_parts" for a list of strings).
        target (LabelDetails): Details about the target.
        recommended_truncation_side (str): "left" or "right".

    Methods:
        from_json(cls, path: Path) -> TaskDetails: Create a TaskDetails object from a JSON file.
    """

    version: str
    jobid: int
    task_name: str
    input_name: str
    target: LabelDetails
    recommended_truncation_side: str

    @classmethod
    def from_json(cls, path: Path) -> "TaskDetails":
        """Create a TaskDetails object from a JSON file.

        Args:
            path (Path): Path to the JSON file.

        Returns:
            TaskDetails: The created TaskDetails object.
        """
        with open(path) as f:
            task_details = json.load(f)
        return cls(
            version=task_details["version"],
            jobid=task_details["jobid"],
            task_name=task_details["task_name"],
            input_name=task_details["input_name"],
            target=LabelDetails.from_label_name(task_details["label_name"]),
            recommended_truncation_side=task_details["recommended_truncation_side"],
        )


class NLPAlgorithm:
    def __init__(self, input_path: Path = Path("/input"), output_path: Path = Path("/output")):
        """
        The base class for NLP algorithms. Sets the environment and controls
        the flow of the processing once `process` is called.

        Args:
            input_path (Path): Path to the data folder. Default: `/input`
            output_path (Path): Path where the output predictions will be written. Default: `/output`
        """

        # defaults
        self.df_train: pd.DataFrame = None  # loaded in self.load()
        self.df_val: pd.DataFrame = None  # loaded in self.load()
        self.df_test: pd.DataFrame = None  # loaded in self.load()
        self.task: TaskDetails = None  # loaded in self.load()
        self.label_scalers: Dict[str, TransformerMixin] = {}  # set in self.scale_labels()
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps:0")
        else:
            self.device = torch.device("cpu")

        # paths
        self._input_path = input_path
        self._output_path = output_path
        self.dataset_train_path = self._input_path / "nlp-training-dataset.json"
        self.dataset_val_path = self._input_path / "nlp-validation-dataset.json"
        self.dataset_test_path = self._input_path / "nlp-test-dataset.json"
        self.task_details_path = self._input_path / "nlp-task-configuration.json"
        self.test_predictions_path = self._output_path / "nlp-predictions-dataset.json"

    def process(self):
        self.load()
        self.validate()
        self.analyze()
        self.preprocess()
        self.train()
        predictions = self.predict(df=self.df_test)
        self.save(predictions)
        self.verify_predictions()

    def load_dataset(self, path: Path) -> Union[pd.DataFrame, None]:
        """Load a dataset."""
        if not path.exists():
            print(f"Path '{path}' does not exist, skipping loading")
            return None

        df = pd.read_json(path, dtype={"uid": str})

        # cast and validate algorithm inputs
        if not "uid" in df.columns:
            raise ValueError("Expected column 'uid' in input data")
        if self.task.input_name == "text":
            df[self.task.input_name] = df[self.task.input_name].astype(str)
        elif self.task.input_name == "text_parts":
            df[self.task.input_name] = df[self.task.input_name].apply(lambda x: [str(value) for value in x])
        else:
            raise ValueError(f"Unexpected input name '{self.task.input_name}'")

        # cast and validate labels
        target = self.task.target
        if target.label_name in df.columns:
            if target.problem_type == ProblemType.SINGLE_LABEL_REGRESSION:
                df[target.label_name] = df[target.label_name].astype(float)
            elif target.problem_type == ProblemType.MULTI_LABEL_REGRESSION:
                df[target.label_name] = df[target.label_name].apply(lambda x: [float(value) if value is not None else np.nan for value in x])
            elif target.problem_type == ProblemType.SINGLE_LABEL_BINARY_CLASSIFICATION:
                df[target.label_name] = df[target.label_name].astype(int)
                if not all(value in [0, 1] for value in df[target.label_name]):
                    raise ValueError(f"Expected values in column '{target.label_name}' to be 0 or 1")
            elif target.problem_type == ProblemType.MULTI_LABEL_BINARY_CLASSIFICATION:
                df[target.label_name] = df[target.label_name].apply(lambda x: [int(value) for value in x])
                if not all(value in [0, 1] for value in df[target.label_name].explode()):
                    raise ValueError(f"Expected values in column '{target.label_name}' to be 0 or 1")
            elif target.problem_type == ProblemType.SINGLE_LABEL_MULTI_CLASS_CLASSIFICATION:
                df[target.label_name] = df[target.label_name].astype(str)
            elif target.problem_type == ProblemType.MULTI_LABEL_MULTI_CLASS_CLASSIFICATION:
                df[target.label_name] = df[target.label_name].apply(lambda x: [str(value) for value in x])
            elif target.problem_type == ProblemType.SINGLE_LABEL_NER:
                df[target.label_name] = df[target.label_name].apply(lambda x: [str(value) for value in x])
            elif target.problem_type == ProblemType.MULTI_LABEL_NER:
                df[target.label_name] = df[target.label_name].apply(lambda x: [[str(val) for val in value] for value in x])
            elif target.problem_type == ProblemType.SINGLE_LABEL_TEXT:
                df[target.label_name] = df[target.label_name].astype(str)
            else:
                raise ValueError(f"Unexpected problem type '{target.problem_type}'")

        return df

    def load(self):
        """Load the input data."""
        self.task = TaskDetails.from_json(self.task_details_path)
        self.df_train = self.load_dataset(self.dataset_train_path)
        self.df_val = self.load_dataset(self.dataset_val_path)
        self.df_test = self.load_dataset(self.dataset_test_path)

    def validate(self):
        """Validate the input data."""
        # validate labels
        for df in [self.df_train, self.df_val]:
            if not self.task.target.label_name in df.columns:
                raise ValueError(f"Expected column '{self.task.target.label_name}' in train and validation data")

    def analyze(self):
        """
        Analyze the data.
        Collect unique labels in the training dataset.
        For regression tasks, characterize the distribution of the labels.
        """
        values = self.df_train[self.task.target.label_name].explode().explode()
        self.task.target.values = sorted(values[~values.isna()].unique().tolist())

        if self.task.target.problem_type in [ProblemType.SINGLE_LABEL_REGRESSION, ProblemType.MULTI_LABEL_REGRESSION]:
            self.task.target.mean = self.df_train[self.task.target.label_name].explode().mean()
            self.task.target.std = self.df_train[self.task.target.label_name].explode().std()
            self.task.target.min = self.df_train[self.task.target.label_name].explode().min()
            self.task.target.max = self.df_train[self.task.target.label_name].explode().max()
            self.task.target.skew = self.df_train[self.task.target.label_name].explode().skew()

    @abstractmethod
    def preprocess(self):
        """Preprocess the data."""
        raise NotImplementedError()

    @abstractmethod
    def train(self):
        """Train the model."""
        raise NotImplementedError()

    @abstractmethod
    def predict(self, *, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def save(self, predictions: pd.DataFrame):
        """Save the predictions."""
        self.test_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_json(self.test_predictions_path, orient="records")

    def cast_predictions(self, predictions: dict[str, Any]) -> dict[str, Any]:
        """Cast the predictions.

        The predictions need to satisfy the following:
        - contain a "uid" column that matches the test data.
        - contain a column with the predictions (named after the problem_type).
        - the values in each label column must be valid.

        This function tries to cast the predictions to the expected type.
        """
        predictions = deepcopy(predictions)
        key = self.task.target.prediction_name

        for entry in predictions:
            if self.task.target.problem_type == ProblemType.SINGLE_LABEL_NER:
                # convert predictions to a list of strings
                entry[key] = [str(value) for value in entry[key]]
            elif self.task.target.problem_type == ProblemType.MULTI_LABEL_NER:
                # convert predictions to a list of lists of strings
                entry[key] = [[str(value) for value in value] for value in entry[key]]
            elif self.task.target.problem_type == ProblemType.SINGLE_LABEL_BINARY_CLASSIFICATION:
                # convert predictions to a float
                entry[key] = float(entry[key])
            elif self.task.target.problem_type == ProblemType.MULTI_LABEL_BINARY_CLASSIFICATION:
                # convert predictions to a list of floats
                entry[key] = [float(value) for value in entry[key]]
            elif self.task.target.problem_type == ProblemType.SINGLE_LABEL_MULTI_CLASS_CLASSIFICATION:
                # convert predictions to string
                entry[key] = str(entry[key])
            elif self.task.target.problem_type == ProblemType.MULTI_LABEL_MULTI_CLASS_CLASSIFICATION:
                # convert predictions to a list of strings
                entry[key] = [str(value) for value in entry[key]]
            elif self.task.target.problem_type == ProblemType.SINGLE_LABEL_REGRESSION:
                # convert predictions to a float
                entry[key] = float(entry[key])
            elif self.task.target.problem_type == ProblemType.MULTI_LABEL_REGRESSION:
                # convert predictions to a list of floats
                entry[key] = [float(value) for value in entry[key]]
            elif self.task.target.problem_type == ProblemType.SINGLE_LABEL_TEXT:
                # convert predictions to a string
                entry[key] = str(entry[key])
            else:
                raise ValueError(f"Unexpected problem type '{self.task.target.problem_type}'")

        return predictions

    def verify_predictions(self):
        """Verify the predictions.
        
        The predictions need to satisfy the following:
        - contain a "uid" column that matches the test data.
        - contain a column with the predictions (named after the problem_type).
        - the values in each label column must be valid.
        """
        df_test = self.load_dataset(self.dataset_test_path)
        with open(self.test_predictions_path) as f:
            predictions = json.load(f)
        df = pd.DataFrame(predictions)
        if not "uid" in df.columns:
            raise ValueError("Expected column 'uid' in predictions")
        prediction_uids = set(df["uid"].tolist())
        test_uids = set(df_test["uid"].tolist())
        if len(prediction_uids) != len(df):
            raise ValueError("Expected unique uids in predictions")
        if prediction_uids != test_uids:
            raise ValueError("Expected predictions to contain the same uids as the test data")
        if len(df.columns) != 2:
            raise ValueError(f"Expected 2 columns in predictions, but got {len(df.columns)}: {df.columns}")

        col = self.task.target.prediction_name
        if not col in df.columns:
            raise ValueError(f"Expected column '{col}' in predictions")
        if self.task.target.problem_type == ProblemType.SINGLE_LABEL_NER:
            df_test = df_test.set_index("uid")
            for _, row in df.iterrows():
                # check if all values are a list of strings
                if not isinstance(row[col], list):
                    raise ValueError(f"Expected lists in column '{col}'")
                if not all(isinstance(value, str) for value in row[col]):
                    raise ValueError(f"Expected lists of strings in column '{col}'")
                # check if the number of values matches the number of words in the report
                if len(row[col]) != len(df_test.loc[row.uid, self.task.input_name]):
                    raise ValueError(f"Expected {len(row[self.task.input_name])} values in column '{col}'")
        elif self.task.target.problem_type == ProblemType.MULTI_LABEL_NER:
            df_test = df_test.set_index("uid")
            for _, row in df.iterrows():
                # check if all values are a list of lists of strings
                if not isinstance(row[col], list):
                    raise ValueError(f"Expected lists in column '{col}'")
                if not all(isinstance(value, list) for value in row[col]):
                    raise ValueError(f"Expected list of lists in column '{col}'")
                if not all(isinstance(value, str) for values in row[col] for value in values):
                    raise ValueError(f"Expected list of lists of strings in column '{col}'")
                # check if the number of values matches the number of words in the report
                if len(row[col]) != len(df_test.loc[row.uid, self.task.input_name]):
                    raise ValueError(f"Expected {len(row[self.task.input_name])} values in column '{col}'")
        elif self.task.target.problem_type == ProblemType.SINGLE_LABEL_BINARY_CLASSIFICATION:
            # check if all values are a float
            if not all(isinstance(value, float) for value in df[col]):
                raise ValueError(f"Expected values in column '{col}' to be a float")
            # check if all values are in [0, 1]
            if not all(0 <= value <= 1 for value in df[col]):
                raise ValueError(f"Expected values in column '{col}' to be in [0, 1]")
        elif self.task.target.problem_type == ProblemType.MULTI_LABEL_BINARY_CLASSIFICATION:
            # check if all values are a list of floats
            if not all(isinstance(value, list) for value in df[col]):
                raise ValueError(f"Expected values in column '{col}' to be a list")
            if not all(isinstance(value, float) for value in df[col].explode()):
                raise ValueError(f"Expected values in column '{col}' to be a list of floats")
            # check if all values are in [0, 1]
            if not all(0 <= value <= 1 for value in df[col].explode()):
                raise ValueError(f"Expected values in column '{col}' to be in [0, 1]")
        elif self.task.target.problem_type == ProblemType.SINGLE_LABEL_MULTI_CLASS_CLASSIFICATION:
            # check if all values are string
            if not all(isinstance(value, str) for value in df[col]):
                raise ValueError(f"Expected values in column '{col}' to be a string")
            # check if all values are in the list of possible values
            if not all(value in self.task.target.values for value in df[col]):
                raise ValueError(f"Expected values in column '{col}' to be in {self.task.target.values}")
        elif self.task.target.problem_type == ProblemType.MULTI_LABEL_MULTI_CLASS_CLASSIFICATION:
            # check if all values are a list of strings
            if not all(isinstance(value, list) for value in df[col]):
                raise ValueError(f"Expected values in column '{col}' to be a list")
            if not all(isinstance(value, str) for value in df[col].explode()):
                raise ValueError(f"Expected values in column '{col}' to be a list of strings")
            # check if all values are in the list of possible values
            if not all(value in self.task.target.values for value in df[col].explode()):
                raise ValueError(f"Expected values in column '{col}' to be in {self.task.target.values}")
        elif self.task.target.problem_type == ProblemType.SINGLE_LABEL_REGRESSION:
            # check if all values are a float
            if not all(isinstance(value, float) for value in df[col]):
                raise ValueError(f"Expected values in column '{col}' to be a float")
        elif self.task.target.problem_type == ProblemType.MULTI_LABEL_REGRESSION:
            # check if all values are a list of floats
            if not all(isinstance(value, list) for value in df[col]):
                raise ValueError(f"Expected values in column '{col}' to be a list")
            if not all(isinstance(value, float) for value in df[col].explode()):
                raise ValueError(f"Expected values in column '{col}' to be a list of floats")
        elif self.task.target.problem_type == ProblemType.SINGLE_LABEL_TEXT:
            # check if all values are a string
            if not all(isinstance(value, str) for value in df[col]):
                raise ValueError(f"Expected values in column '{col}' to be a string")
        else:
            raise ValueError(f"Unexpected problem type '{self.task.target.problem_type}'")
