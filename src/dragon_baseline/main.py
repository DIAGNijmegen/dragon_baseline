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

import re
import subprocess
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, softmax
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification,
                          AutoModelForTokenClassification, AutoTokenizer,
                          TokenClassificationPipeline)
from transformers.modeling_outputs import SequenceClassifierOutput

from dragon_baseline.architectures.clf_multi_head import \
    AutoModelForMultiHeadSequenceClassification
from dragon_baseline.architectures.reg_multi_head import \
    AutoModelForMultiHeadSequenceRegression
from dragon_baseline.nlp_algorithm import NLPAlgorithm, ProblemType

__all__ = [
    # expose the algorithm classes
    "AutoModelForMultiHeadSequenceClassification",
    "AutoModelForMultiHeadSequenceRegression",
]

class CustomLogScaler(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.standard_scaler = StandardScaler()

    def fit(self, X, y=None):
        # Apply log(1 + value) transformation
        log_X = np.log1p(X)

        # Fit the standard scaler
        self.standard_scaler.fit(log_X)
        return self

    def transform(self, X, y=None):
        # Apply log(1 + value) transformation
        log_X = np.log1p(X)

        # Transform using standard scaler
        return self.standard_scaler.transform(log_X)

    def inverse_transform(self, X):
        # Reverse the standard scaling
        inv_X = self.standard_scaler.inverse_transform(X)

        # Reverse the log(1 + value) transformation
        return np.expm1(inv_X)


def merge_overlapping_labels(tags: list[list[str]]) -> list[list[str]]:
    """Function to merge overlapping labels"""
    merged_tags = []
    for tag_group in tags:
        if len(tag_group) > 1:
            # Split each tag into its prefix (B/I) and entity
            split_tags = [tag.split('-', 1) for tag in tag_group]
            # Group tags by their prefixes
            prefix_groups = {}
            for prefix, entity in split_tags:
                if prefix in prefix_groups:
                    prefix_groups[prefix].append(entity)
                else:
                    prefix_groups[prefix] = [entity]
            # Merge entities with the same prefix
            merged_group = [f'{prefix}-{"[MERGED]".join(entities)}' for prefix, entities in prefix_groups.items()]
            merged_tags.append(merged_group)
        else:
            merged_tags.append(tag_group)
    return merged_tags


def split_merged_labels(tags: list[list[str]]) -> list[list[str]]:
    """Function to split merged labels"""
    split_tags = []
    for tag_group in tags:
        split_group = []
        for tag in tag_group:
            if tag == "O":
                split_group.append(tag)
                continue

            # Split each tag into its prefix (B/I) and entity
            prefix, entity = tag.split('-', 1)
            # Split entities that were merged
            if '[MERGED]' in entity:
                split_entities = entity.split('[MERGED]')
                for split_entity in split_entities:
                    split_group.append(f'{prefix}-{split_entity}')
            else:
                split_group.append(tag)
        split_tags.append(split_group)
    return split_tags


def select_entity_labels(df: pd.DataFrame, entity_number: str, label_name: str) -> pd.DataFrame:
    df = df.copy()

    # select the labels for the current entity number
    df[label_name] = df[label_name].apply(lambda labels: [
        [lbl.replace(f"-{entity_number}-", "-") for lbl in token_labels if re.match(f"^[BI]-{entity_number}-", lbl)]
        for token_labels in labels
    ])

    # merge the labels for the current entity number if they overlap
    labels_before = df[label_name].explode().explode()
    labels_before = labels_before[labels_before.notna()].unique()
    df[label_name] = df[label_name].apply(merge_overlapping_labels)
    labels_after = df[label_name].explode().explode()
    labels_after = labels_after[labels_after.notna()].unique()
    labels_added = set(labels_after) - set(labels_before)
    if labels_added:
        print(f"Merged overlapping labels, {labels_before} -> {labels_after}")

    if not df[label_name].apply(lambda labels: all(len(token_labels) <= 1 for token_labels in labels)).all():
        raise ValueError("Expected all samples to have exactly one label per token after selecting the entity labels")

    df[label_name] = df[label_name].apply(lambda labels: [
        (token_labels[0] if len(token_labels) > 0 else "O")
        for token_labels in labels
    ])
    df["block_number"] = entity_number
    return df


def split_multi_label_ner_samples(df: pd.DataFrame, entity_numbers: List[str], label_name: str) -> pd.DataFrame:
    """
    Splits multi-label NER samples in a DataFrame based on entity numbers.

    When labels overlap even after splitting based on entity number (i.e. multiple labels for the same token), the labels are merged.

    Args:
        df (pd.DataFrame): The DataFrame containing the NER samples.
        entity_numbers (List[str]): The list of entity numbers to split the samples.
        label_name (str): The name of the label column in the DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with the multi-label samples split based on entity numbers.
    """
    return pd.concat([
        select_entity_labels(df=df, entity_number=entity_number, label_name=label_name)
        for entity_number in entity_numbers
    ], ignore_index=True)


def balance_negative_samples(df: pd.DataFrame, label_name: str, seed: int) -> pd.DataFrame:
    """Select an equal number of samples with a labeled entity, as ones with no labeled entity"""
    mask = df[label_name].apply(lambda labels: any(any(lbl != "O" for lbl in token_labels) for token_labels in labels))

    if len(df) > 2 * mask.sum():
        # select a random subset of samples without an entity
        df = pd.concat([df[mask], df[~mask].sample(n=mask.sum(), random_state=seed)], ignore_index=True)

    return df


class DragonBaseline(NLPAlgorithm):
    def __init__(self, input_path: Path = Path("/input"), output_path: Path = Path("/output"), workdir: Path = Path("/opt/app"), model_name: Union[str, Path] = "distilbert-base-multilingual-cased", **kwargs):
        """
        Baseline implementation for the DRAGON Challenge (https://dragon.grand-challenge.org/).
        This baseline uses the HuggingFace Transformers library (https://huggingface.co/transformers/).

        The baseline must implement the following methods:
        - `preprocess`: preprocess the data
        - `train`: train the model
        - `predict`: predict the labels for the test data
        """
        super().__init__(input_path=input_path, output_path=output_path, **kwargs)

        # default training settings
        self.model_name = model_name
        self.per_device_train_batch_size = 4
        self.gradient_accumulation_steps = 2
        self.gradient_checkpointing = False
        self.max_seq_length = 512
        self.learning_rate = 1e-5
        self.num_train_epochs = 5
        self.warmup_ratio = 0.1
        self.load_best_model_at_end = True
        self.metric_for_best_model = "loss"
        self.fp16 = True
        self.create_strided_training_examples = True

        # paths for saving the preprocessed data and model checkpoints
        self.nlp_dataset_train_preprocessed_path = Path(workdir / "nlp-dataset-train-preprocessed.json")
        self.nlp_dataset_val_preprocessed_path = Path(workdir / "nlp-dataset-val-preprocessed.json")
        self.nlp_dataset_test_preprocessed_path = Path(workdir / "nlp-dataset-test-preprocessed.json")
        self.model_save_dir = Path(workdir / "checkpoints")

        # keep track of the common prefix of the reports, to remove it
        self.common_prefix = None

    @staticmethod
    def longest_common_prefix(strs: List[str]) -> str:
        if not strs:
            return ""

        # Assume the first string is the longest common prefix
        prefix = strs[0]

        # Compare the prefix with each string
        for s in strs:
            while not s.startswith(prefix):
                # Reduce the prefix by one character at a time
                prefix = prefix[:-1]
                if not prefix:
                    return ""
        return prefix
    
    @staticmethod
    def longest_common_prefix_tokenized(strs: List[List[str]]) -> List[str]:
        """Find the longest common prefix of a list of tokenized strings."""
        if not strs or not strs[0]:
            return []

        # Assume the first string is the longest common prefix
        prefix = strs[0]

        # Compare the prefix with each string
        for s in strs:
            # Check each token in the prefix
            for i in range(len(prefix)):
                # If the current string is shorter than the prefix length or tokens don't match
                if i >= len(s) or prefix[i] != s[i]:
                    # Reduce the prefix length
                    prefix = prefix[:i]
                    break

            if not prefix:
                return []

        return prefix


    def remove_common_prefix_from_reports(self):
        """Remove the common prefix from the reports."""
        # find the common prefix
        if self.task.input_name == "text":
            reports = self.df_train[self.task.input_name].to_list()
            self.common_prefix = self.longest_common_prefix(reports)

            if not self.common_prefix:
                return

            # remove the common prefix
            print(f"Removing common prefix from all reports: {self.common_prefix}")
            for df in [self.df_train, self.df_val, self.df_test]:
                df[self.task.input_name] = df[self.task.input_name].apply(lambda x: re.sub(f"^{self.common_prefix}", "", x))
        elif self.task.input_name == "text_parts":
            reports = self.df_train[self.task.input_name].to_list()
            self.common_prefix = self.longest_common_prefix_tokenized(reports)

            if not self.common_prefix:
                return

            # remove the common prefix
            print(f"Removing common prefix from all reports: {self.common_prefix}")
            for df in [self.df_train, self.df_val, self.df_test]:
                df[self.task.input_name] = df[self.task.input_name].apply(lambda x: x[len(self.common_prefix):])
                if self.task.target.label_name in df.columns:
                    df[self.task.target.label_name] = df[self.task.target.label_name].apply(lambda x: x[len(self.common_prefix):])

    def scale_labels(self) -> pd.DataFrame:
        """Scale the labels."""
        if self.task.target.problem_type in [ProblemType.SINGLE_LABEL_REGRESSION, ProblemType.MULTI_LABEL_REGRESSION]:
            if self.task.target.skew > 1:
                scaler = CustomLogScaler()
            else:
                scaler = StandardScaler()

            # fit the scaler on the training data
            scaler = scaler.fit(self.df_train[self.task.target.label_name].explode().values.astype(float).reshape(-1, 1))
            self.label_scalers[self.task.target.label_name] = scaler

            # scale the labels
            if self.task.target.problem_type == ProblemType.SINGLE_LABEL_REGRESSION:
                self.df_train[self.task.target.label_name] = scaler.transform(self.df_train[self.task.target.label_name].values.reshape(-1, 1))
                self.df_val[self.task.target.label_name] = scaler.transform(self.df_val[self.task.target.label_name].values.reshape(-1, 1))
            elif self.task.target.problem_type == ProblemType.MULTI_LABEL_REGRESSION:
                self.df_train[self.task.target.label_name] = self.df_train[self.task.target.label_name].apply(lambda x: np.ravel(scaler.transform(np.array(x).reshape(-1, 1))))
                self.df_val[self.task.target.label_name] = self.df_val[self.task.target.label_name].apply(lambda x: np.ravel(scaler.transform(np.array(x).reshape(-1, 1))))

    def unscale_predictions(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Unscale the predictions."""
        if self.task.target.problem_type in [ProblemType.SINGLE_LABEL_REGRESSION, ProblemType.MULTI_LABEL_REGRESSION]:
            scaler = self.label_scalers[self.task.target.label_name]
            if self.task.target.problem_type == ProblemType.SINGLE_LABEL_REGRESSION:
                predictions[self.task.target.prediction_name] = scaler.inverse_transform(predictions[self.task.target.prediction_name].values.reshape(-1, 1))
            elif self.task.target.problem_type == ProblemType.MULTI_LABEL_REGRESSION:
                predictions[self.task.target.prediction_name] = predictions[self.task.target.prediction_name].apply(lambda x: np.ravel(scaler.inverse_transform(np.array(x).reshape(-1, 1))))
        return predictions

    def add_dummy_test_labels(self):
        """Add dummy labels for test data. This allows to use the dataset in the huggingface pipeline."""
        if self.task.target.problem_type in [ProblemType.SINGLE_LABEL_NER, ProblemType.MULTI_LABEL_NER]:
            train_labels = self.df_train[self.task.target.label_name].explode()
            dummy_label = train_labels[~train_labels.isna()].iloc[0]
            self.df_test[self.task.target.label_name] = self.df_test.apply(lambda row: [dummy_label]*len(row[self.task.input_name]), axis=1)
        else:
            dummy_label = self.df_train[self.task.target.label_name].iloc[0]
            self.df_test[self.task.target.label_name] = [dummy_label]*len(self.df_test)

    def prepare_labels_for_huggingface(self):
        """
        Prepare labels for training in the HuggingFace pipeline.

        For multi-label binary classification tasks, convert the list of 0/1 ints to list of "labelX" strings.
        For multi-label regression and multi-class classification tasks, convert the multiple values in the label column to one value per column.
        For mutli-label NER tasks, split the samples into one sample per entity group.
        """
        if self.task.target.problem_type == ProblemType.MULTI_LABEL_BINARY_CLASSIFICATION:
            for df in [self.df_train, self.df_val, self.df_test]:
                # convert list of 0/1 ints to list of "labelX" strings
                df[self.task.target.label_name] = df[self.task.target.label_name].apply(lambda labels: [
                    f"label{i}" for i, lbl in enumerate(labels) if lbl == 1
                ])
            # set the possible values (exclude NaN, which we get from `.explode()` for empty lists)
            self.task.target.values = sorted([lbl for lbl in self.df_train[self.task.target.label_name].explode().unique() if isinstance(lbl, str)])
        elif self.task.target.problem_type in [
            ProblemType.MULTI_LABEL_REGRESSION,
            ProblemType.MULTI_LABEL_MULTI_CLASS_CLASSIFICATION,
        ]:
            num_labels = len(self.df_train[self.task.target.label_name].iloc[0])
            for df in [self.df_train, self.df_val, self.df_test]:
                for i in range(num_labels):
                    df[f"{self.task.target.label_name}_{i}"] = df[self.task.target.label_name].apply(lambda x: x[i])
        elif self.task.target.problem_type == ProblemType.MULTI_LABEL_NER:
            # collect all unique labels and remove the "B-" and "I-" prefixes
            self.task.target.values = sorted(set([
                re.sub(r"^[BI]-", "", lbl)
                for lbl in self.df_train[self.task.target.label_name].explode().explode().unique()
                if lbl != "O"
            ]))

            if not all(re.match(r"^\d+-.+$", lbl) for lbl in self.task.target.values):
                raise ValueError(f"Expected all labels to start with a number followed by a dash, but got {self.task.target.values}")

            # split the samples into one sample per entity group
            entity_numbers = sorted(set([lbl.split("-")[0] for lbl in self.task.target.values]))
            self.df_train = split_multi_label_ner_samples(df=self.df_train, entity_numbers=entity_numbers, label_name=self.task.target.label_name)
            self.df_val = split_multi_label_ner_samples(df=self.df_val, entity_numbers=entity_numbers, label_name=self.task.target.label_name)
            self.df_test = split_multi_label_ner_samples(df=self.df_test, entity_numbers=entity_numbers, label_name=self.task.target.label_name)

            # select an equal number of samples with a labeled entity, as ones with no labeled entity
            self.df_train = balance_negative_samples(df=self.df_train, label_name=self.task.target.label_name, seed=self.task.jobid)
            self.df_val = balance_negative_samples(df=self.df_val, label_name=self.task.target.label_name, seed=self.task.jobid)

            # add block number to the text parts
            for df in [self.df_train, self.df_val, self.df_test]:
                df[self.task.input_name] = df.apply(lambda row: [row["block_number"]] + row[self.task.input_name], axis=1)
                df[self.task.target.label_name] = df[self.task.target.label_name].apply(lambda labels: ["O"] + labels)

    def shuffle_train_data(self):
        """Shuffle the training data."""
        self.df_train = self.df_train.sample(frac=1, random_state=self.task.jobid)

    def preprocess(self):
        """Preprocess the data."""
        # prepare the reports
        self.remove_common_prefix_from_reports()

        # prepare the labels
        self.scale_labels()
        self.add_dummy_test_labels()
        self.prepare_labels_for_huggingface()
        self.shuffle_train_data()

    def train(self):
        """Train the model."""
        # save the preprocessed data for training through command line interface of the HuggingFace library
        for path in [
            self.nlp_dataset_train_preprocessed_path,
            self.nlp_dataset_val_preprocessed_path,
            self.nlp_dataset_test_preprocessed_path,
        ]:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.df_train.to_json(self.nlp_dataset_train_preprocessed_path, orient="records")
        self.df_val.to_json(self.nlp_dataset_val_preprocessed_path, orient="records")
        self.df_test.to_json(self.nlp_dataset_test_preprocessed_path, orient="records")

        # load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, truncation_side=self.task.recommended_truncation_side)
        tokenizer.model_max_length = self.max_seq_length  # set the maximum sequence length, if not already set

        # train the model
        if self.task.target.problem_type in [ProblemType.SINGLE_LABEL_NER, ProblemType.MULTI_LABEL_NER]:
            trainer = "ner"
        elif self.task.target.problem_type in [ProblemType.MULTI_LABEL_REGRESSION, ProblemType.MULTI_LABEL_MULTI_CLASS_CLASSIFICATION]:
            trainer = "multi_label_classification"
        else:
            trainer = "classification"

        cmd = [
            "python", "-m", "dragon_baseline",
            trainer,
            "--do_train",
            "--do_eval",
            "--do_predict",
            "--learning_rate", self.learning_rate,
            "--model_name_or_path", self.model_name,
            "--ignore_mismatched_sizes",
            "--num_train_epochs", self.num_train_epochs,
            "--warmup_ratio", self.warmup_ratio,
            "--max_seq_length", self.max_seq_length,
            "--truncation_side", self.task.recommended_truncation_side,
            "--load_best_model_at_end", self.load_best_model_at_end,
            "--save_strategy", "epoch",
            "--eval_strategy", "epoch",
            "--per_device_train_batch_size", self.per_device_train_batch_size,
            "--gradient_accumulation_steps", self.gradient_accumulation_steps,
            "--gradient_checkpointing", self.gradient_checkpointing,
            "--train_file", self.nlp_dataset_train_preprocessed_path,
            "--validation_file", self.nlp_dataset_val_preprocessed_path,
            "--test_file", self.nlp_dataset_test_preprocessed_path,
            "--output_dir", self.model_save_dir,
            "--overwrite_output_dir",
            "--save_total_limit", "2",
            "--seed", self.task.jobid,
            "--report_to", "none",
            "--text_column_name" + ("s" if not "ner" in trainer else ""), self.task.input_name,
            "--remove_columns", "uid",
        ]
        if self.task.target.problem_type in [
            ProblemType.MULTI_LABEL_REGRESSION,
            ProblemType.MULTI_LABEL_MULTI_CLASS_CLASSIFICATION,
        ]:
            label_names = [col for col in self.df_train.columns if col.startswith(f"{self.task.target.label_name}_")]
            cmd.extend([
                "--label_column_names", ",".join(label_names),
            ])
        else:
            cmd.extend([
                "--label_column_name", self.task.target.label_name,
            ])
        if self.task.target.problem_type in [ProblemType.SINGLE_LABEL_NER, ProblemType.MULTI_LABEL_NER]:
            if self.create_strided_training_examples:
                cmd.append("--create_strided_training_examples")
        else:
            cmd.extend([
                "--text_column_delimiter", tokenizer.sep_token,
            ])
        if self.metric_for_best_model is not None:
            cmd.extend([
                "--metric_for_best_model", str(self.metric_for_best_model),
            ])
        if self.fp16:
            cmd.append("--fp16")

        cmd = [str(arg) for arg in cmd]
        print("Training command:")
        print(" ".join(cmd))
        subprocess.check_call(cmd)

    def predict_ner(self, *, df: pd.DataFrame) -> pd.DataFrame:
        """Predict the labels for the test data.

        The pipeline below returns a list of dictionaries, one for each entity. An example is shown below:
        result = [
            {'entity_group': 'SYMPTOM', 'score': 0.25475845, 'word': 'persistent cough', 'start': 22, 'end': 38},
            {'entity_group': 'DIAGNOSIS', 'score': 0.49139544, 'word': 'likely viral infection', 'start': 39, 'end': 61},
        ]

        We convert this to a list of labels, one for each word. An example is shown below:
        prediction = [B-SYMPTOM, I-SYMPTOM, B-DIAGNOSIS, I-DIAGNOSIS, I-DIAGNOSIS]
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_save_dir, truncation_side=self.task.recommended_truncation_side)
        tokenizer.model_max_length = self.max_seq_length  # set the maximum sequence length, if not already set
        model = AutoModelForTokenClassification.from_pretrained(self.model_save_dir)
        classifier = TokenClassificationPipeline(
            model=model,
            tokenizer=tokenizer,
            stride=tokenizer.model_max_length // 2,
            aggregation_strategy="first",
            device=self.device,
        )

        results = []
        for _, row in tqdm(df.iterrows(), desc="Predicting", total=len(df)):
            # predict
            inputs = row[self.task.input_name]
            text = " ".join(inputs)
            result = classifier(text)

            # convert to one label per word
            start, end = 0, 0
            prediction = []
            for word in inputs:
                end = start + len(word)
                found_match = False
                for entity_group in result:
                    if entity_group["start"] <= start <= entity_group["end"]:
                        if end > entity_group["end"]:
                            # Even with `aggregation_strategy="first"` a word can span multiple entities. This happens when
                            # the word is split into multiple tokens but not merged back by the `TokenClassificationPipeline`.
                            # This happens for example with dates, e.g. "01/01/2021". We take the prediction for the first part.
                            # Feel free to implement a better solution!
                            pass

                        BI_tag = "B" if start == entity_group["start"] else "I"
                        prediction.append(BI_tag + "-" + entity_group["entity_group"])
                        found_match = True
                        break

                if not found_match:
                    prediction.append("O")
                start = end + 1

            # add "O" tokens for the tokens that were removed
            prediction = ["O"] * len(self.common_prefix) + prediction

            pred = {
                self.task.target.prediction_name: prediction
            }
            results.append({"uid": row["uid"], **pred})
        return pd.DataFrame(results)

    def predict_multi_label_ner(self, *, df: pd.DataFrame) -> pd.DataFrame:
        """Predict the labels for the test data."""
        # predict the labels for each entity group
        predictions = self.predict_ner(df=df)
        predictions["block_number"] = df["block_number"]

        results = []
        for uid, group in predictions.groupby("uid"):
            prediction = []

            # combine the predictions for each entity group
            for _, row in group.iterrows():
                label_list = [re.sub("^(B|I)-", r"\1-" + row["block_number"] + "-", lbl) for lbl in row[self.task.target.prediction_name]]
                prediction.append(label_list)

            # remove the prediction for the block number
            prediction = np.array(prediction).transpose()[1:].tolist()

            # convert to list of labels per token
            prediction = [(["O"] if set(token_labels) == {"O"} else sorted(set(token_labels) - {"O"})) for token_labels in prediction]
            prediction = split_merged_labels(prediction)

            results.append({"uid": uid, self.task.target.prediction_name: prediction})

        return pd.DataFrame(results)

    def predict_huggingface(self, *, df: pd.DataFrame) -> pd.DataFrame:
        """Predict the labels for the test data."""
        # load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_save_dir, truncation_side=self.task.recommended_truncation_side)
        tokenizer.model_max_length = self.max_seq_length  # set the maximum sequence length, if not already set
        if self.task.target.problem_type == ProblemType.MULTI_LABEL_REGRESSION:
            model = AutoModelForMultiHeadSequenceRegression.from_pretrained(self.model_save_dir).to(self.device)
        elif self.task.target.problem_type == ProblemType.MULTI_LABEL_MULTI_CLASS_CLASSIFICATION:
            model = AutoModelForMultiHeadSequenceClassification.from_pretrained(self.model_save_dir).to(self.device)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(self.model_save_dir).to(self.device)

        # predict
        results = []
        for _, row in tqdm(df.iterrows(), desc="Predicting", total=len(df)):
            # tokenize inputs
            inputs = row[self.task.input_name] if self.task.input_name == "text_parts" else [row[self.task.input_name]]
            tokenized_inputs = tokenizer(*inputs, return_tensors="pt", truncation=True).to(self.device)

            # predict
            result: SequenceClassifierOutput = model(**tokenized_inputs)

            if self.task.target.problem_type == ProblemType.MULTI_LABEL_MULTI_CLASS_CLASSIFICATION:
                logits: List[np.ndarray] = [logits.detach().cpu().numpy() for logits in result.logits]
            else:
                logits: np.ndarray = result.logits.detach().cpu().numpy()

            # convert to labels
            if self.task.target.problem_type == ProblemType.SINGLE_LABEL_REGRESSION:
                expected_shape = (1, 1)
                if logits.shape != expected_shape:
                    raise ValueError(f"Expected logits to have shape {expected_shape}, but got {logits.shape}")

                prediction = {self.task.target.prediction_name: logits[0][0]}
            elif self.task.target.problem_type == ProblemType.MULTI_LABEL_REGRESSION:
                expected_shape = (1, len(self.df_train[self.task.target.label_name].iloc[0]))
                if logits.shape != expected_shape:
                    raise ValueError(f"Expected logits to have shape {expected_shape}, but got {logits.shape}")

                prediction = {self.task.target.prediction_name: logits[0]}
            elif self.task.target.problem_type == ProblemType.SINGLE_LABEL_BINARY_CLASSIFICATION:
                expected_shape = (1, 2)
                if logits.shape != expected_shape:
                    raise ValueError(f"Expected logits to have shape {expected_shape}, but got {logits.shape}")

                # calculate sigmoid to map the logits to [0, 1]
                prediction = softmax(logits, axis=-1)[0, 1]
                prediction = {self.task.target.prediction_name: prediction}
            elif self.task.target.problem_type == ProblemType.SINGLE_LABEL_MULTI_CLASS_CLASSIFICATION:
                expected_shape = (1, len(self.task.target.values))
                if logits.shape != expected_shape:
                    raise ValueError(f"Expected logits to have shape {expected_shape}, but got {logits.shape}")

                p = model.config.id2label[np.argmax(logits[0])]
                prediction = {self.task.target.prediction_name: p}
            elif self.task.target.problem_type == ProblemType.MULTI_LABEL_BINARY_CLASSIFICATION:
                expected_shape = (1, len(self.task.target.values))
                if logits.shape != expected_shape:
                    raise ValueError(f"Expected logits to have shape {expected_shape}, but got {logits.shape}")

                prediction = expit(logits)[0]  # calculate sigmoid to map the logits to [0, 1]
                prediction = {self.task.target.prediction_name: prediction}
            elif self.task.target.problem_type == ProblemType.MULTI_LABEL_MULTI_CLASS_CLASSIFICATION:
                expected_length = len(self.df_train[self.task.target.label_name].iloc[0])
                if len(logits) != expected_length:
                    raise ValueError(f"Expected logits to have length {expected_length}, but got {len(logits)}")
                label_names = [f"{self.task.target.label_name}_{i}" for i in range(len(logits))]
                for logits_, label_name in zip(logits, label_names):
                    expected_shape = (1, len(self.df_train[label_name].unique()))
                    if logits_.shape != expected_shape:
                        raise ValueError(f"Expected logits to have shape {expected_shape}, but got {logits_.shape}")

                preds = [np.argmax(p) for p in logits]
                prediction = {
                    self.task.target.prediction_name: [
                        id2label[str(p)]
                        for p, id2label in zip(preds, model.config.id2labels)
                    ]
                }
            else:
                raise ValueError(f"Unexpected problem type '{self.task.target.problem_type}'")

            results.append({"uid": row["uid"], **prediction})

        df_pred = pd.DataFrame(results)

        # scale the predictions (inverse of the normalization during preprocessing)
        df_pred = self.unscale_predictions(df_pred)

        return df_pred

    def predict(self, *, df: pd.DataFrame) -> pd.DataFrame:
        """Predict the labels for the test data."""
        with torch.no_grad():
            if self.task.target.problem_type == ProblemType.SINGLE_LABEL_NER:
                return self.predict_ner(df=df)
            elif self.task.target.problem_type == ProblemType.MULTI_LABEL_NER:
                return self.predict_multi_label_ner(df=df)
            else:
                return self.predict_huggingface(df=df)


if __name__ == "__main__":
    # Note: to debug (outside of Docker), you can set the input and output paths.
    for job_name in [
        "Task101_Example_sl_bin_clf-fold0",
        "Task102_Example_sl_mc_clf-fold0",
        "Task103_Example_mednli-fold0",
        "Task104_Example_ml_bin_clf-fold0",
        "Task105_Example_ml_mc_clf-fold0",
        "Task106_Example_sl_reg-fold0",
        "Task107_Example_ml_reg-fold0",
        "Task108_Example_sl_ner-fold0",
        "Task109_Example_ml_ner-fold0",
    ]:
        DragonBaseline(
            input_path=Path(f"test-input/{job_name}"),
            output_path=Path(f"test-output/{job_name}"),
            workdir=Path(f"test-workdir/{job_name}"),
        ).process()
