import os
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def preprocessed_df():
    data_path = os.path.join("data", "preprocessed.csv")
    assert os.path.exists(data_path), "Expected preprocessed dataset at data/preprocessed.csv. Run preprocess.py first."
    return pd.read_csv(data_path)


def test_preprocessed_file_exists(preprocessed_df):
    assert not preprocessed_df.empty, "Preprocessed dataset should not be empty."


def test_target_column_present(preprocessed_df):
    assert "target" in preprocessed_df.columns, "Preprocessed dataset must include the target column."


def test_expected_row_count(preprocessed_df):
    assert len(preprocessed_df) == 150, "Iris dataset should have 150 rows after preprocessing."
