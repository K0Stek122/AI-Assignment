import pytest
from unittest.mock import patch
import os
from pathlib import Path
import sys

# PLEASE keep in mind these tests will always throw 20 warnings because of tensorflow compatibility and update warnings.

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from train_model import main

class TestTrainModel:
    def test_dense_model_training(self):
        path = Path("trained_models/pytest_dense_model.keras")
        if path.exists():
            os.remove(path)
        testargs = [
            "train_model.py",
            "-o",
            str(path),
            "-i",
            "dataset/TestReviews.csv",
            "-nt",
            "BASIC",
            "-e",
            "1"
        ]

        with patch.object(sys, 'argv', testargs):
            main()
        assert path.exists()

    def test_gru_model_training(self):
        path = Path("trained_models/pytest_gru_model.keras")
        if path.exists():
            os.remove(path)
        testargs = [
            "train_model.py",
            "-o",
            str(path),
            "-i",
            "dataset/TestReviews.csv",
            "-nt",
            "GRU",
            "-e",
            "1"
        ]

        with patch.object(sys, 'argv', testargs):
            main()
        assert path.exists()
    
    def test_lstm_model_training(self):
        path = Path("trained_models/pytest_lstm_model.keras")
        if path.exists():
            os.remove(path)
        testargs = [
            "train_model.py",
            "-o",
            str(path),
            "-i",
            "dataset/TestReviews.csv",
            "-nt",
            "LSTM",
            "-e",
            "1"
        ]

        with patch.object(sys, 'argv', testargs):
            main()
        assert path.exists()