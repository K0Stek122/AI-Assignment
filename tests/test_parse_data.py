import pytest
import sys
from unittest.mock import patch
from pathlib import Path
from parse_data import *

# Add root directory to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

class TestParseData:
    def test_setup_arguments(self):
        testargs = ["parse_data.py", "-v", "testfile"]
        with patch.object(sys, 'argv', testargs):
            args = setup_arguments()
            assert args != None

    def test_open_csv_improper_filename(self):
        testargs = ["parse_data.py", "not_existing_file"]
        with patch.object(sys, 'argv', testargs):
            df = open_csv(testargs[1])
            assert df.empty == True

    def test_open_csv_proper_filename(self):
        testargs = ["parse_data.py", "dataset/TestReviews.csv"]
        with patch.object(sys, 'argv', testargs):
            df = open_csv(testargs[1])
            assert df.empty == False
            
    def test_download_nltk_packages(self):
        testargs = ["parse_data.py", "dataset/TestReviews.csv", "-cc"]
        with patch.object(sys, 'argv', testargs):
            args = setup_arguments()
            download_nltk_packages(args)

    def test_tokenise(self):
        testargs = ["parse_data.py", "dataset/TestReviews.csv", "-t"]
        with patch.object(sys, 'argv', testargs):
            args = setup_arguments()
            download_nltk_packages(args)
            df = open_csv(testargs[1])
            word_count = tokenise(df, args)
            assert type(word_count) == list
            assert len(word_count) > 0
    def test_sentiment_analysis(self):
        testargs = ["parse_data.py", "dataset/TestReviews.csv", "-t"]
        with patch.object(sys, "argv", testargs):
            args = setup_arguments()
            download_nltk_packages(args)
            df = open_csv(testargs[1])
            word_count = tokenise(df, args)
            out = sentiment_analysis(args, word_count)
            assert len(out) == 3
            assert type(out) == tuple
            for i in range(0, 2):
                assert len(out[i]) > 0
                assert type(out[i]) == list