import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow import keras
import tensorflow as tf
from keras.layers import Embedding, Dense, LSTM, Dropout, Bidirectional, TextVectorization, GlobalAveragePooling1D, GRU, Conv1D, GlobalMaxPooling1D, Input, Concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.saving import load_model
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import sys
from parser_types_util import type_keras_nonreal
os.system("clear")

def setup_arguments():
    parser = argparse.ArgumentParser(prog="evaluate_model", description="Evaluate a trained model for review recognition.")
    
    parser.add_argument(
        '-m',
        '--mode',
        type=str.lower,
        choices=["single", "csv"],
        required=True,
        help="Mode of the evaluator. You can either provide a single review to test or a spreadsheet of reviews."
    )
    
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        help="input CSV or string to evaluate the model.",
        required=True
    )
    
    parser.add_argument(
        '-im',
        '--input-model',
        type=str,
        help="The model to evaluate",
        required=True
    )
    return parser.parse_args()

def main():
    global args
    args = setup_arguments()
    if args.mode == "csv": 
        df = pd.read_csv(args.input)
        x_test = df["x_test"].values
        y_test = df["y_test"].values
        
        model = load_model(args.input_model)
        loss, accuracy = model.evaluate(x_test, y_test)
        print(f"Model evaluated on training dataset:\nAccuracy = {accuracy}\nLoss = {loss}")
    elif args.mode == "single":
        model = load_model(args.input_model)
        val = model.predict(tf.constant([args.input]), verbose=0)[0][0]
        print(val)


if __name__ == "__main__":
    main()