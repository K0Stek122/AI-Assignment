import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow import keras
import tensorflow as tf
from keras.layers import Embedding, Dense, LSTM, Dropout, Bidirectional, TextVectorization, GlobalAveragePooling1D, GRU, Conv1D, GlobalMaxPooling1D, Input, Concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import sys
os.system("clear")

# TODO - Add transformer if time permits

def type_csv(value):
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError("Provided CSV does not exist.")
    if not path.is_file():
        raise argparse.ArgumentTypeError("Provided CSV is not a file.")
    if path.suffix.lower() != ".csv":
        raise argparse.ArgumentTypeError("Provided CSV is not a CSV file.")
    return path

def type_csv_nonreal(value):
    path = Path(value)
    if not path.suffix.lower() != ".csv":
        raise argparse.ArgumentTypeError("Provided CSV is not a CSV file.")
    return path
    
def type_keras(value):
    path = Path(value)
    if not path.parent.exists():
        raise argparse.ArgumentTypeError("Provided output path does not exist.")
    if path.suffix.lower() != ".keras":
        raise argparse.ArgumentTypeError("Output must end in .keras")
    return path

def setup_arguments():
    parser = argparse.ArgumentParser(prog="train_model", description="Train the model for recognising negative and positive reviews") 
    
    parser.add_argument(
        '-o',
        '--output',
        type=type_keras,
        help="where to output the trained model",
        required=True
    )
    
    parser.add_argument(
        '-i',
        '--input',
        type=type_csv,
        help="Input dataset with two columns labelled 'review' and 'class', see parse_data for details.",
        required=True
    )
    
    parser.add_argument(
        '-v',
        '--vocab-size',
        type=int,
        help="Unique features of the model",
        default=38552
    )

    parser.add_argument(
        '-ml',
        '--max-len',
        type=int,
        help="Maximum review length.",
        default=300
    )
    
    parser.add_argument(
        '-ed',
        '--embed-dim',
        type=int,
        help="embedding dimension. How many numbers describe each word. If it's too small we cannot describe any word, if it's too large we  cause overfitting.",
        default=128
    )
    
    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        help="Sets the epochs (iterations) that the ML will train.",
        default=5
    )
    
    parser.add_argument(
        '-nt',
        '--neural-network-type',
        type=str.upper,
        choices=["DENSE", "LSTM", "GRU", "CNN"],
        required=True)
    
    parser.add_argument(
        '-to',
        '--test-output',
        type=type_csv_nonreal,
        help="Outputs the test dataset to a specified CSV file. Otherwise discards test dataset."
    )

    return parser.parse_args()


def prepare_train_test_data(df, split=0.2, r_state=42):
    x = df["review"]
    y = df["class"]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split, random_state=r_state)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

def setup_text_vectorisation():
    return TextVectorization(
        max_tokens=args.vocab_size,
        output_mode="int",
        output_sequence_length=args.max_len
    )

def create_dense_model(vectoriser):
    #ReLU in the hidden layer to learn nonlinear features cheaply (avoids vanishing gradients), then a sigmoid on the final 1-unit layer to squash outputs for binary classification probability
    return keras.Sequential([
    vectoriser,
    Embedding(input_dim=args.vocab_size, output_dim=args.embed_dim), #Embedding creates a sequence of vectors for each sample
    GlobalAveragePooling1D(), # Pooling limits each sample to one vector.
    Dense(units=64, activation="relu"),
    Dropout(0.2),
    Dense(units=1, activation="sigmoid")
    ])

def create_gru_model(vectoriser):
    return keras.Sequential([
        vectoriser,
        Embedding(input_dim=args.vocab_size, output_dim=args.embed_dim), # No pooling needed here due to GRU already being a sequence reducer that reads word vectors one by one.
        GRU(64, dropout=0.2),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])

def create_lstm_model(vectoriser):
    return keras.Sequential([
        vectoriser,
        Embedding(input_dim=args.vocab_size, output_dim=args.embed_dim), # No pooling needed here due to GRU already being a sequence reducer that reads word vectors one by one.
        Bidirectional(LSTM(64, dropout=0.2)),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])

def create_cnn_model(vectoriser):
    # This model is built with a Functional API. That is, it parallelises training and trains...
    # n-length bigrams simultaneously.
    inputs = Input(shape=(1,), dtype="string")
    x = vectoriser(inputs=inputs)
    x = Embedding(args.vocab_size, args.embed_dim)(x)
    conv_3 = Conv1D(args.embed_dim, 3, activation='relu')(x)
    conv_3 = GlobalMaxPooling1D()(conv_3)
    
    conv_4 = Conv1D(args.embed_dim, 4, activation='relu')(x)
    conv_4 = GlobalMaxPooling1D()(conv_4)

    conv_5 = Conv1D(args.embed_dim, 5, activation='relu')(x)
    conv_5 = GlobalMaxPooling1D()(conv_5)
    
    x = Concatenate()([conv_3, conv_4, conv_5])
    outputs = Dense(1, activation="sigmoid")(x)
    
    return Model(inputs, outputs)

def setup_early_stopping():
    return EarlyStopping(
        monitor='accuracy',
        patience=2,
        restore_best_weights=True
    )

def main():
    global args
    args = setup_arguments()
    
    df = pd.read_csv(args.input)
    if df.empty:
        print(f"[ERROR] the provided dataset does not exist")
        sys.exit(1)

    x_train, x_test, y_train, y_test = prepare_train_test_data(df)

    vectoriser = setup_text_vectorisation()
    vectoriser.adapt(x_train)

    model = None

    if args.neural_network_type == "DENSE":
        model = create_dense_model(vectoriser)
    elif args.neural_network_type == "GRU":
        model = create_gru_model(vectoriser)
    elif args.neural_network_type == "LSTM":
        model = create_gru_model(vectoriser)
    elif args.neural_network_type == "CNN":
        model = create_cnn_model(vectoriser)
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=32,
        callbacks=[setup_early_stopping()]
    )
    
    if args.test_output:
        df = pd.DataFrame()
        df["review"] = x_test
        df["class"] = y_test
        df.to_csv(args.test_output)

    if model is None:
        print(f"[ERROR] model is NULL, something went terribly wrong!")
        sys.exit(1)
    else:
        model.save(args.output)


if __name__ == "__main__":
    main()

# df = open_csv("dataset/TestReviews.csv")
# x_train, x_test, y_train, y_test = prepare_train_test_data(df)

# vectoriser = setup_text_vectorisation()
# vectoriser.adapt(x_train)


# model = create_model(vectoriser)
# model.compile(
#     optimizer='adam',
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )
# model.fit(x_train, y_train, epochs=5, batch_size=32)
# loss, accuracy = model.evaluate(x_test, y_test)

# print(f"Model Trained.\n Accuracy = {accuracy}\nLoss = {loss}")

# print(model.predict(tf.constant(["This product is great! Love it."])))