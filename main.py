import os
from tensorflow import keras
import tensorflow as tf
from keras.layers import Embedding, Dense, LSTM, Dropout, Bidirectional, TextVectorization, GlobalAveragePooling1D
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys


vocab_size = 38552 # Unique features of the model
max_len = 300 # Review length. WIll be truncated / Padded to match this length.
embed_dim = 128 # How many numbers describe each word. If it's too small we can't describe any word, if it's too large we cause Overfitting.

def open_csv(filename : str):
    if not filename.endswith('.csv'):
        return pd.DataFrame()
    if not os.path.exists(filename):
        return pd.DataFrame()
    df = pd.read_csv(filename)
    return df

def get_data(df):
    x = df["review"]
    y = df["class"]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

df = open_csv("dataset/TestReviews.csv")
x_train, x_test, y_train, y_test = get_data(df)

vectoriser = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=max_len
)

vectoriser.adapt(x_train)

#ReLU in the hidden layer to learn nonlinear features cheaply (avoids vanishing gradients), then a sigmoid on the final 1-unit layer to squash outputs for binary classification probability
model = keras.Sequential([
    vectoriser,
    Embedding(input_dim=vocab_size, output_dim=embed_dim),
    GlobalAveragePooling1D(),
    Dense(units=64, activation="relu"),
    Dense(units=1, activation="sigmoid")
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=5, batch_size=32)

loss, accuracy = model.evaluate(x_test, y_test)

print(f"Model Trained.\n Accuracy = {accuracy}\nLoss = {loss}")

print(model.predict(tf.constant(["This product is great! Love it."])))