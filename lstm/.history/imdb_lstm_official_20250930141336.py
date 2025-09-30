"""Trains an LSTM model on the IMDB sentiment classification task.

This is a close adaptation of the classic Keras example, with minor
adjustments for TensorFlow Keras imports and a lower default epoch count
to keep runtime reasonable.
"""
from __future__ import print_function

import argparse

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.datasets import imdb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_features", type=int, default=20000)
    parser.add_argument("--maxlen", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--units", type=int, default=128)
    args = parser.parse_args()

    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=args.max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=args.maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=args.maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Build model...')
    model = Sequential()
    model.add(Embedding(args.max_features, 128))
    model.add(LSTM(args.units, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print('Train...')
    model.fit(
        x_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(x_test, y_test),
        verbose=2,
    )
    score, acc = model.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=0)
    print('Test score:', score)
    print('Test accuracy:', acc)


if __name__ == '__main__':
    main()


