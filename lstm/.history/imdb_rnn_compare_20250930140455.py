import argparse
import os
import sys
import time
from typing import Tuple

import numpy as np

try:
    # Prefer TensorFlow Keras (widely available)
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception as import_err:  # pragma: no cover
    raise SystemExit(
        "TensorFlow is required. Please install with: pip install tensorflow"
    ) from import_err


def load_imdb_dataset(max_features: int, maxlen: int) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features)
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
    return (x_train, y_train), (x_test, y_test)


def build_model(
    cell_type: str,
    max_features: int,
    embed_dim: int,
    rnn_units: int,
    sequence_length: int,
    dropout: float,
) -> keras.Model:
    inputs = keras.Input(shape=(sequence_length,), dtype="int32")
    x = layers.Embedding(max_features, embed_dim, name="embedding")(inputs)

    cell_type_lower = cell_type.lower()
    if cell_type_lower == "simplernn":
        x = layers.SimpleRNN(rnn_units, dropout=dropout, recurrent_dropout=0.0, name="rnn")(x)
    elif cell_type_lower == "gru":
        x = layers.GRU(rnn_units, dropout=dropout, recurrent_dropout=0.0, name="gru")(x)
    elif cell_type_lower == "lstm":
        x = layers.LSTM(rnn_units, dropout=dropout, recurrent_dropout=0.0, name="lstm")(x)
    else:
        raise ValueError(f"Unsupported cell_type: {cell_type}. Use one of: SimpleRNN, GRU, LSTM")

    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="classifier")(x)

    model = keras.Model(inputs, outputs, name=f"imdb_{cell_type_lower}")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_and_evaluate(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    epochs: int,
) -> Tuple[float, float]:
    start = time.time()
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        verbose=2,
    )
    elapsed = time.time() - start
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    return acc, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SimpleRNN, GRU, LSTM on IMDB sentiment.")
    parser.add_argument("--max_features", type=int, default=20000, help="Vocabulary size")
    parser.add_argument("--maxlen", type=int, default=200, help="Sequence length (time steps)")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--rnn_units", type=int, default=64, help="Hidden units for RNN cells")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Epochs per model")
    parser.add_argument(
        "--models",
        type=str,
        default="SimpleRNN,GRU,LSTM",
        help="Comma-separated list among SimpleRNN,GRU,LSTM",
    )
    args = parser.parse_args()

    print("Loading IMDB dataset...")
    (x_train, y_train), (x_test, y_test) = load_imdb_dataset(args.max_features, args.maxlen)
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    results = []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        print("\n" + "=" * 80)
        print(f"Training model: {name}")
        model = build_model(
            cell_type=name,
            max_features=args.max_features,
            embed_dim=args.embed_dim,
            rnn_units=args.rnn_units,
            sequence_length=args.maxlen,
            dropout=args.dropout,
        )
        acc, elapsed = train_and_evaluate(
            model, x_train, y_train, x_test, y_test, args.batch_size, args.epochs
        )
        print(f"Model {name} test accuracy: {acc:.4f} (time: {elapsed:.1f}s)")
        results.append((name, acc, elapsed))

    print("\nSummary (higher accuracy is better):")
    for name, acc, elapsed in results:
        print(f"- {name:10s}  acc={acc:.4f}  time={elapsed:.1f}s")


if __name__ == "__main__":
    # Ensure deterministic behavior where possible (may not be perfect on some backends)
    os.environ.setdefault("PYTHONHASHSEED", "0")
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)


