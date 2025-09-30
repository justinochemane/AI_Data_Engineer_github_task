import argparse
import os
import sys
import time
from typing import Tuple

import numpy as np

try:
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception as import_err:  # pragma: no cover
    raise SystemExit(
        "TensorFlow is required. Please install with: pip install tensorflow"
    ) from import_err


def _manual_load_reuters(max_features: int):
    path = keras.utils.get_file(
        "reuters.npz",
        origin="https://storage.googleapis.com/tensorflow/tf-keras-datasets/reuters.npz",
    )
    with np.load(path, allow_pickle=True) as f:
        keys = set(list(f.keys()))
        expected = {"x_train", "y_train", "x_test", "y_test"}
        if expected.issubset(keys):
            x_train, y_train = f["x_train"], f["y_train"]
            x_test, y_test = f["x_test"], f["y_test"]
        elif {"x", "y"}.issubset(keys):
            # Older archive format with combined arrays. Use Keras canonical split sizes.
            x_all, y_all = f["x"], f["y"]
            num_train = 8982  # per Keras dataset
            x_train, y_train = x_all[:num_train], y_all[:num_train]
            x_test, y_test = x_all[num_train:], y_all[num_train:]
        else:
            raise RuntimeError(
                "Unexpected contents in reuters.npz. Found keys: "
                f"{sorted(keys)}. Please delete the cached file at: {path} "
                "and rerun to redownload."
            )
    if max_features is not None:
        x_train = [[w for w in seq if w < max_features] for seq in x_train]
        x_test = [[w for w in seq if w < max_features] for seq in x_test]
    return (x_train, y_train), (x_test, y_test)


def load_reuters_dataset(
    max_features: int,
    maxlen: int,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], int]:
    """Load Reuters via manual .npz path to avoid allow_pickle issues entirely."""
    (x_train, y_train), (x_test, y_test) = _manual_load_reuters(max_features)
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
    num_classes = int(np.max(y_train)) + 1
    return (x_train, y_train), (x_test, y_test), num_classes


def build_model(
    cell_type: str,
    max_features: int,
    embed_dim: int,
    rnn_units: int,
    sequence_length: int,
    dropout: float,
    num_classes: int,
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
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    model = keras.Model(inputs, outputs, name=f"reuters_{cell_type_lower}")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
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
    parser = argparse.ArgumentParser(description="Compare SimpleRNN, GRU, LSTM on Reuters topics classification.")
    parser.add_argument("--max_features", type=int, default=20000, help="Vocabulary size")
    parser.add_argument("--maxlen", type=int, default=200, help="Sequence length (time steps)")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--rnn_units", type=int, default=64, help="Hidden units for RNN cells")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per model")
    parser.add_argument(
        "--models",
        type=str,
        default="SimpleRNN,GRU,LSTM",
        help="Comma-separated list among SimpleRNN,GRU,LSTM",
    )
    args = parser.parse_args()

    print("Loading Reuters dataset...")
    (x_train, y_train), (x_test, y_test), num_classes = load_reuters_dataset(args.max_features, args.maxlen)
    print(f"Train: {x_train.shape}, Test: {x_test.shape}, Classes: {num_classes}")

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
            num_classes=num_classes,
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
    os.environ.setdefault("PYTHONHASHSEED", "0")
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)


