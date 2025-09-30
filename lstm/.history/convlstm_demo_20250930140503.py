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


def generate_moving_squares_dataset(
    num_sequences: int,
    timesteps: int,
    frame_size: int = 40,
    square_size: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    # Binary classification: does the square move left-to-right (label 1) or top-to-bottom (label 0)?
    rng = np.random.default_rng(1234)
    x = np.zeros((num_sequences, timesteps, frame_size, frame_size, 1), dtype=np.float32)
    y = np.zeros((num_sequences,), dtype=np.int32)
    for i in range(num_sequences):
        horizontal = rng.random() < 0.5
        y[i] = 1 if horizontal else 0
        if horizontal:
            row = rng.integers(low=0, high=frame_size - square_size)
            col = rng.integers(low=0, high=frame_size - square_size - timesteps)
            for t in range(timesteps):
                frame = np.zeros((frame_size, frame_size), dtype=np.float32)
                frame[row : row + square_size, col + t : col + t + square_size] = 1.0
                x[i, t, :, :, 0] = frame
        else:
            col = rng.integers(low=0, high=frame_size - square_size)
            row = rng.integers(low=0, high=frame_size - square_size - timesteps)
            for t in range(timesteps):
                frame = np.zeros((frame_size, frame_size), dtype=np.float32)
                frame[row + t : row + t + square_size, col : col + square_size] = 1.0
                x[i, t, :, :, 0] = frame
    return x, y


def build_convlstm_model(
    timesteps: int,
    frame_size: int,
    filters: int,
    kernel_size: int,
    dropout: float,
) -> keras.Model:
    inputs = keras.Input(shape=(timesteps, frame_size, frame_size, 1))
    x = layers.ConvLSTM2D(
        filters=filters,
        kernel_size=(kernel_size, kernel_size),
        padding="same",
        return_sequences=False,
        dropout=dropout,
        name="convlstm",
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="moving_squares_convlstm")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal ConvLSTM2D demo using synthetic moving squares")
    parser.add_argument("--num_sequences", type=int, default=1000)
    parser.add_argument("--timesteps", type=int, default=10)
    parser.add_argument("--frame_size", type=int, default=40)
    parser.add_argument("--filters", type=int, default=32)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    print("Generating dataset...")
    x, y = generate_moving_squares_dataset(
        num_sequences=args.num_sequences,
        timesteps=args.timesteps,
        frame_size=args.frame_size,
    )
    # Train/val/test split
    num_train = int(0.7 * len(x))
    num_val = int(0.15 * len(x))
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train : num_train + num_val], y[num_train : num_train + num_val]
    x_test, y_test = x[num_train + num_val :], y[num_train + num_val :]

    model = build_convlstm_model(
        timesteps=args.timesteps,
        frame_size=args.frame_size,
        filters=args.filters,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    )
    model.summary()

    start = time.time()
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=2,
    )
    elapsed = time.time() - start

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f} (time: {elapsed:.1f}s)")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)


