"""ConvLSTM2D next-frame prediction on synthetic moving squares.

Adapted from the classic Keras example with TensorFlow Keras imports and
reduced default epochs for faster execution.
"""
import argparse
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import BatchNormalization


def generate_movies(n_samples=1200, n_frames=15):
    row = 80
    col = 80
    noisy_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float32)
    shifted_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float32)

    for i in range(n_samples):
        n = np.random.randint(3, 8)

        for _ in range(n):
            xstart = np.random.randint(20, 60)
            ystart = np.random.randint(20, 60)
            directionx = np.random.randint(0, 3) - 1
            directiony = np.random.randint(0, 3) - 1
            w = np.random.randint(2, 4)

            for t in range(n_frames):
                x_shift = xstart + directionx * t
                y_shift = ystart + directiony * t
                noisy_movies[i, t, x_shift - w: x_shift + w, y_shift - w: y_shift + w, 0] += 1

                if np.random.randint(0, 2):
                    noise_f = (-1) ** np.random.randint(0, 2)
                    noisy_movies[i, t, x_shift - w - 1: x_shift + w + 1, y_shift - w - 1: y_shift + w + 1, 0] += noise_f * 0.1

                x_shift = xstart + directionx * (t + 1)
                y_shift = ystart + directiony * (t + 1)
                shifted_movies[i, t, x_shift - w: x_shift + w, y_shift - w: y_shift + w, 0] += 1

    noisy_movies = noisy_movies[::, ::, 20:60, 20:60, ::]
    shifted_movies = shifted_movies[::, ::, 20:60, 20:60, ::]
    noisy_movies[noisy_movies >= 1] = 1
    shifted_movies[shifted_movies >= 1] = 1
    return noisy_movies.astype(np.float32), shifted_movies.astype(np.float32)


def build_model():
    seq = Sequential()
    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), input_shape=(None, 40, 40, 1), padding='same', return_sequences=True))
    seq.add(BatchNormalization())
    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
    seq.add(BatchNormalization())
    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
    seq.add(BatchNormalization())
    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
    seq.add(BatchNormalization())
    seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3), activation='sigmoid', padding='same', data_format='channels_last'))
    seq.compile(loss='binary_crossentropy', optimizer='adadelta')
    return seq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=1200)
    parser.add_argument("--n_frames", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=10)
    args = parser.parse_args()

    noisy_movies, shifted_movies = generate_movies(n_samples=args.n_samples, n_frames=args.n_frames)
    model = build_model()
    model.fit(noisy_movies[:1000], shifted_movies[:1000], batch_size=args.batch_size, epochs=args.epochs, validation_split=0.05, verbose=2)

    which = 1004
    track = noisy_movies[which][:7, ::, ::, ::]
    for _ in range(16):
        new_pos = model.predict(track[np.newaxis, ::, ::, ::, ::], verbose=0)
        new = new_pos[::, -1, ::, ::, ::]
        track = np.concatenate((track, new), axis=0)

    # No plotting for simplicity; `track` contains predictions
    print("Prediction sequence shape:", track.shape)


if __name__ == '__main__':
    main()


