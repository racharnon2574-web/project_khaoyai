import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler


def create_sequences(data, window=12):

    X, y = [], []

    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])

    return np.array(X), np.array(y)


def run_lstm(train_ts, test_ts):

    scaler = MinMaxScaler()

    full_series = np.concatenate(
        [train_ts.values, test_ts.values]
    ).reshape(-1,1)

    full_scaled = scaler.fit_transform(full_series)

    train_size = len(train_ts)
    train_scaled = full_scaled[:train_size]

    window = 12

    X_train, y_train = create_sequences(train_scaled, window)
    X_train = X_train.reshape((X_train.shape[0], window, 1))

    model = Sequential()
    model.add(LSTM(32, activation="tanh", input_shape=(window,1)))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(
        monitor="loss",
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        epochs=200,
        batch_size=8,
        verbose=0,
        callbacks=[early_stop]
    )

    predictions = []
    current_batch = train_scaled[-window:].reshape(1, window, 1)

    for _ in range(len(test_ts)):

        pred = model.predict(current_batch, verbose=0)[0]
        predictions.append(pred)

        current_batch = np.append(
            current_batch[:,1:,:],
            [[pred]],
            axis=1
        )

    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1,1)
    )

    return predictions.flatten()