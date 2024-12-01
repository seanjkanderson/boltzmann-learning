
import numpy as np
from tensorflow.keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import pandas as pd


class LossHistory(Callback):
    def __init__(self):
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))


class StreamingLSTM:
    def __init__(self, measurement_set, action_set, finite_measurement: bool, lstm_units=50, window_size=1, lag=10):
        self.measurement_set = measurement_set
        self.action_set = action_set
        self.window_size = window_size
        self.lag = lag
        self.finite_measurement = finite_measurement
        self.model = Sequential()
        self.model.add(LSTM(units=lstm_units, stateful=True, return_sequences=False))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

        self.encoding_map = self.one_hot_encoding(measurement_set)
        self.action_set = action_set
        self.action_encoding_map = self.one_hot_encoding(action_set)
        self.finite_measurement = finite_measurement
        self.n_act = 1 # len(action.set)
        self.model.build(input_shape=(1, self.window_size, (len(measurement_set)+len(action_set))))
        self.trained = False
        self.inputs = []
        self.outputs = []
        self.loss_history = LossHistory()

    @staticmethod
    def one_hot_encoding(measurement_set):
        n_elements = len(measurement_set)
        one_hot = np.eye(n_elements)
        encoding_map = dict()
        for idx, meas in enumerate(measurement_set):
            encoding_map[meas] = one_hot[idx]
        return encoding_map

    def measurement_to_embedding(self, measurement, action):
        one_hot_action = self.action_encoding_map[action]
        if not self.finite_measurement:
            embedding = np.array([measurement[k] for k in self.measurement_set])
        else:
            embedding = self.encoding_map[measurement]
        return np.hstack((embedding, one_hot_action))

    def reset_states(self):
        """Reset the states of the LSTM layer."""
        self.model.reset_states()

    def update_energies(self, measurement, costs: dict, time, **kwargs):
        m = 0
        n = 1
        for action, cost in costs.items():
            embedding = self.measurement_to_embedding(measurement, action)
            self.inputs.append(embedding)
            self.outputs.append(cost)

        self.inputs = self.inputs[-self.window_size*self.n_act:]
        self.outputs = self.outputs[-self.window_size*self.n_act:]
        if time % self.window_size == 0 and time > 0:
            print(time)
            features = np.array(self.inputs)
            N = features.shape[0]  # Total number of samples
            k = N - (self.lag + m + n)  # TODO: parameterize
            in_slice = np.array([range(i, i + self.lag) for i in range(k)])
            op_slice = np.array([range(i + self.lag + m, i + self.lag + m + n) for i in range(k)])

            features = features[in_slice, :]

            outputs = np.array(self.outputs)
            outputs = outputs[op_slice]
            self.model.fit(x=features, y=outputs, epochs=1, batch_size=1, callbacks=[self.loss_history])
            self.trained = True

    def get_action(self, measurement, time: float = 0.0, **kwargs):
        """
        Make predictions with the model.

        :param data: The input data for prediction, expected shape is (1, 1, input_dim).
        :return: The predicted value.
        """

        if self.trained:
            # one_hot_measurement = self.encoding_map[measurement]
            min_cost = 1e10
            best_action = ''
            for action in self.action_set:
                inputs = self.inputs[-self.lag + 1:]
                embedding = self.measurement_to_embedding(measurement, action)
                self.inputs.append(embedding)
                inputs.append(embedding)
                total_input = np.vstack(inputs)[np.newaxis]
                predicted_cost = self.model.predict(total_input, batch_size=3, verbose=False)[0, 0]
                if predicted_cost < min_cost:
                    best_action = action
                    min_cost = predicted_cost
        else:
            best_action = self.action_set[0]

        return best_action, None, None

# if __name__ == '__main__':
#     # Example usage:
#     # Assuming your input feature dimension is 3:
#     input_dim = 3
#     a_set = [0, 1]
#     m_set = [-1, -2]
#     model = StreamingLSTM(input_shape=(1, input_dim))
#
#     # Simulate streaming data:
#     import numpy as np
#     for t in range(100):
#         new_data = np.random.random((1, 1, input_dim))  # Example new data
#         target = np.random.random((1, 1))               # Example target value
#         model.update_model(new_data, target)
#         prediction = model.predict(new_data)
#         print(f'Time step {t}, Prediction: {prediction}')
