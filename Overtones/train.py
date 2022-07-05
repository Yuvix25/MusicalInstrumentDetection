import os

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Conv1D, MaxPooling1D
from keras.losses import SparseCategoricalCrossentropy

from tqdm import tqdm
from utils import *


AUDIO_LENGTH = 4 # seconds
WINDOW_SIZE = 0.25 # seconds
AMPLITUDE_THRESHOLD = 0.15 # precentage of highest amplitude


class InstrumentDetector:
    def __init__(self):
        self.labels = None

    def load_data(self, folder):
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.wav')]

        x_data = []
        y_data = []
        labels = [] if self.labels == None else self.labels
        for f in tqdm(files):
            windows = preprocess(*load_wav(f, 1, 1+AUDIO_LENGTH), WINDOW_SIZE, AMPLITUDE_THRESHOLD)
            # curr_x_data = [[amps for amps in window.values()] for window in windows if any((window[i] != 0 and i != 0) for i in window)]
            # x_data += curr_x_data
            x_data.append([[amps for amps in window.values()] for window in windows])
            label = filename_to_instrument(f)
            # y_data += [label] * len(curr_x_data)
            y_data.append(label)
            if label not in labels:
                labels.append(label)
        
        y_data = list(map(lambda x: labels.index(x), y_data))
        y_data = to_categorical(y_data, len(labels))
        return tf.convert_to_tensor(x_data), tf.convert_to_tensor(y_data), labels
    
    def create_dataset(self, train_dir, test_dir):
        self.x_data, self.y_data, self.labels = self.load_data(train_dir)
        self.test_x_data, self.test_y_data, _ = self.load_data(test_dir)
        self.dataset = tf.data.Dataset.from_tensor_slices((self.x_data, self.y_data)).shuffle(len(self.x_data))
        self.test_dataset = tf.data.Dataset.from_tensor_slices((self.test_x_data, self.test_y_data)).shuffle(len(self.test_x_data))
    
    def train(self, epochs=10, batch_size=32): # train classifier
        print(self.x_data.shape, self.y_data.shape)
        print(self.test_x_data.shape, self.test_y_data.shape)
        # 1D:

        # self.model = Sequential([
        #     Conv1D(16, 3, activation='relu', input_shape=(self.x_data.shape[1], 1)),
        #     MaxPooling1D(2),
        #     Conv1D(32, 3, activation='relu'),
        #     MaxPooling1D(2),
        #     Conv1D(32, 3, activation='relu'),
        #     MaxPooling1D(2),
        #     Flatten(),
        #     Dense(16, activation='relu'),
        #     Dense(len(self.labels), activation='softmax')
        # ])

        # 2D:
        self.model = Sequential([
            Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(self.x_data.shape[1], self.x_data.shape[2], 1)),
            MaxPooling2D(2),
            Conv2D(32, (3, 3), padding='same', activation='relu'),
            MaxPooling2D(2),
            Conv2D(32, (3, 3), padding='same', activation='relu'),
            MaxPooling2D(2),
            Flatten(),
            Dense(16, activation='relu'),
            Dense(len(self.labels), activation='softmax')
        ])

        # self.model.summary()

        self.model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
        self.history = self.model.fit(self.dataset.batch(batch_size), epochs=epochs)

    def evaluate(self):
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.show()

        test_loss, test_acc = self.model.evaluate(self.test_x_data, self.test_y_data)
        print('Test accuracy:', format(test_acc, '.2%'))

    def predict(self, x):
        return self.model.predict(x)
    
    def save(self):
        name = f'mode_W{WINDOW_SIZE}_F{AMPLITUDE_THRESHOLD}'
        if (name in os.listdir('models')):
            i = 1
            while (name + f'_{i}' in os.listdir('models')):
                i += 1
            name += f'_{i}'
        self.model.save(f'models/{name}.h5')

if __name__ == '__main__':
    from test import test

    detector = InstrumentDetector()
    detector.create_dataset('audio/dataset', 'audio/test')
    detector.train()
    detector.evaluate()
    detector.save()
    test(detector)