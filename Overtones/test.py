# load classifier.h5 and run tests:

import tensorflow as tf
import numpy as np
from keras.models import load_model
from train import InstrumentDetector
import time





def test(model, test_dir=None):
    if (type(model) == str):
        detector = InstrumentDetector()
        detector.create_dataset('audio/test', test_dir)
        model = load_model(model)
    else:
        detector = model
    
    x_data = detector.test_x_data.numpy()
    y_data = detector.test_y_data.numpy()
    for _ in range(10):
        indices = np.random.choice(len(x_data), size=min(10, len(x_data)), replace=False)
        inp = x_data[indices]

        s = time.perf_counter()
        pred = model.predict(inp)
        print((time.perf_counter() - s) / 10)

        pred = list(map(lambda x: list(x).index(max(x)), pred))
        actual = list(map(lambda x: list(x).index(max(x)), y_data[indices]))
        pred = map(lambda x: detector.labels[x], pred)
        actual = map(lambda x: detector.labels[x], actual)

        print(list(zip(actual, pred)))


if __name__ == '__main__':
    test('models/model_W0.25_F0.15_88.h5', 'audio/test2')