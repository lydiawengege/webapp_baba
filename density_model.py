from tensorflow import keras
import numpy as np


def predict_gas_density(t, p, yco2):
    x1 = np.array((float(t) - 50)/(250 - 50)).reshape((1, 1))
    x2 = np.array((float(p) - 100) / (350 - 100)).reshape((1, 1))
    x3 = np.array((float(yco2) - 0.9) / (1 - 0.9)).reshape((1, 1))
    x = np.concatenate([x1, x2, x3], axis=1)
    model = keras.models.load_model('models_density/BDENG')
    output = model.predict(x) * (864.8884 - 276.0645) + 276.0645
    return output[0][0]


def predict_liquid_density(t, p, xco2):
    x1 = np.array((float(t) - 30)/(170 - 30)).reshape((1, 1))
    x2 = np.array((float(p) - 100) / (370 - 100)).reshape((1, 1))
    x3 = np.array(float(xco2) / 0.030754).reshape((1, 1))
    x = np.concatenate([x1, x2, x3], axis=1)
    model = keras.models.load_model('models_density/BDENW')
    output = model.predict(x)
    output = output * (1197.856 - 916.0229) + 916.0229
    return output[0][0]
