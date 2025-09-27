import pandas as pd
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

building = 'AT_SFH'
pred_hour = 24
length = 24 * 365
windows_length = 24
variable_num = 4
temporal_num = 1
epoch = 200
cnt = 0

def test(building):
    generator = load_model(f'Baseline_Model/generator_{building}.h5')

    test_for_plot = []
    ori_test = []

    def get_points_1():
        points = []
        mask = []
        for i in range(1):
            m = np.zeros((pred_hour, variable_num, 1), dtype=np.uint8)  # 24,4,1
            x1 = np.random.randint(0, pred_hour - pred_hour + 1, 1)

            m[:, -2] = 1
            mask.append(m)
        return np.array(mask)

    for i in range((length - pred_hour) // windows_length + 1):
        data_1 = np.load(f'train_data/test_{building}.npy')
        data_1 = data_1[i]

        data_1 = data_1.reshape(1, pred_hour, variable_num, 1)
        mask_batch = get_points_1()
        generator_input_1 = data_1 * (1 - mask_batch)

        test_pred_1 = generator.predict(generator_input_1)

        for j in range(test_pred_1.shape[1]):
            test_for_plot.append(test_pred_1[0][j][variable_num - temporal_num - 1])
            ori_test.append(data_1[0][j][variable_num - temporal_num - 1])

    max = np.load(f'train_data/{building}_max.npy')
    min = np.load(f'train_data/{building}_min.npy')

    test_for_plot = np.array(test_for_plot) * (max - min) + min
    ori_test = np.array(ori_test) * (max - min) + min

    nrmse = np.sqrt(np.mean((np.array(test_for_plot) - np.array(ori_test)) ** 2)) / (
            np.max(np.array(ori_test)) - np.min(np.array(ori_test)))

    plt.plot(test_for_plot, label='Pred')
    plt.plot(ori_test, label='Ori')
    plt.legend()
    plt.show()

    return nrmse

nrmse = test(building)
print(nrmse)

