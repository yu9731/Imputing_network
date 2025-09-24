import pandas as pd
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

building_lst = ['AT_SFH']

pred_hour = 24
length = 24*365
windows_length = 24
variable_num = 4
temporal_num = 1
epoch = 200
cnt = 0

def test(building):
    generator = load_model(f'generator_after_discriminator_{building}.h5')
    
    test_for_plot = []
    ori_test = []

    def get_points():
        points = []
        mask = []
        for i in range(1):
            m = np.zeros((pred_hour, variable_num, 1), dtype=np.uint8)  # 1D
            x1 = np.random.randint(0, pred_hour - pred_hour + 1, 1)

            m[:, variable_num-temporal_num-1] = 1
            mask.append(m)
        return np.array(mask)

    for i in range((length-pred_hour)//windows_length+1):
        data = np.load(f'train_data/{pred_hour}/task_{building}_data_solar_test_{pred_hour}.npy')
        data = data.reshape((data.shape[0], data.shape[1], data.shape[2]))
        data = data[i]

        data = data.reshape(1, pred_hour, variable_num, 1)
        mask_batch = get_points_1()
        generator_input = data * (1 - mask_batch)

        test_pred = generator.predict(generator_input)

        for j in range(test_pred_baseline.shape[1]):
            if pred_hour == 24:
                test_for_plot.append(test_pred[0][j][variable_num-temporal_num-1][0])
                ori_test.append(data[0][j][variable_num-temporal_num-1])
            else:
                if i == 0:
                    test_for_plot.append(test_pred[0][j][variable_num-temporal_num-1][0])
                    ori_test.append(data[0][j][variable_num-temporal_num-1][0])
                else:
                    if j >= (pred_hour - 24):
                        test_for_plot.append(test_pred[0][j][variable_num-temporal_num-1][0])
                        ori_test.append(data[0][j][variable_num-temporal_num-1][0])
                    else:
                        pass
    nrmse = np.sqrt(np.mean((np.array(test_for_plot)-np.array(ori_test))**2)) / (np.max(np.array(ori_test)) - np.min(np.array(ori_test)))

    min_data = np.load(f'{building}_min.npy')
    max_data = np.load(f'{building}_max.npy')
    test_for_plot = np.array(test_for_plot) * (max_data - min_data) + min_data
    ori_test = np.array(ori_test) * (max_data - min_data) + min_data

    plt.plot(test_for_plot, label='Pred')
    plt.plot(ori_test, label='Ori')
    plt.legend()
    plt.show()

    return nrmse

nrmse = test(building, 'global', dilation=True, UNet=True)
print(nrmse)
