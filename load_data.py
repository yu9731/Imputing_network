import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def pd_time_information(Timestamp):
    pd_time_information = pd.DataFrame(columns=['month', 'day_of_week', 'day_of_month', 'hour'])
    for i in range(len(Timestamp)):
        month = int(str(pd.Timestamp(Timestamp[i]))[5:7])
        day_of_week = pd.Timestamp(Timestamp[i]).dayofweek + 1
        day_of_month = int(str(pd.Timestamp(Timestamp[i]))[8:10])
        hour = int(str(pd.Timestamp(Timestamp[i]))[11:13])
        pd_time_information.loc[i] = [month, day_of_week, day_of_month, hour]
    pd_time_information.index = Timestamp
    return pd_time_information

def filtering_by_hours(train_data, pd_time, Timestamp, save_file=False):
    '''
    Filtering outliers in measured data by the correlation between heat demand and specific hours
    :param df: Dataframe after data pre-processing
    '''
    pd_time_information_index = pd.to_datetime(Timestamp)
    pd_time_information.index = pd_time_information_index
    pd_time['Energy [KW]'] = train_data

    for i in range(24):
        hour = i
        energy_demand = pd_time[pd_time['hour'] == hour]['Energy [KW]'].values
        energy_demand_25 = np.percentile(energy_demand, 25)
        energy_demand_75 = np.percentile(energy_demand, 75)
        energy_demand_median = np.percentile(energy_demand, 50)
        lower_bound = energy_demand_25 - 1.5 * (energy_demand_75 - energy_demand_25)
        upper_bound = energy_demand_75 + 1.5 * (energy_demand_75 - energy_demand_25)
        for j in range(len(energy_demand)):
            if (lower_bound <= energy_demand[j] <= upper_bound) or np.isnan(energy_demand[j]):
                pass
            else:
                outlier_index = pd_time[pd_time['hour'] == hour]['Energy [KW]'].index[j]
                pd_time.loc[outlier_index, 'Energy [KW]'] = np.percentile(energy_demand, 50)

    return pd_time
'''
variable_name = ['AT_SFH', 'temp', 'solarradiation']
pred_hour = 24
windows_length = 24

heat_data = pd.read_csv('AT_SFH.csv', index_col=[0],parse_dates=True) #.astype('float')
heat_data_index = heat_data.index
heat_data_index = pd.to_datetime(heat_data.index,utc=True)
heat_data.index = heat_data_index

pd_train_data = heat_data.loc['2016-01-01 00:00:00':'2017-12-31 23:00:00', variable_name]
pd_test_data = heat_data.loc['2018-01-01 00:00:00':'2018-12-31 23:00:00', variable_name]

# weather_data_test = weather_data_.loc['2018-01-01 00:00:00':'2018-12-31 23:00:00', [f'{region[i]}_temperature', f'{region[i]}_radiation_direct_horizontal']]

pd_time = pd_time_information(pd.to_datetime(pd_train_data.index))
np_hour = pd_time.loc[:, 'hour'].values

pd_time_test = pd_time_information(pd.to_datetime(pd_test_data.index))
np_hour_test = pd_time_test.loc[:, 'hour'].values

for j in range(np_hour.shape[0]):
    np_hour[j] = np.sin(2 * np.pi * np_hour[j] / 24.0)
np_hour = np_hour.reshape((np_hour.shape[0], 1))

for j in range(np_hour_test.shape[0]):
    np_hour_test[j] = np.sin(2 * np.pi * np_hour_test[j] / 24.0)
np_hour_test = np_hour_test.reshape((np_hour_test.shape[0], 1))

train_data = np.hstack((pd_train_data.values, np_hour))
test_data = np.hstack((pd_test_data.values, np_hour_test))

max_data = np.max(train_data)
min_data = np.min(test_data)

np.save(f'train_data/AT_SFH_max.npy', max_data)
np.save(f'train_data/AT_SFH_min.npy', min_data)

for j in range(train_data.shape[1]):
    train_data[:, j] = (train_data[:, j] - min_data) / (max_data - min_data)
    test_data[:, j] = (test_data[:, j] - min_data) / (max_data - min_data)

data_train = np.zeros(((train_data.shape[0]-pred_hour)//windows_length+1, pred_hour, 4))
for j in range((train_data.shape[0]-pred_hour)//windows_length+1):
    idx = j
    for k in range(4):
        data_train[idx, :, k] = train_data[j*windows_length:(j*windows_length+pred_hour), k]
data_train = data_train.reshape((data_train.shape[0], data_train.shape[1], data_train.shape[2], 1))

data_test = np.zeros(((test_data.shape[0]-pred_hour)//windows_length+1, pred_hour, 4))
for j in range((test_data.shape[0]-pred_hour)//windows_length+1):
    idx = j
    for k in range(4):
        data_test[idx, :, k] = test_data[j*windows_length:(j*windows_length+pred_hour), k]
data_test = data_test.reshape((data_test.shape[0], data_test.shape[1], data_test.shape[2], 1))

np.save(f'train_data/train_AT_SFH.npy', data_train)
np.save(f'train_data/test_AT_SFH.npy', data_test)
'''

variable_name = ['L17', 'temp', 'solarradiation']
pred_hour = 24
windows_length = 24

heat_data = pd.read_csv(f'L17.csv', index_col=[0], parse_dates=True) #.astype('float')
heat_data_test = pd.read_csv(f'L17_test.csv', index_col=[0], parse_dates=True) #.astype('float')

heat_data = heat_data.loc[:, 'L17'].astype('float')
heat_data_index = heat_data.index
heat_data_index = pd.to_datetime(heat_data.index)
heat_data.index = heat_data_index

pd_train_data = heat_data.loc['2022-10-03 23:00:00': '2023-03-31 23:00:00', variable_name]
pd_test_data = heat_data.loc['2023-12-01 00:00:00':'2023-12-31 23:00:00', variable_name]

pd_time = pd_time_information(pd.to_datetime(pd_train_data.index))
pd_time = filtering_by_hours(pd_train_data, pd_time, pd_time.index)
np_hour = pd_time.loc[:, 'hour'].values

pd_time_test = pd_time_information(pd.to_datetime(pd_test_data.index))
pd_time_test = filtering_by_hours(pd_test_data, pd_time_test, pd_time_test.index)
np_hour_test = pd_time_test.loc[:, 'hour'].values

train_data = np.hstack((pd_train_data.values, np_hour))
test_data = np.hstack((pd_test_data.values, np_hour_test))

max_data = np.max(train_data)
min_data = np.min(test_data)

np.save(f'train_data/L17_max.npy', max_data)
np.save(f'train_data/L17_min.npy', min_data)

data_train = np.zeros(((train_data.shape[0]-pred_hour)//windows_length+1, pred_hour, 4))
for j in range((train_data.shape[0]-pred_hour)//windows_length+1):
    idx = j
    for k in range(4):
        data_train[idx, :, k] = train_data[j*windows_length:(j*windows_length+pred_hour), k]
data_train = data_train.reshape((data_train.shape[0], data_train.shape[1], data_train.shape[2], 1))

data_test = np.zeros(((test_data.shape[0]-pred_hour)//windows_length+1, pred_hour, 4))
for j in range((test_data.shape[0]-pred_hour)//windows_length+1):
    idx = j
    for k in range(4):
        data_test[idx, :, k] = test_data[j*windows_length:(j*windows_length+pred_hour), k]
data_test = data_test.reshape((data_test.shape[0], data_test.shape[1], data_test.shape[2], 1))

np.save(f'train_data/train_L17.npy', data_train)
np.save(f'train_data/test_L17.npy', data_test)