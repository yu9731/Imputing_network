# Imputing_network
Network for paper

#### This github repository is used to reproduce the results of the model reconstructed for missing values in heating load data with context encoder, including three parts:
#### load_data.py: Normalizing and Preprocessing the training data and model the training data and test data with sliding window
#### model.py: Modeling, compiling and training the context encoder and saving the trained model
#### test.py: Validating with test data

# Data

#### The data provided here for reproducing the results are from When2Heat Profiles and Niš datasets. Due to the data protection requirements of Green Fusion GmbH in Berlin, we can only provide the electrical load data here to make the results of the TCN-BiLSTM-CE reproducible.

## Dependecies

To avoid unnecessary trouble, the author recommends that you install the following version of the library:

tensorflow == 2.15.0

keras == 2.15.0

numpy == 1.26.4

pandas == 2.1.1

matplotlib == 3.6.3