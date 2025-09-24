import numpy as np
import pandas as pd
import os
import tracemalloc
import time

import tensorflow as tf
from tensorflow.keras.layers import Input
from keras import models
from keras.models import load_model

class MissingData2DModel:
    def __init__(self, building, variable_num, x_train, input_dim):
        self.building = building
        self.variable_num = variable_num
        self.x_train = x_train

        self.input_dim = input_dim
        self.vector_size = input_dim
        self.local_size = input_dim

        self.batch_size = 32
        self.lr = 0.0001
        self.pre_trained_epoch = 100

        self.model_save_path = f'Baseline_Model'
    def build_autoencoder(self):
        '''
        Build an 2D contextual autoencoder model for multi-variante missing data imputation
        :return: contextual autoencoder model, which will serve as a generator for pre-training
        '''
        encoder_input = Input(shape=(self.input_dim, self.variable_num, 1))
        x = tf.keras.layers.Conv2D(128, 3, (2, 2), padding='same', name="Encoder_1")(encoder_input)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.Conv2D(256, 3, (2, 2), padding='same', name="Encoder_2")(x)
        x = tf.keras.activations.relu(x)

        x = tf.keras.layers.Conv2D(256, 3, dilation_rate=(1, 1), padding='same', name="Feature_1")(x)
        x = tf.keras.layers.Conv2D(256, 3, dilation_rate=(2, 2), padding='same', name="Feature_2")(x)
        x = tf.keras.layers.Conv2D(256, 3, dilation_rate=(4, 4), padding='same', name="Feature_3")(x)  # 4,3
        x = tf.keras.layers.Conv2D(256, 3, dilation_rate=(8, 8), padding='same', name="Feature_4")(x)  # 8,3

        x = tf.keras.layers.Conv2DTranspose(128, 3, (2, 2), padding='same', name="Decoder_1")(x)
        x = tf.keras.activations.relu(x)

        x = tf.keras.layers.Conv2DTranspose(1, 3, (2, 2), padding='same', name="Decoder_3")(x)
        decoder_output = tf.keras.activations.tanh(x)

        generator = models.Model(encoder_input, decoder_output)
        return generator
    def discriminator(self):
        '''
        Discriminator get the output of the real and fake seperately and then using them to calculate the joint loss
        '''
        global_input = Input(shape=(self.input_dim, self.variable_num, 1))
        local_input = Input(shape=(self.input_dim, 1, 1))
        def global_discriminator(global_input):
            x = tf.keras.layers.Conv2D(64, 5, (2,2), activation='relu', padding='same', name="Global_Discriminator_1")(global_input)
            x = tf.keras.layers.Conv2D(128, 5, (2,2), activation='relu', padding='same', name="Global_Discriminator_2")(x)

            flatten_layer = tf.keras.layers.Flatten()
            result = flatten_layer(x)
            result = tf.keras.layers.Dense(1024)(result)
            return result

        def local_discriminator(local_input):
            x = tf.keras.layers.Conv2D(64, 5, (2, 2), activation='relu', padding='same', name="Local_Discriminator_1")(local_input)
            x = tf.keras.layers.Conv2D(128, 5, (2, 2), activation='relu', padding='same', name="Local_Discriminator_2")(x)

            flatten_layer = tf.keras.layers.Flatten()
            result = flatten_layer(x)
            result = tf.keras.layers.Dense(1024)(result)
            return result

        global_d = global_discriminator(global_input)
        local_d = local_discriminator(local_input)

        output = tf.concat([global_d, local_d], axis=1)
        output = tf.keras.layers.Dense(1)(output)

        model = tf.keras.Model(inputs=[global_input, local_input], outputs=output)
        return model
    def get_points(self, batch_size):
        '''
        Manuelly created the missing data for training the imputation model
        '''
        mask = []
        points = []
        for i in range(batch_size):
            m = np.zeros((self.vector_size, self.variable_num, 1), dtype=np.uint8)
            x1 = np.random.randint(0, self.vector_size - self.local_size + 1, 1)
            x2 = x1 + self.local_size
            points.append([x1, x2])

            m[:, -2] = 1
            mask.append(m)
        return np.array(points), np.array(mask)
    def joint_loss(self, real, fake):
        '''
        :param real: Output from discriminator for real samples
        :param fake: Output from discriminator for fake samples
        :return:
        '''
        alpha = 4e-4
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake)))
        return tf.add(d_loss_real, d_loss_fake) * alpha
    def pre_trained_encoder(self):
        '''
        According to the training procedure, the encoder for reconstruting the missing part would be firstly trained, then,
        it would come to training the discriminator
        :param x_train: Training data
        '''
        generator = self.build_autoencoder()
        generator.compile(optimizer=tf.keras.optimizers.Adam(self.lr), loss='mean_squared_error')

        for epoch in range(self.pre_trained_epoch + 1):
            np.random.shuffle(self.x_train)
            step_num = int(len(self.x_train) * 0.8 / self.batch_size)
            for i in range(step_num):
                x_batch = self.x_train[0:int(len(self.x_train) * 0.8)][i * self.batch_size:(i + 1) * self.batch_size]
                points_batch, mask_batch = self.get_points(x_batch.shape[0])

                generator_input = x_batch * (1 - mask_batch)
                loss = generator.train_on_batch(generator_input, x_batch)

            x_batch_test = self.x_train[int(len(self.x_train) * 0.8):][:self.batch_size]
            points_batch, mask_batch_test = self.get_points(x_batch_test.shape[0])
            generator_test_input = x_batch_test * (1 - mask_batch_test)
            test_loss = generator.evaluate(generator_test_input, x_batch_test)

            if epoch % 10 == 0:
                print(f"Autoencoder Pretrain Epoch [{epoch}/{self.pre_trained_epoch}] | Loss: {loss}| Test Loss: {test_loss}")
                generator.save(f'{self.model_save_path}/generator_{self.building}_warm.h5')

    def discriminator_training(self, adversarial_epoch=50, real_time_training=False):
        '''
        After firstly training the autoencoder, the discriminator will be introduced for better performance of missing data imputation
        :param adversarial_epoch: How many training epochs with the combination of generator and discriminator
        :param real_time_training: Whether the model is firstly trained or further trained
        '''
        if real_time_training:
            generator = load_model(f'{self.model_save_path}/generator_after_discriminator_{self.building}_50.h5')
            print(f'Load model from path {self.model_save_path}/generator_after_discriminator_{self.building}_50.h5')
        else:
            generator = load_model(f'{self.model_save_path}/generator_{self.building}_warm.h5')

        discriminator = self.discriminator()

        generator.compile(optimizer=tf.keras.optimizers.Adam(self.lr), loss='mean_squared_error')
        discriminator.compile(optimizer=tf.keras.optimizers.Adam(self.lr), loss='mean_squared_error')

        optimizer_D = tf.keras.optimizers.Adam(self.lr)
        optimizer_G = tf.keras.optimizers.Adam(self.lr)

        for epoch in range(adversarial_epoch+1):
            np.random.shuffle(self.x_train)
            step_num = int(len(self.x_train) * 0.8 / self.batch_size)
            for i in range(step_num):
                x_batch = self.x_train[i * self.batch_size:(i + 1) * self.batch_size]
                _, mask_batch = self.get_points(x_batch.shape[0])
                generator_input = x_batch * (1 - mask_batch)

                generator_output = generator.predict(generator_input)
                completion = generator_output * mask_batch + generator_input * (1 - mask_batch)

                local_x_batch = []
                local_completion_batch = []
                # for i in range(self.batch_size):
                for i in range(x_batch.shape[0]):
                    local_x_batch.append(x_batch[i][0:24, -2])
                    local_completion_batch.append(completion[i][0:24, -2])

                local_x_batch = np.array(local_x_batch)
                local_completion_batch = np.array(local_completion_batch)

                with tf.GradientTape() as tape:
                    real_output = discriminator([x_batch, local_x_batch], training=True)
                    fake_output = discriminator([completion, local_completion_batch], training=True)
                    d_loss = self.joint_loss(real_output, fake_output)

                gradients = tape.gradient(d_loss, discriminator.trainable_variables)
                optimizer_D.apply_gradients(zip(gradients, discriminator.trainable_variables))

                # Test Discriminator
                x_batch_test = self.x_train[int(len(self.x_train) * 0.8):][:self.batch_size]
                _, mask_batch_test = self.get_points(x_batch_test.shape[0])
                generator_test_input = x_batch_test * (1 - mask_batch_test)

                test_generator_output = generator.predict(generator_test_input)
                test_completion = test_generator_output * mask_batch_test + generator_test_input * (1 - mask_batch_test)

                local_x_batch_test = []
                local_completion_batch_test = []
                #for i in range(self.batch_size):
                for i in range(x_batch_test.shape[0]):
                    # x1, x2 = points_batch_test[:, 0, :][i][0], points_batch_test[:, 1, :][i][0]
                    local_x_batch_test.append(x_batch_test[i][0:24, :])
                    local_completion_batch_test.append(test_completion[i][0:24, :])

                local_x_batch_test = np.array(local_x_batch_test)
                local_completion_batch_test = np.array(local_completion_batch_test)

                test_real_output = discriminator([x_batch_test, local_x_batch_test], training=False)
                test_fake_output = discriminator([test_completion, local_completion_batch_test], training=False)

                d_test_loss = self.joint_loss(test_real_output, test_fake_output)

                # Train Generator
                # with tf.GradientTape() as tape:
                generator_input = x_batch * (1 - mask_batch)
                g_loss = generator.train_on_batch(generator_input, x_batch)
                # generator_output = generator.predict(generator_input)

                # gradients = tape.gradient(g_loss, generator.trainable_variables)
                # optimizer_G.apply_gradients(zip(gradients, generator.trainable_variables))

                # Test Generator
                x_batch_test = self.x_train[int(len(self.x_train) * 0.8):][:self.batch_size]
                points_batch_test, mask_batch_test = self.get_points(x_batch_test.shape[0])
                generator_test_input = x_batch_test * (1 - mask_batch_test)

                g_test_loss = generator.evaluate(generator_test_input, x_batch_test)

            if epoch % 10 == 0:
                print(
                    f"Epoch [{epoch}/{adversarial_epoch}] | D Loss: {d_loss} | G Loss: {g_loss}| D Test Loss: {d_test_loss} | G Test Loss: {g_test_loss}")
                generator.save(f'{self.model_save_path}/generator_after_discriminator_{self.building}.h5')

hour = 24
building = 'AT_SFH'

tracemalloc.start()
t0 = time.time()

x_train_1 = np.load(
    f'C:/Users\meyu\PycharmProjects\RLC/venv\Multi_tasking/train_data/{hour}/task_{building}_data_solar_{hour}.npy')#[150:180]
model = MissingData2DModel(building, 4, x_train_1, hour)
model.pre_trained_encoder()
model.discriminator_training(adversarial_epoch=50)

#time_usage.append(time.time() - t0)
#print(time.time() - t0)

print("Memory usage:", tracemalloc.get_traced_memory())
print("Training time:", time.time() - t0)
tracemalloc.stop()
