import numpy as np
import pandas as pd
import os

import tensorflow as tf
from tensorflow.keras.layers import Input
from keras import models
from keras.models import load_model
from tensorflow.keras.layers import Concatenate

SEED = 1
tf.random.set_seed(SEED)

def build_autoencoder():
    '''
    Build an 2D contextual autoencoder model for multi-variante missing data imputation
    :return: contextual autoencoder model, which will serve as a generator for pre-training
    '''
    encoder_input = Input(shape=(24, 4, 1))
    x = tf.keras.layers.Conv2D(128, 3, (2, 2), padding='same', activation='relu', name="Encoder_1")(encoder_input)
    x = tf.keras.layers.Conv2D(256, 3, (2, 2), padding='same', activation='relu', name="Encoder_2")(x)

    x = tf.keras.layers.Conv2D(256, 3, dilation_rate=(1, 1), padding='same', name="Feature_1")(x)
    x = tf.keras.layers.Conv2D(256, 3, dilation_rate=(2, 2), padding='same', name="Feature_2")(x)
    x = tf.keras.layers.Conv2D(256, 3, dilation_rate=(4, 4), padding='same', name="Feature_3")(x)  # 4,3
    x = tf.keras.layers.Conv2D(256, 3, dilation_rate=(8, 8), padding='same', name="Feature_4")(x)  # 8,3

    x = tf.keras.layers.Conv2DTranspose(128, 3, (2, 2), padding='same', activation='relu', name="Decoder_1")(x)

    decoder_output = tf.keras.layers.Conv2DTranspose(1, 3, (2, 2), padding='same', activation='tanh', name="Decoder_3")(
        x)

    generator = models.Model(encoder_input, decoder_output)
    return generator

def discriminator():
    '''
    Discriminator get the output of the real and fake seperately and then using them to calculate the joint loss
    '''
    global_input = Input(shape=(24, 4, 1))
    local_input = Input(shape=(24, 1, 1))

    def global_discriminator(global_input):
        x = tf.keras.layers.Conv2D(64, 5, (2, 2), activation='relu', padding='same', name="Global_Discriminator_1")(
            global_input)
        x = tf.keras.layers.Conv2D(128, 5, (2, 2), activation='relu', padding='same', name="Global_Discriminator_2")(x)

        flatten_layer = tf.keras.layers.Flatten()
        result = flatten_layer(x)
        result = tf.keras.layers.Dense(1024)(result)
        return result

    def local_discriminator(local_input):
        x = tf.keras.layers.Conv2D(64, 5, (2, 2), activation='relu', padding='same', name="Local_Discriminator_1")(
            local_input)
        x = tf.keras.layers.Conv2D(128, 5, (2, 2), activation='relu', padding='same', name="Local_Discriminator_2")(x)

        flatten_layer = tf.keras.layers.Flatten()
        result = flatten_layer(x)
        result = tf.keras.layers.Dense(1024)(result)
        return result

    global_d = global_discriminator(global_input)
    local_d = local_discriminator(local_input)

    output = Concatenate(axis=1)([global_d, local_d])
    output = tf.keras.layers.Dense(1)(output)

    model = tf.keras.Model(inputs=[global_input, local_input], outputs=output)
    return model

def get_points(batch_size):
    '''
    Manuelly created the missing data for training the imputation model
    '''
    mask = []
    for i in range(batch_size):
        m = np.zeros((24, 4, 1), dtype=np.uint8)
        x1 = np.random.randint(0, 24 - 24 + 1, 1)
        x2 = x1 + 24

        m[:, -2] = 1
        mask.append(m)
    return np.array(mask)

def generator_loss(gen_output, target):
    l2_loss = tf.reduce_mean(tf.square(target - gen_output))
    return l2_loss

def joint_loss(real, fake):
    '''
    :param real: Output from discriminator for real samples
    :param fake: Output from discriminator for fake samples
    :return:
    '''
    alpha = 4e-4
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake)))
    return tf.add(d_loss_real, d_loss_fake) * alpha

@tf.function
def pre_trained_encoder(generator, optimizer_G, batch_size, batch_tuple, variable_num, building):
    masks = get_points(batch_size)
    masks = tf.cast(masks, tf.float32)

    total_g_loss = 0.0
    with tf.GradientTape() as tape:
        x_batch = tf.cast(batch_tuple, tf.float32)
        gen_input = x_batch * (1 - masks)
        gen_output = generator(gen_input, training=True)

        gen_total_loss = generator_loss(gen_output, x_batch)
        total_g_loss += gen_total_loss

    grads_g = tape.gradient(total_g_loss, generator.trainable_variables)
    optimizer_G.apply_gradients(zip(grads_g, generator.trainable_variables))

    return total_g_loss

@tf.function
def discriminator_training(discriminator, generator, optimizer_G, optimizer_D, batch_size, batch_tuple, building,
                           adversarial_epoch=50):
    masks = get_points(batch_size)
    masks = tf.cast(masks, tf.float32)

    total_g_loss = 0.0
    total_d_loss = 0.0
    with tf.GradientTape(persistent=True) as tape:
        x_batch = tf.cast(batch_tuple, tf.float32)

        gen_input = x_batch * (1 - masks)
        gen_output = generator(gen_input)

        completion = gen_output * masks + gen_input * (1 - masks)

        local_x_batch = []
        local_completion_batch = []
        for i in range(batch_size):
            local_x_batch.append(batch_tuple[i][0:24, -2])
            local_completion_batch.append(completion[i][0:24, -2])

        local_x_batch = tf.convert_to_tensor(local_x_batch, dtype=tf.float32)
        local_completion_batch = tf.convert_to_tensor(local_completion_batch,
                                                      dtype=tf.float32)

        real_output = discriminator([batch_tuple, local_x_batch], training=True)
        fake_output = discriminator([completion, local_completion_batch], training=True)
        d_loss = joint_loss(real_output, fake_output)
        total_d_loss += d_loss

        gen_output = generator(gen_input, training=True)
        gen_total_loss = generator_loss(gen_output, x_batch)
        total_g_loss += gen_total_loss

    grad_g = tape.gradient(total_g_loss, generator.trainable_variables)
    optimizer_G.apply_gradients(zip(grad_g, generator.trainable_variables))

    grad_d = tape.gradient(total_d_loss, discriminator.trainable_variables)
    optimizer_D.apply_gradients(zip(grad_d, discriminator.trainable_variables))

    return total_g_loss, total_d_loss

def make_dataset(x, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(x)
    ds = ds.shuffle(buffer_size=len(x))
    ds = ds.take(int(len(x))).batch(batch_size).repeat()
    return ds.map(lambda batch: tf.cast(batch, tf.float32),
                  num_parallel_calls=tf.data.AUTOTUNE)

building_lst = ['AT_SFH']
for i, building in enumerate(building_lst):
    x_train = np.load(f'train_data/train_{building}.npy')

    batch_size = 64  # GF:32, When2Heat: 64
    dataset = make_dataset(x_train, batch_size=batch_size)
    ds_iter = iter(dataset)
    steps_per_epoch = int(len(x_train) / batch_size)

    generator = build_autoencoder()
    optimizer_G = tf.keras.optimizers.Adam(0.0001)

    zero_grads_G = [tf.zeros_like(v) for v in generator.trainable_variables]
    optimizer_G.apply_gradients(zip(zero_grads_G, generator.trainable_variables))

    for epoch in range(100 + 1):
        for _ in range(steps_per_epoch):
            batch = next(ds_iter)
            bs = batch.shape[0]
            total_g_loss = pre_trained_encoder(generator, optimizer_G, bs, batch, 4, building)
        if epoch % 5 == 0:
            print(f"Autoencoder Pretrain Epoch [{epoch}/{100}] | Loss: {total_g_loss}")
            generator.save(f'Baseline_Model/generator_{building}.h5')

    disc = discriminator()
    generator = load_model(f'Baseline_Model/generator_{building}.h5')
    optimizer_D = tf.keras.optimizers.Adam(0.0001)
    optimizer_G = tf.keras.optimizers.Adam(0.0001)

    zero_grads_G = [tf.zeros_like(v) for v in generator.trainable_variables]
    optimizer_G.apply_gradients(zip(zero_grads_G, generator.trainable_variables))
    zero_grads_D = [tf.zeros_like(v) for v in disc.trainable_variables]
    optimizer_D.apply_gradients(zip(zero_grads_D, disc.trainable_variables))
    for epoch in range(50+1):
        for _ in range(steps_per_epoch):
            batch = next(ds_iter)
            bs = batch.shape[0]
            total_g_loss, total_d_loss = discriminator_training(disc, generator, optimizer_G, optimizer_D, bs, batch,
                                                                building)
        if epoch % 5 == 0:
            print(f"Epoch [{epoch}/{50}] | D Loss: {total_d_loss} | G Loss: {total_g_loss}")
            generator.save(f'Baseline_Model/generator_after_discriminator_{building}.h5')

print("Memory usage:", tracemalloc.get_traced_memory())
print("Training time:", time.time() - t0)
tracemalloc.stop()
