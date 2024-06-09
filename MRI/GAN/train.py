import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten, LeakyReLU, Input, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential

from MRI.plot_mri import plot_mri

from MRI.image_preprocessing import trim_rows, normalize, check_dimensions
from MRI.file_io import read_pickle

# Global variables to store training and validation accuracy and loss
global_train_d_loss = []
global_train_g_loss = []
global_train_d_accuracy = []
global_val_d_loss = []
global_val_g_loss = []
global_val_d_accuracy = []

def train_gan(save=True, data_type="ADHD", pickle_path=".", gan_model_path="."):
    """
    Trains a GAN using MRI data.

    Args:
        save (bool): Whether to save the model after training.
        data_type (str): The type of data (ADHD or CONTROL).
        pickle_path (str): The path to the pickle files with the data.
        gan_model_path (str): The path to save the trained model.
    """
    from MRI.config import (GAN_EPOCHS_MRI, GAN_BATCH_SIZE_MRI, GAN_INPUT_SHAPE_MRI, GAN_LEARNING_RATE,
                            TRAIN_GAN_DISP_INTERVAL, TRAIN_GAN_PRINT_INTERVAL)
    try:
        print(f"TRAINING GAN {data_type} STARTED for {GAN_EPOCHS_MRI} EPOCHS...")
        print(f"RESULTS WILL BE DISPLAYED EVERY {TRAIN_GAN_PRINT_INTERVAL}")
        print(f"PREVIEW IMAGE GENERATED EVERY {TRAIN_GAN_DISP_INTERVAL}")
        print("\n")

        try:
            if data_type == "ADHD":
                DATA = read_pickle(os.path.join(pickle_path, "ADHD_REAL.pkl"))
            elif data_type == "CONTROL":
                DATA = read_pickle(os.path.join(pickle_path, "CONTROL_REAL.pkl"))
        except Exception as e:
            print(f"Error loading data: {e}")
            return

        try:
            TRIMMED = trim_rows(DATA)
            check_dimensions(TRIMMED)
            NORMALIZED = normalize(TRIMMED)
        except Exception as e:
            print(f"Error processing data: {e}")
            return

        try:
            train_data, val_data = train_test_split(NORMALIZED, test_size=0.2)
            train_data = np.expand_dims(train_data, axis=-1)
            val_data = np.expand_dims(val_data, axis=-1)
        except Exception as e:
            print(f"Error splitting data into training and validation sets: {e}")
            return

        def build_generator():
            try:
                model = Sequential()
                model.add(Input(shape=(100,)))
                model.add(Dense(256))
                model.add(LeakyReLU(negative_slope=0.2))
                model.add(BatchNormalization())
                model.add(Dense(512))
                model.add(LeakyReLU(negative_slope=0.2))
                model.add(BatchNormalization())
                model.add(Dense(1024))
                model.add(LeakyReLU(negative_slope=0.2))
                model.add(BatchNormalization())
                model.add(Dense(120 * 120 * 1, activation='tanh'))
                model.add(Reshape(GAN_INPUT_SHAPE_MRI))
                return model
            except Exception as e:
                print(f"Error building generator: {e}")
                return

        def build_discriminator():
            try:
                model = Sequential()
                model.add(Input(shape=GAN_INPUT_SHAPE_MRI))
                model.add(Flatten())
                model.add(Dense(512))
                model.add(LeakyReLU(negative_slope=0.2))
                model.add(Dropout(0.3))
                model.add(Dense(256))
                model.add(LeakyReLU(negative_slope=0.2))
                model.add(Dropout(0.3))
                model.add(Dense(1, activation='sigmoid'))
                return model
            except Exception as e:
                print(f"Error building discriminator: {e}")
                return

        def build_gan(generator, discriminator):
            try:
                model = Sequential()
                model.add(generator)
                model.add(discriminator)
                return model
            except Exception as e:
                print(f"Error building GAN: {e}")
                return

        generator = build_generator()
        if generator is None:
            return

        discriminator = build_discriminator()
        if discriminator is None:
            return

        try:
            generator_optimizer = tf.keras.optimizers.Adam(GAN_LEARNING_RATE)
            discriminator_optimizer = tf.keras.optimizers.Adam(GAN_LEARNING_RATE)
        except Exception as e:
            print(f"Error creating optimizers: {e}")
            return

        try:
            discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer, metrics=['accuracy'])
            gan = build_gan(generator, discriminator)
            gan.compile(loss='binary_crossentropy', optimizer=generator_optimizer)
        except Exception as e:
            print(f"Error compiling models: {e}")
            return

        def generate_image(generator, epoch):
            try:
                noise = np.random.normal(0, 1, [1, 100])
                generated_image = generator.predict(noise)
                generated_image = generated_image * 0.5 + 0.5
                plot_mri(generated_image[0], f'Epoch: {epoch}')
            except Exception as e:
                print(f"Failed to generate image in epoch {epoch}: {e}")

        @tf.function
        def train_step(generator, discriminator, real_imgs, batch_size, generator_optimizer, discriminator_optimizer):
            """
            Performs one training step for the GAN.

            Args:
                generator (tf.keras.Model): The generator model.
                discriminator (tf.keras.Model): The discriminator model.
                real_imgs (tf.Tensor): A batch of real images.
                batch_size (int): The number of images in the batch.
                generator_optimizer (tf.keras.optimizers.Optimizer): The optimizer for the generator.
                discriminator_optimizer (tf.keras.optimizers.Optimizer): The optimizer for the discriminator.

            Returns:
                tuple: A tuple containing the discriminator loss, the generator loss, and the discriminator accuracy.
            """
            try:
                noise = tf.random.normal([batch_size, 100])
                with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
                    generated_imgs = generator(noise, training=True)

                    real_output = discriminator(real_imgs, training=True)
                    fake_output = discriminator(generated_imgs, training=True)

                    d_loss_real = tf.keras.losses.BinaryCrossentropy(from_logits=False)(tf.ones_like(real_output), real_output)
                    d_loss_fake = tf.keras.losses.BinaryCrossentropy(from_logits=False)(tf.zeros_like(fake_output), fake_output)
                    d_loss = d_loss_real + d_loss_fake

                    g_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(tf.ones_like(fake_output), fake_output)

                gradients_of_discriminator = disc_tape.gradient(d_loss, discriminator.trainable_variables)
                discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

                gradients_of_generator = gen_tape.gradient(g_loss, generator.trainable_variables)
                generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

                accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(real_output), tf.ones_like(real_output)), tf.float32))
                return d_loss, g_loss, accuracy
            except Exception as e:
                print(f"Error during training step: {e}")
                return None, None, None

        def train_gan(generator, discriminator, epochs, batch_size, train_data, val_data):
            """
            Trains the GAN model.

            Args:
                generator (tf.keras.Model): The generator model.
                discriminator (tf.keras.Model): The discriminator model.
                epochs (int): The number of epochs to train the GAN.
                batch_size (int): The number of images in each batch.
                train_data (np.ndarray): The training data containing real images.
                val_data (np.ndarray): The validation data containing real images.
            """
            global global_train_d_loss, global_train_g_loss, global_train_d_accuracy
            global global_val_d_loss, global_val_g_loss, global_val_d_accuracy
            gen_loss = np.array([])
            for epoch in range(epochs):
                try:
                    idx = np.random.randint(0, train_data.shape[0], batch_size)
                    real_imgs = train_data[idx]
                    d_loss, g_loss, accuracy = train_step(generator, discriminator, real_imgs, batch_size, generator_optimizer, discriminator_optimizer)
                    if d_loss is None or g_loss is None:
                        return
                    global_train_d_loss.append(d_loss.numpy().mean())
                    global_train_g_loss.append(g_loss.numpy().mean())
                    global_train_d_accuracy.append(accuracy.numpy().mean())
                    if (epoch + 1) % 1000 == 0: gen_loss = np.append(gen_loss, g_loss.numpy().mean())
                    if (epoch + 1) % TRAIN_GAN_PRINT_INTERVAL == 0:
                        val_idx = np.random.randint(0, val_data.shape[0], batch_size)
                        val_real_imgs = val_data[val_idx]
                        val_d_loss, val_g_loss, val_accuracy = train_step(generator, discriminator, val_real_imgs, batch_size, generator_optimizer, discriminator_optimizer)
                        if val_d_loss is None or val_g_loss is None:
                            return
                        global_val_d_loss.append(val_d_loss.numpy().mean())
                        global_val_g_loss.append(val_g_loss.numpy().mean())
                        global_val_d_accuracy.append(val_accuracy.numpy().mean())
                        print(f"Epoch {epoch + 1} [Val D loss: {val_d_loss.numpy().mean()} | Val G loss: {val_g_loss.numpy().mean()} | Val D accuracy: {val_accuracy.numpy().mean()}]")

                    if (epoch + 1) % TRAIN_GAN_DISP_INTERVAL == 0:
                        generate_image(generator, epoch + 1)
                except Exception as e:
                    print(f"Error in epoch {epoch + 1}: {e}")
                    return

            if save:
                try:
                    generator.save(os.path.join(gan_model_path, f'{round(np.mean(gen_loss), 4)}_{data_type}_GAN.keras'))
                except Exception as e:
                    print(f"Error saving model: {e}")
                    return
            return np.mean(gen_loss)

        name = train_gan(generator, discriminator, epochs=GAN_EPOCHS_MRI, batch_size=GAN_BATCH_SIZE_MRI, train_data=train_data, val_data=val_data)
        return round(name, 4)
    except Exception as e:
        print(f"An error occurred during GAN training: {e}")
        return
