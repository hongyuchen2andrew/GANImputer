from loguru import logger
import numpy as np
import pandas as pd
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam
import os


def cal_latent_dim(attr_num, miss_rate):
    num = int((1 - miss_rate) * attr_num / 2)
    return num


def define_discriminator(input_dim):
    model = Sequential()
    # 1st layer
    model.add(layers.Dense(input_dim, activation="relu"))
    # 2nd layer
    model.add(layers.Dense(int(2*input_dim), activation="relu"))
    # 3rd layer
    model.add(layers.Dense(int(2*input_dim), activation="relu"))
    # 4th layer
    model.add(layers.Dense(input_dim, activation="sigmoid"))
    # compile model
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])

    return model


def define_generator(latent_dim, target_dim):
    model = Sequential()
    # 1st layer
    model.add(layers.Dense(latent_dim, activation="relu"))
    # 2nd layer
    model.add(layers.Dense(target_dim, activation="relu"))
    # 3rd layer
    model.add(layers.Dense(target_dim*2, activation="relu"))
    # 4th layer
    model.add(layers.Dense(target_dim, activation="sigmoid"))

    return model


def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(g_model)
    # add discriminator
    model.add(d_model)
    # compile model
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt)
    return model


def generate_real_samples(data, sign, n_samples, attr_mean, attr_std):
    ix = np.random.randint(0, data.shape[0], n_samples)
    data = data[ix].copy()
    sign = sign[ix].copy()
    row, col = data.shape
    invalid_index = [(i, j) for i in range(row) for j in range(col) if sign[i, j] == 0]
    # fill nan data with normal distributed numbers
    for i in invalid_index:
        data[i] = np.random.normal(size=1, loc=attr_mean[i[1]], scale=attr_std[i[1]])
    return data, sign


def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = np.random.normal(size=latent_dim*n_samples, loc=0, scale=1)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def generate_fake_samples(g_model, latent_dim, n_samples, n_column):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = g_model.predict(x_input, verbose=0)
    # create "fake" class labels 0
    y = np.zeros((n_samples, n_column))
    return X, y


def train(g_model, d_model, gan_model, dataset, sign, latent_dim, num_attr, attr_mean, attr_std, n_epochs=250, n_batch=250):
    bat_per_epoch = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch/2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epoch):
            # get randomly selected "real" samples
            data_mix, sign_mix = generate_real_samples(dataset, sign, half_batch, attr_mean, attr_std)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch(data_mix, sign_mix)
            # generate "fake" examples
            data_fake, sign_fake = generate_fake_samples(g_model, latent_dim, half_batch, num_attr)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch(data_fake, sign_fake)
            # update points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = np.ones((n_batch, num_attr))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on the batch
            # if j == bat_per_epoch - 1:
            #     print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
            #     (i+1, j+1, bat_per_epoch, d_loss1, d_loss2, g_loss))


def build_model(name, miss_rate, n_epoch, n_batch):
    # load data
    dataset = pd.read_table(f"data/{name}_clean.csv", sep=",")
    sign = pd.read_table(f"data/{name}_sign.csv", sep=",")
    dataset = np.array(dataset)
    sign = np.array(sign)
    n_attr = dataset.shape[1]
    # size of the latent space
    latent_dim = cal_latent_dim(n_attr, miss_rate)
    # Report the information
    logger.info(f"Number of attributes is {n_attr}")
    logger.info(f"Latent dimension is {latent_dim}")

    # statistics of the data
    attr_mean = np.nanmean(dataset, axis=0)
    attr_std = np.nanstd(dataset, axis=0)

    # create the discriminator
    d_model = define_discriminator(n_attr)
    # create the generator
    g_model = define_generator(latent_dim, n_attr)
    # create the gan
    gan_model = define_gan(g_model, d_model)
    # train the model
    train(g_model, d_model, gan_model, dataset, sign, latent_dim, n_attr, attr_mean, attr_std, n_epoch, n_batch)
    g_model.save(f"model/{name}_g_model.h5")
