import pandas as pd
import numpy as np
from keras.models import load_model
from build import cal_latent_dim
from loguru import logger
import tensorflow as tf
import tensorflow_addons as tfa
import os


def cal_loss_latent(var, mask, data_true):
    loss = np.sqrt(np.sum(((generator(var)-data_true)*mask)**2)/np.sum(mask))
    return loss


def opt_loss_latent(var):
    loss = tf.reduce_sum(((generator(var)-var_data)*var_sign)**2)/tf.reduce_sum(var_sign)
    return loss


def cal_loss_pred(var, w1, w2, w3, w4, b1, b2, b3, b4, mask, data_true):
    pred = tf.keras.activations.sigmoid(tf.matmul(tf.keras.activations.relu(tf.matmul(tf.keras.activations.relu(tf.matmul(tf.keras.activations.relu(tf.matmul(var,w1)+b1),w2)+b2),w3)+b3),w4)+b4)
    loss = tf.sqrt(tf.reduce_sum(((pred-data_true)*mask)**2)/tf.reduce_sum(mask))
    return loss, pred


def opt_loss_pred(var, w1, w2, w3, w4, b1, b2, b3, b4):
    pred = tf.keras.activations.sigmoid(tf.matmul(tf.keras.activations.relu(tf.matmul(tf.keras.activations.relu(tf.matmul(tf.keras.activations.relu(tf.matmul(var,w1)+b1),w2)+b2),w3)+b3),w4)+b4)
    loss = tf.sqrt(tf.reduce_sum(((pred-var_data)*var_sign)**2)/tf.reduce_sum(var_sign))
    return loss


def optimize(name, miss_rate, epochs):
    global generator, var_data, var_sign
    generator = load_model(f"model/{name}_g_model.h5")
    attr = list(pd.read_table(f"data/{name}_std.csv", sep=",").columns.values)
    data_x = np.array(pd.read_table(f"data/{name}_std.csv", sep=","))
    data_miss = np.array(pd.read_table(f"data/{name}_clean.csv", sep=","))
    # Handle exceptions
    data_x[data_x > 100*np.nanmax(data_miss)] = 1
    row, col = data_miss.shape
    for i in range(col):
        avg_val = np.nanmean(data_miss[:, i])
        std_val = np.nanstd(data_miss[:, i])
        F = len(data_miss[:, i][np.isnan(data_miss[:, i])])
        data_miss[:, i][np.isnan(data_miss[:, i])] = avg_val
    sign = np.array(pd.read_table(f"data/{name}_sign.csv", sep=","), dtype="float32")
    latent_dim = cal_latent_dim(col, miss_rate)
    latent_var = np.random.normal(size=(row, latent_dim), loc=0, scale=1)

    mask = sign.copy()
    mask[sign == 0] = 1.0
    mask[sign == 1] = 0.0
    var_sign = tf.Variable(sign, dtype=tf.float32, trainable=False, name="sign")
    var_data = tf.Variable(data_miss, dtype=tf.float32)
    var_latent = tf.Variable(latent_var, dtype=tf.float32, trainable=True, name="var1")

    # Optimize the latent variable
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    train_variable_latent = [var_latent]
    pre_loss_latent = cal_loss_latent(latent_var, mask, data_x)
    logger.info(name)
    logger.info(f"The original loss is {pre_loss_latent}")
    for k in range(epochs):
        with tf.GradientTape() as tp:
            loss = opt_loss_latent(var_latent)
        gradients = tp.gradient(loss, train_variable_latent)
        opt.apply_gradients(zip(gradients, train_variable_latent))
    latent_var = var_latent.numpy()
    after_loss_latent = cal_loss_latent(latent_var, mask, data_x)
    logger.info(f"The loss after optimizing latent variable is {after_loss_latent}")

    # Optimize the latent variable and GAN
    opt = tfa.optimizers.AdamW(weight_decay=0.01, learning_rate=0.01)
    layer1 = generator.layers[0].get_weights()
    layer2 = generator.layers[1].get_weights()
    layer3 = generator.layers[2].get_weights()
    layer4 = generator.layers[3].get_weights()
    weight1 = layer1[0]
    bias1 = layer1[1]
    weight2 = layer2[0]
    bias2 = layer2[1]
    weight3 = layer3[0]
    bias3 = layer3[1]
    weight4 = layer4[0]
    bias4 = layer4[1]
    var_w1 = tf.Variable(weight1, dtype=tf.float32, trainable=True, name="weight1")
    var_w2 = tf.Variable(weight2, dtype=tf.float32, trainable=True, name="weight2")
    var_w3 = tf.Variable(weight3, dtype=tf.float32, trainable=True, name="weight3")
    var_w4 = tf.Variable(weight4, dtype=tf.float32, trainable=True, name="weight4")
    var_b1 = tf.Variable(bias1, dtype=tf.float32, trainable=False, name="bias1")
    var_b2 = tf.Variable(bias2, dtype=tf.float32, trainable=False, name="bias2")
    var_b3 = tf.Variable(bias3, dtype=tf.float32, trainable=False, name="bias3")
    var_b4 = tf.Variable(bias4, dtype=tf.float32, trainable=False, name="bias4")

    train_variable_pred = [var_latent, var_w1, var_w2, var_w3, var_w4]
    for k in range(epochs):
        with tf.GradientTape() as tp:
            loss = opt_loss_pred(var_latent, var_w1, var_w2, var_w3, var_w4, var_b1, var_b2, var_b3, var_b4)
        gradients = tp.gradient(loss, train_variable_pred)
        opt.apply_gradients(zip(gradients, train_variable_pred))
        # latent, w1, w2, w3, w4, b1, b2, b3, b4 = var_latent.numpy(), var_w1.numpy(), var_w2.numpy(), var_w3.numpy(), var_w4.numpy(), var_b1.numpy(), var_b2.numpy(), var_b3.numpy(), var_b4.numpy()
        # after_loss_pred = cal_loss_pred(var_latent, var_w1, var_w2, var_w3, var_w4, var_b1, var_b2, var_b3, var_b4, mask, data_x)
        # print(after_loss_pred)
    latent, w1, w2, w3, w4, b1, b2, b3, b4 = var_latent.numpy(), var_w1.numpy(), var_w2.numpy(), var_w3.numpy(), var_w4.numpy(), var_b1.numpy(), var_b2.numpy(), var_b3.numpy(), var_b4.numpy()
    after_loss_pred, pred = cal_loss_pred(var_latent, var_w1, var_w2, var_w3, var_w4, var_b1, var_b2, var_b3, var_b4, mask, data_x)
    original = pred * mask + data_miss * sign
    original_array = original.numpy()
    original_df = pd.DataFrame(original_array)
    original_df.columns = attr
    # original_df.to_csv(f"data/{name}/{name}_fill.csv", index=False)
    logger.info(f"The loss after optimizing latent variable and generator is {after_loss_pred}")
    # print(str(np.around(pre_loss_latent, 4))+ "-" + str(np.around(after_loss_latent, 4))+ "-" + str(np.around(after_loss_pred, 4)))
    print(str(np.around(after_loss_pred, 4)))
