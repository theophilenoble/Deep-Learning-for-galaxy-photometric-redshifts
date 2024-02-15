import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import initializers
from tensorflow.keras import optimizers

import keras
from keras import layers

from scipy.stats import median_abs_deviation

#Data preparation
with np.load('../../../../../scratch/noble/deepdip/sdss.npz') as data_file:
    data = data_file['cube']
    z = data_file['labels']['z']
    ebv = data_file['labels']['EBV']


#Preprocessing

scaling = []

for i, b in enumerate(['u', 'g', 'r', 'i', 'z']):
    sigma = 1.4826*median_abs_deviation(data[..., i].flatten())
    scaling.append(sigma)

data = np.arcsinh(data/scaling/3)


#Data split
(x_data, ebv_data, y_data), (x_test, ebv_test, y_test) = (data[:500000], ebv[:500000], z[:500000]), (data[500000:], ebv[500000:], z[500000:])

### Metrics
#We define here the different metrics that are used to check the accuracy of the model.

def metrics(z, pred):
    delta_z = (pred-z)/(1+z) #residuals
    pred_bias = np.mean(delta_z) #prediction bias
    nmad = 1.4826 * np.median(np.abs(delta_z-np.median(delta_z))) #MAD deviation
    outliers = np.sum(np.abs(delta_z)>0.05)/len(z) #Fraction of outliers
    return delta_z, pred_bias, nmad, outliers

def plot_results(z, pred, delta_z, pred_bias, nmad, outliers, title):
    z = z.squeeze()
    pred = pred.squeeze()
    plt.hist2d(z, pred, 150, range=[[0,0.6],[0,0.6]], cmap='gist_stern', cmin=1e-3)
    plt.plot([0,0.7],[0,0.7],color='black')
    plt.xlabel('Spectroscopic Redshift' , fontsize=14)
    plt.ylabel('Predicted Redshift', fontsize=14)
    plt.title(title, fontsize=18)
    cbar = plt.colorbar()
    cbar.set_label('Samples')
    number = 0.1
    plt.text(0.1, 0.45, '$ \Delta_z =$' + str(round(pred_bias, 4)) + '\n'
             + '$\eta =$' + str(round(outliers*100, 2)) + '%' + '\n'
             + '$\sigma_{MAD}=$'+ str(round(nmad, 4)),
             bbox=dict(facecolor='w', alpha=0.8, pad=8), fontsize=14)
    plt.savefig('results.pdf')
    plt.close()



### Data augmentation

# flip = layers.RandomFlip()(x_data)
rot = layers.RandomRotation(0.2)(x_data[:100000])
x_data = layers.Concatenate(axis=0)([x_data, rot])

ebv_data = layers.Concatenate(axis=0)([ebv_data, ebv_data[:100000]])
y_data = layers.Concatenate(axis=0)([y_data, y_data[:100000]])

x_test = layers.RandomRotation(0.1)(x_test)

data_shape = x_data.shape[0]

x_train = x_data[:int(0.8*data_shape)]
x_val = x_data[int(0.8*data_shape):]
ebv_train = ebv_data[:int(0.8*data_shape)]
ebv_val = ebv_data[int(0.8*data_shape):]
y_train = y_data[:int(0.8*data_shape)]
y_val = y_data[int(0.8*data_shape):]


### CNN Definition

#Inception Module
def inception(x, nb_in, nb_out, last_layer=False):
  branch1 = layers.Conv2D(nb_in, 1, activation=keras.layers.PReLU(), padding="same")(x)
  branch1 = layers.Conv2D(nb_out, 3, activation=keras.layers.PReLU(), padding="same")(branch1)

  if not(last_layer):
    branch2 = layers.Conv2D(nb_in, 1, activation=keras.layers.PReLU(), padding="same")(x)
    branch2 = layers.Conv2D(nb_out, 5, activation=keras.layers.PReLU(), padding="same")(branch2)

  branch3 = layers.Conv2D(nb_in, 1, activation=keras.layers.PReLU(), padding="same")(x)
  branch3 = layers.AveragePooling2D(pool_size=2, strides=1, padding="same")(branch3)

  branch4 = layers.Conv2D(nb_out, 1, activation=keras.layers.PReLU(), padding="same")(x)

  if not(last_layer):
    output = layers.Concatenate()([branch1, branch2, branch3, branch4])
  else:
    output = layers.Concatenate()([branch1, branch3, branch4])

  return output


reddening = keras.Input(dtype=tf.float32, shape=1, name="reddening")
inputs = keras.Input(dtype=tf.float32, shape=(64, 64, 5), name="inputs")


x = layers.Conv2D(64, 5, activation=keras.layers.PReLU(), padding="same")(inputs)
x = layers.AveragePooling2D(pool_size=2, strides=2, padding="same")(x)

x = inception(x, 48, 64)
x = inception(x, 64, 92)
x = layers.AveragePooling2D(pool_size=2, strides=2, padding="same")(x)
x = inception(x, 92, 128, True)

x = layers.Flatten()(x)

x = layers.Concatenate()([x, reddening])

x = layers.Dense(512, activation="relu")(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(1, activation="linear")(x)


model = keras.Model(
    inputs=[inputs, reddening],
    outputs=[x]
)

opt = optimizers.RMSprop(learning_rate=0.001, rho=0.9)


### CNN Training

# Learning rate schedule
LEARNING_RATE=0.001
LEARNING_RATE_EXP_DECAY=0.9
lr_decay = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: LEARNING_RATE * LEARNING_RATE_EXP_DECAY**epoch,
    verbose=True)

patience = tf.keras.callbacks.EarlyStopping(patience=4)


model.compile(optimizer=opt, loss='mse')
history = model.fit([x_train, ebv_train],
          y_train,
          epochs=25,
          batch_size=64,
          validation_data=([x_val, ebv_val], y_val),
          callbacks=[patience, lr_decay])


# We plot the learning curve (loss)

# plot loss
plt.title('Loss')
plt.plot(history.history['loss'], color='blue', label='training loss')
plt.plot(history.history['val_loss'], color='orange', label='validation loss')
plt.legend()
ax=plt.gca()
ax.set_ylim(0,0.005)
plt.savefig('loss.pdf')
plt.close()


predictions = model.predict([x_test, ebv_test])
predictions = predictions.flatten()

delta_z, pred_bias, nmad, outliers = metrics(y_test, predictions)
plot_results(y_test, predictions, delta_z, pred_bias, nmad, outliers, 'Inception')
