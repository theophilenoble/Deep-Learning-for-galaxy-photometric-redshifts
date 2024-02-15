import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import initializers
from tensorflow.keras import optimizers

import keras
from keras import layers

from scipy.stats import median_abs_deviation


# Data preparation
with np.load('../../../../scratch/noble/deepdip/sdss.npz') as data_file:
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

x_train = x_data[:400000]
x_val = x_data[400000:]
ebv_train = ebv_data[:400000]
ebv_val = ebv_data[400000:]
y_train = y_data[:400000]
y_val = y_data[400000:]


### Metrics

def metrics(z, pred):
    delta_z = (pred-z)/(1+z) #residuals
    pred_bias = np.mean(delta_z) #prediction bias
    nmad = 1.4826 * np.median(np.abs(delta_z-np.median(delta_z))) #MAD deviation
    outliers = np.sum(np.abs(delta_z)>0.05)/len(z) #Fraction of outliers
    return delta_z, pred_bias, nmad, outliers

def plot_results(z, pred, delta_z, pred_bias, nmad, outliers, title):
    z = z.squeeze()
    pred = pred.squeeze()
    plt.hist2d(z, pred, 150, range=[[0,0.6],[0,0.6]], cmap='gist_stern', cmin=1e-3); 
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


### Transfer learning

base_model = tf.keras.applications.Xception(include_top=False, input_shape=(72,72,3))
base_model.trainable = False

new_base_model = keras.Model(inputs=base_model.layers[2].input, outputs=base_model.layers[-1].output)

inputs = keras.Input(shape=(64, 64, 5), name="inputs")
new_inputs = layers.ZeroPadding2D(4)(inputs)
new_inputs = layers.Conv2D(32, 3, strides=2)(new_inputs)

x = new_base_model(new_inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1096, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(1096, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(1, activation="linear")(x)

model_xception = keras.Model(
    inputs=inputs,
    outputs=x
)

opt = optimizers.RMSprop(learning_rate=0.001, rho=0.9)


### Training

# Learning rate schedule
LEARNING_RATE=0.001
LEARNING_RATE_EXP_DECAY=0.9
lr_decay = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: LEARNING_RATE * LEARNING_RATE_EXP_DECAY**epoch,
    verbose=True)

patience = tf.keras.callbacks.EarlyStopping(patience=4)

model_xception.compile(optimizer=opt, loss='mse')

history = model_xception.fit(x_train, y_train,
                    epochs=25,
                    batch_size=64,
                    validation_data=(x_val, y_val),
                    callbacks=[patience, lr_decay])

### Results

# We plot the learning curve (loss)

# plot loss
plt.title('Cross Entropy Loss')
plt.plot(history.history['loss'], color='blue', label='training loss')
plt.plot(history.history['val_loss'], color='orange', label='validation loss')
plt.legend()
ax = plt.gca()
ax.set_ylim(0, 0.005)
plt.savefig('loss.pdf')
plt.close()


predictions_xception = model_xception.predict(x_test)
predictions_xception = predictions_xception.flatten()

delta_z, pred_bias, nmad, outliers = metrics(y_test, predictions_xception)
plot_results(y_test, predictions_xception, delta_z, pred_bias, nmad, outliers, 'Transfer Learning')
