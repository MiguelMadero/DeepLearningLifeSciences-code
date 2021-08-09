import deepchem as dc
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import os
import re

RETRAIN = True
# from tensorflow.python.keras.engine import input_layer

# NOTE: downloaded from https://data.broadinstitute.org/bbbc/BBBC005/BBBC005_v1_images.zip
# image_dir = '~/Downloads/BBBC005_v1_images/'
image_dir = '/Users/mmadero/Downloads/BBBC005_v1_images/'
files = []
labels = []

for f in os.listdir(image_dir):
    if f.endswith('.TIF'):
        files.append(os.path.join(image_dir, f))
        labels.append((int(re.findall('_C(.*?)_', f)[0])))

dataset = dc.data.ImageDataset(files, np.array(labels))
# should I? 
# loader = dc.data.ImageLoader()
# dataset= loader.featurize(files, np.array(labels))

splitter = dc.splits.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
    dataset, 
    seed=123)

# The last is the label?
features = tf.keras.Input(shape=(520,696, 1))
# TODO: ? is this needed
# labels = layers.Label()

prev_layer = features
for num_outputs in [16, 32, 64, 128, 256]:
    prev_layer = layers.Conv2D(num_outputs,
        kernel_size=5,
        strides=2,
        activation=tf.nn.relu)(prev_layer)

output = layers.Dense(units=1)(layers.Flatten()(prev_layer))
# TODO: WTF is this? 
learning_rate = dc.models.optimizers.ExponentialDecay(
    0.0001,
    0.9,
    250
)

keras_model = tf.keras.Model(inputs=features, outputs=output)
model = dc.models.KerasModel(keras_model,
    loss=dc.models.losses.L2Loss(),
    # output_types=['prediction', 'loss']
    learning_rate =learning_rate,
    model_dir='models/model/')
    # learning_rate =learning_rate, model_dir='model/')

# model.add_output(output)
# loss = layers.ReduceSum(layers.L2Loss(in_layers=(output)))
# model.set_loss(loss)
# loss = dc.models.losses.L2Loss()

if RETRAIN: 
    model.fit(train_dataset, nb_epoch=50)
else: 
    model.restore()


y_pred = model.predict(test_dataset).flatten()
print(np.sqrt(np.mean( (y_pred-test_dataset.y)**2) ))
print(y_pred)
print(test_dataset.y)
y_pred = model.predict(valid_dataset).flatten()
print(np.sqrt(np.mean( (y_pred-valid_dataset.y)**2) ))
