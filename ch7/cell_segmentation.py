import deepchem as dc
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import os
import re

RETRAIN = False

# NOTE: downloaded from https://data.broadinstitute.org/bbbc/BBBC005/BBBC005_v1_ground_truth.zip
# image_dir = '~/Downloads/BBBC005_v1_images/'
image_dir = '/Users/mmadero/Downloads/BBBC005_v1_images/'
label_dir = '/Users/mmadero/Downloads/BBBC005_v1_ground_truth/BBBC005_v1_ground_truth/'
rows = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P')
blurs = (1, 4, 7, 10, 14, 17, 20, 23, 26, 29, 32, 35, 39, 42, 45, 48)
files = []
labels = []

for f in os.listdir(label_dir):
    if f.endswith('.TIF'):
        for row, blur in zip(rows, blurs):
            fname = f.replace('_F1', '_F%d'%blur).replace('_A', '_%s'%row)
            files.append(os.path.join(image_dir, fname))
            labels.append(os.path.join(label_dir, f))

# print(files)
# print(labels)

dataset = dc.data.ImageDataset(files, labels)

splitter = dc.splits.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset, seed=123)

features = tf.keras.Input(shape=(520,696, 1))
# TODO: are labels needed in Keras?

# downsample three times
conv1 = layers.Conv2D(16, kernel_size=5, strides=2, activation=tf.nn.relu, padding='same')(features/255.0)
conv2 = layers.Conv2D(32, kernel_size=5, strides=2, activation=tf.nn.relu, padding='same')(conv1)
conv3 = layers.Conv2D(64, kernel_size=5, strides=2, activation=tf.nn.relu, padding='same')(conv2)

# Do a 1x1 convolution TODO: WTF does that mean?. It's 64 to 64 on the ouput and stride is 1, does it have something to do with that? 
# Does that mean 1 kernel x 1 stride  ?
conv4 = layers.Conv2D(64, kernel_size=1, strides=1)(conv3)

# Upsample three times
# TODO: what's the axis here?
concat1 = layers.Concatenate(axis=3)([conv3, conv4])
deconv1 = layers.Conv2DTranspose(32, kernel_size=5, strides=2, activation=tf.nn.relu, padding='same')(concat1)
concat2 = layers.Concatenate(axis=3)([conv2, deconv1])
deconv2 = layers.Conv2DTranspose(16, kernel_size=5, strides=2, activation=tf.nn.relu, padding='same')(concat2)
concat3 = layers.Concatenate(axis=3)([conv1, deconv2])
deconv3 = layers.Conv2DTranspose(1, kernel_size=5, strides=2, activation=tf.nn.relu, padding='same')(concat3)

# Compute the final output
concat4 = layers.Concatenate(axis=3)([features, deconv3])
# TODO: what's stride?
# TODO: what's the default activation? 
# In the book they set it to =None and when I had that the score dropped to 24% (withthe default it's 75%)
# logits = layers.Conv2D(1, kernel_size=5, strides=1, activation=None, padding='same')(concat4)
logits = layers.Conv2D(1, kernel_size=5, strides=1, padding='same')(concat4)
output = layers.Activation(tf.math.sigmoid)(logits)


keras_model = tf.keras.Model(inputs=features, outputs=[output, logits])
learning_rate = dc.models.optimizers.ExponentialDecay(0.01,  0.9, 250)
model = dc.models.KerasModel(keras_model,
    loss=dc.models.losses.SigmoidCrossEntropy(),
    output_types=['prediction', 'loss'],
    learning_rate =learning_rate,
    model_dir='models/segmentation/')

if RETRAIN: 
    model.fit(train_dataset, nb_epoch=50)
else: 
    model.restore()

scores = []
for x,y,w,id in test_dataset.itersamples():
    y_pred = model.predict_on_batch([x]).squeeze()
    scores.append(np.mean((y>0) == (y_pred>0.5)))
print(scores)
print(np.mean(scores))
