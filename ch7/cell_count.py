import numpy as np
import deepchem as dc
import tensorflow as tf
import tensorflow.keras.layers as layers

import os 
import re

from tensorflow.python.keras.engine import input_layer

# NOTE: downloaded from https://data.broadinstitute.org/bbbc/BBBC005/BBBC005_v1_images.zip
image_dir = '~/Downloads/BBBC005_v1_images'
files = []
labels = []

for f in os.listdir(image_dir):
    if f.endswith('.TIF'):
        files.append(os.path.join(image_dir, f))
        labels.append((int(re.findall('_C.(*?)_', f)[0])))

loader = dc.data.ImageLoader()
dataset= loader.featurize(files, np.array(labels))

splitter = dc.splits.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
    dataset, 
    seed=123)

learning_rate = dc.models.tensorgraph.optimizers.ExponentialDecay(
    0.0001,
    0.9,11,
    250
)
model = dc.models.TensorGraph(learning_rate =learning_rate, model_dir='model/')
features = tf.keras.Input(shape=(520,696))
# TODO: ? is this needed
labels = layers.Label()

prev_layer = features
for num_outputs in [16, 32, 64, 128, 256]:
    prev_layer = layers.Conv2D(num_outputs,
        kernel_size=5,
        stride=2,
        input_layer=prev_layer    
    )

output = layers.Dense(units=1)
model.add_output(output)
loss = layers.ReduceSum(layers.L2Loss(in_layers=(output, labels)))
model.set_loss(loss)
# loss = dc.models.losses.L2Loss()

model.restore()

y_pred = model.predict(test_dataset).flatten()
print(np.sqrt(np.mean(y_pred-test_dataset.y)**2))

model.fit(train_dataset, nb_epoch=50)

