import deepchem as dc
import tensorflow as tf
import tensorflow.keras.layers as layers

features = tf.keras.Input(shape=(21, 4))
# In the keras version we're not specifying this... why and how?
# labels = layers.Label(shape=(1))
# weights = layers.Weights(shape=(1))

prev = features
for i in range(2):
    prev = layers.Conv1D(
        filters=10,
        kernel_size=10,
        activation=tf.nn.relu,
        padding='same')(prev)
    prev = layers.Dropout(rate=0.3)(prev)

prev = layers.Flatten()(prev)
output = layers.Dense(units=1, activation=tf.math.sigmoid)(prev)

# NOTE: no need for the sigmoid since this is a regression, not classification
# sets to 0/1 as opposed to a random range of probability
# output = layers.Activation(tf.math.sigmoid)(logits)

# TODO: what does it mean to have an array of outputs? How does this relates to loss
# In the previous version we connected them sequentially, the output was output
# but the loss function still relied on logits
keras_model = tf.keras.Model(inputs=features, outputs=[output])

# layers.L2Loss
# Why is it that loss doesn't need to know the structure of my in_layers anymore (as in prev versions)
# loss = layers.SigmoidCrossEntropy(in_layers=[labels, logits])
model = dc.models.KerasModel(
    keras_model,
    loss=dc.models.losses.L2Loss(),
    # output_types=['prediction', 'loss'],
    batch_size=1000,
    model_dir='siRNA'  # TODO: what's this? Is this to store the generated model?
)


train = dc.data.DiskDataset('train_siRNA')
valid = dc.data.DiskDataset('valid_siRNA')

metric = dc.metrics.Metric(dc.metrics.pearsonr, mode='regression')
for i in range(20):
    model.fit(train, nb_epoch=10)
    print(model.evaluate(train, [metric]))
    print(model.evaluate(valid, [metric]))
