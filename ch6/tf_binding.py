import deepchem as dc
import tensorflow as tf
import tensorflow.keras.layers as layers

features = tf.keras.Input(shape=(101, 4))
# In the keras version we're not specifying this... why and how?
# labels = layers.Label(shape=(1))
# weights = layers.Weights(shape=(1))

prev = features
for i in range(3):
    convLayer = layers.Conv1D(
        filters=15,
        kernel_size=10,
        activation=tf.nn.relu,
        padding='same')
    prev = convLayer(prev)
    dropoutLayer = layers.Dropout(rate=0.5)
    prev = dropoutLayer(prev)

logitsLayer = layers.Dense(units=1)
logits = logitsLayer(layers.Flatten()(prev))

# sets to 0/1 as opposed to a random range of probability
output = layers.Activation(tf.math.sigmoid)(logits)
# TODO: what does it mean to have an array of outputs? How does this relates to loss
# In the previous version we connected them sequentially, the output was output
# but the loss function still relied on logits
keras_model = tf.keras.Model(inputs=features, outputs=[output, logits])

loss = dc.models.losses.SigmoidCrossEntropy()
# Why is it that loss doesn't need to know the structure of my in_layers anymore.
# loss = layers.SigmoidCrossEntropy(in_layers=[labels, logits])
# That's because we specify it below as output_types loss and above as outputs=[_,logits]
# Not sure about labels...
model = dc.models.KerasModel(
    keras_model,
    loss=loss,
    output_types=['prediction', 'loss'],
    batch_size=1000,
    model_dir='tf'  # TODO: what's this? Is this to store the generated model?
)


train = dc.data.DiskDataset('train_dataset')
valid = dc.data.DiskDataset('valid_dataset')

metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
for i in range(20):
    model.fit(train, nb_epoch=10)
    print(model.evaluate(train, [metric]))
    print(model.evaluate(valid, [metric]))
