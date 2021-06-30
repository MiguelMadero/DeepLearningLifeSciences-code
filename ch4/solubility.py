import deepchem as dc
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

tasks, datasets, transformers = dc.molnet.load_delaney(featurized='GraphConv')
train_dataset, valid_dataset, test_dataset = datasets

model = dc.models.GraphConvModel(n_tasks=1, mode='regression', dropout=0.2)
model.fit(train_dataset, nb_epoch=100)
