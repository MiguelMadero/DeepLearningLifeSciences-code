# variational autoencoder to generate SMILES molecules
import deepchem as dc
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers


tasks, datasets, transformers = dc.molnet.load_muv()
train_dataset, valid_dataset, test_dataset = datasets
train_smiles = train_dataset.ids

tokens = set()
for s in train_smiles:
    tokens = tokens.union(set(s))
tokens = sorted(list(tokens))
max_length = max(len(s) for s in train_smiles)

print("max_length: %n", max_length) #82

from deepchem.models.optimizers import Adam, ExponentialDecay
from deepchem.models.seqtoseq import AspuruGuzikAutoEncoder
batch_size = 100
batches_per_epoch = len(train_smiles)/batch_size
learning_rate = ExponentialDecay(0.001, 0.95, batches_per_epoch)
import pdb;pdb.set_trace()
model = AspuruGuzikAutoEncoder(tokens, max_length, model_dir='vae', batch_size=batch_size, learning_rate=learning_rate)


print("batches_per_epoch: %n", batches_per_epoch)
# model.set_optimizer(Adam(learning_rate=learning_rate))

def generate_sequences(epochs):
    for i in range(epochs):
        for s in train_smiles:
            yield (s, s)

model.fit_sequences(generate_sequences(50))

from rdkit import Chem
predictions = model.predict_from_embeddings(np.random.normal(size=(1000,196)))
molecules = []
for p in predictions:
    smiles = ''.join(p)
    if Chem.MolFromSmiles(smiles) is not None:
        molecules.append(smiles)
print()
print('Generated molecules:')
for m in molecules:
    print(m)
