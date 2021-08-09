from model import DRModel
from data import load_images_DR


train, valid, test = load_images_DR(split='random', seed=123)

model = DRModel(
    n_init_kernel=32,
    batch_size=32,
    learning_rate=1e-5,
    augment=True,
    model_dir='./test_model'
)

