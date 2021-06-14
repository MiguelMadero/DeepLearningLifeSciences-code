import deepchem as dc 

grid_featurizer = dc.feat.RdkitGridFeaturizer(
    voxel_width=2.0,
    feature_types=['hbond', 'salt_bridge', 'pi_stack',
        'cation_pi', 'ecfp', 'splif'],
    sanitize=True, 
    flatten=True
)

tasks, datasets, transformers = dc.molnet.load_pdbbind(
    featurizer="grid", 
    split="random",
    subset="core"
)

train_dataset, valid_dataset, test_dataset = datasets

from sklearn.ensemble import RandomForestRegressor
sklearn_model = RandomForestRegressor(n_estimators=100)
model = dc.models.SklearnModerl(sklearn_model)
model.fit(train_dataset)

# n_features = train_dataset.X.shape[1]
# model = dc.models.MultitaskRgressor(
#     n_tasks=len(pdbbind_tasks),
#     n_features=n_features,
#     layer_sizes=[2000,1000],
#     dropout=0.5,
#     learning_rate=0.0003
# )
# model.fit(train_dataset, nb_epoch=250)