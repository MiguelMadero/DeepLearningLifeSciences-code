from sklearn.ensemble import RandomForestRegressor
import deepchem as dc

grid_featurizer = dc.feat.RdkitGridFeaturizer(
    voxel_width=2.0,
    feature_types=['hbond', 'salt_bridge', 'pi_stack',
                   'cation_pi', 'ecfp', 'splif'],
    sanitize=True,
    flatten=True
)

tasks, datasets, transformers = dc.molnet.load_pdbbind(
    featurizer=grid_featurizer,
    splitter="random",
    subset="core"
)

train_dataset, valid_dataset, test_dataset = datasets

sklearn_model = RandomForestRegressor(n_estimators=100)
randomForestModel = dc.models.SklearnModel(
    sklearn_model, model_dir="pdbbind_rf")

# Evaluate it.


def evaluate(model, transformers):
    metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
    train_scores = model.evaluate(train_dataset, [metric], transformers)
    test_scores = model.evaluate(test_dataset, [metric], transformers)
    print("Train scores")
    print(train_scores)
    print("Test scores")
    print(test_scores)


# With NN
n_features = train_dataset.X.shape[1]
multitaskRegressorModel = dc.models.MultitaskRegressor(
    n_tasks=len(tasks),
    n_features=n_features,
    layer_sizes=[2000, 1000],
    dropout=0.5,
    model_dir="pdbbind_nn",
    learning_rate=0.0003
)

# fit models
# randomForestModel.fit(train_dataset)
multitaskRegressorModel.fit(train_dataset, nb_epoch=250)

# evaluate(randomForestModel, transformers)
evaluate(multitaskRegressorModel, transformers)
