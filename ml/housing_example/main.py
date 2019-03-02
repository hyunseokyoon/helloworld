from __future__ import print_function

import math

from IPython import display
import numpy as np
import tensorflow as tf
import pandas as pd

from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt

from sklearn import metrics
from tensorflow.python.data import Dataset


def preprocess_features(california_housing_dataframe):
    selected_features = california_housing_dataframe[['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                                                      'total_bedrooms', 'population', 'households', 'median_income']]
    processed_features = selected_features.copy()

    # synthetic feature
    processed_features["room_per_person"] = (california_housing_dataframe["total_rooms"] / california_housing_dataframe["population"])

    # # handle outlier
    # processed_features["room_per_person"] = processed_features["room_per_person"].apply(
    #     lambda x: min(x, 5))

    return processed_features


def construct_feature_column(input_features):
    return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])


def preprocess_targets(california_housing_dataframe):
    output_targets = pd.DataFrame()

    output_targets["median_house_value"] = california_housing_dataframe["median_house_value"] / 1000.0

    return output_targets


def check_geo(examples, targets):
    plt.figure(figsize=(13, 8))

    ax = plt.subplot(1, 2, 1)
    ax.set_ylim([32, 43])
    ax.set_xlim([-125, -114])
    ax.set_autoscaley_on(False)
    ax.set_autoscalex_on(False)
    plt.scatter(examples['longitude'], examples['latitude'], cmap="coolwarm",
                c=targets["median_house_value"] / targets["median_house_value"].max())

    plt.show()


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """
    :return: (features, labels)
    """
    features = {key: np.array(value) for key, value in dict(features).items()}

    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds.shuffle(buffer_size=10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def train_model(learning_rate, steps, batch_size,
                training_examples,
                training_targets,
                validation_examples,
                validation_targets):
    periods = 10
    steps_per_periods = steps / periods

    # model
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=construct_feature_column(training_examples),
        optimizer=my_optimizer
    )

    training_input_fn = lambda: my_input_fn(
        training_examples,
        training_targets["median_house_value"],
        batch_size=batch_size)

    predict_training_input_fn = lambda: my_input_fn(
        training_examples,
        training_targets["median_house_value"],
        num_epochs=1,
        shuffle=False)

    predict_validation_input_fn = lambda: my_input_fn(
        validation_examples,
        validation_targets["median_house_value"],
        num_epochs=1,
        shuffle=False)

    # training
    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []
    for period in range(0, periods):
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_periods
        )

        training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item["predictions"][0] for item in training_predictions])

        validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item["predictions"][0] for item in validation_predictions])

        training_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(validation_predictions, validation_targets))

        print("  error of period %02d : training=%0.2f, validation=%0.2f" % (period, training_root_mean_squared_error, validation_root_mean_squared_error))

        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)

    print("Model training finished")

    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title("Root Mean Squared Error vs. Periods")

    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()

    plt.show()

    # test
    california_housing_test_data = pd.read_csv(
        "https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv", sep=",")

    test_features = preprocess_features(california_housing_test_data)
    test_targets = preprocess_targets(california_housing_test_data)

    predict_test_input_fn = lambda: my_input_fn(
        test_features,
        test_targets["median_house_value"],
        num_epochs=1,
        shuffle=False
    )

    test_prediction = linear_regressor.predict(input_fn=predict_test_input_fn)
    test_prediction = np.array([item["predictions"][0] for item in test_prediction])

    test_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(test_prediction, test_targets))
    print("Final RMSE (on test data): %0.2f" % test_root_mean_squared_error)

# main
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.max_columns = 100
pd.options.display.float_format = '{:.1f}'.format

# load data
california_housing_dataframe = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

# prepare data
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))

training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_targets = preprocess_targets(california_housing_dataframe.head(12000))

validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))

train_model(
    learning_rate=0.00003,
    steps=500,
    batch_size=5,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets
)