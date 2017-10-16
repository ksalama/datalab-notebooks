import parameters
import tensorflow as tf
import numpy as np


def extend_feature_columns(feature_columns):
    """ Use to define additional feature columns, such as bucketized_column and crossed_column
    Default behaviour is to return the original feature_column list as is

    Args:
        feature_columns: [tf.feature_column] - list of base feature_columns to be extended
    Returns:
        [tf.feature_column]: extended feature_column list
    """

    # examples - given:
    # 'x' and 'y' are two numeric features:
    # 'alpha' and 'beta' are two categorical features

    # feature_columns['alpha_X_beta'] = tf.feature_column.crossed_column(
    #     [feature_columns['alpha'], feature_columns['beta']], int(1e4))
    #
    # num_buckets = parameters.HYPER_PARAMS.num_buckets
    # buckets = np.linspace(-2, 2, num_buckets).tolist()
    #
    # feature_columns['x_bucketized'] = tf.feature_column.bucketized_column(
    #     feature_columns['x'], buckets)
    #
    # feature_columns['y_bucketized'] = tf.feature_column.bucketized_column(
    #     feature_columns['y'], buckets)
    #
    # feature_columns['x_bucketized_X_y_bucketized'] = tf.feature_column.crossed_column(
    #     [feature_columns['x_bucketized'], feature_columns['y_bucketized']], int(1e4))

    return feature_columns


def process_features(features):
    """ Use to implement custom feature engineering logic, e.g. polynomial expansion
    Default behaviour is to return the original feature tensors dictionary as is

    Args:
        features: {string:tensors} - dictionary of feature tensors
    Returns:
        {string:tensors}: extended feature tensor dictionary
    """

    # examples - given:
    # 'x' and 'y' are two numeric features:
    # 'alpha' and 'beta' are two categorical features

    # features['x_2'] = tf.pow(features['x'],2)
    # features['y_2'] = tf.pow(features['y'], 2)
    # features['xy'] = features['x'] * features['y']
    # features['sin_x'] = tf.sin(features['x'])
    # features['cos_y'] = tf.cos(features['x'])
    # features['log_xy'] = tf.log(features['xy'])
    # features['sqrt_xy'] = tf.sqrt(features['xy'])
    # features['x_grt_y'] = features['x'] > features['y']

    return features

