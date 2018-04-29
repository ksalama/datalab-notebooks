#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tensorflow as tf
from tensorflow.python.feature_column import feature_column

import metadata
import task
import input


# **************************************************************************
# YOU MAY IMPLEMENT THUS FUNCTION TO ADD EXTENDED FEATURES
# **************************************************************************

def extend_feature_columns(feature_columns, hyper_params=None):
    """ Use to define additional feature columns, such as bucketized_column, crossed_column,
    and embedding_column. hyper_params can be used to parameterise the creation
    of the extended columns (e.g., embedding dimensions, number of buckets, etc.
    Note that, extensions can be applied on features constructed in process_features function

    Default behaviour is to return the original feature_columns list as-is.

    Args:
        feature_columns: [tf.feature_column] - list of base feature_columns to be extended
        hyper_params: dictionary of hyper-parameters
    Returns:
        [tf.feature_column]: extended feature_column list
    """

    cigarette_use = feature_columns['cigarette_use']
    alcohol_use = feature_columns['alcohol_use']
    mother_age = feature_columns['mother_age']
    mother_race = feature_columns['mother_race']

    cigarette_use_X_alcohol_use = tf.feature_column.crossed_column([cigarette_use, alcohol_use], 9)

    mother_age_bucketized = tf.feature_column.bucketized_column(mother_age,
                                                                boundaries=[18, 22, 28, 32, 36, 40, 42, 45, 50])

    mother_race_X_mother_age_bucketized = tf.feature_column.crossed_column([mother_age_bucketized, mother_race], 120)

    mother_race_X_mother_age_bucketized_embedded = tf.feature_column.embedding_column(
        mother_race_X_mother_age_bucketized, 5)

    feature_columns['cigarette_use_X_alcohol_use'] = cigarette_use_X_alcohol_use
    feature_columns['mother_race_X_mother_age_bucketized'] = mother_race_X_mother_age_bucketized
    feature_columns['mother_race_X_mother_age_bucketized_embedded'] = mother_race_X_mother_age_bucketized_embedded

    return feature_columns


# **************************************************************************
# YOU MAY NOT TO CHANGE THIS FUNCTION TO CREATE FEATURE COLUMNS
# **************************************************************************


def create_feature_columns(hyper_params=None):
    """Creates tensorFlow feature_column definitions based on the metadata of the features.

    the tensorFlow feature_column objects are created based on the data types of the features
    defined in the metadata.py module. Extended featured (if any) are created, based on the base features,
    as the extend_feature_columns method is called.

    Returns:
      {string: tf.feature_column}: dictionary of name:feature_column .
    """

    # load the numeric feature stats (if exist)
    feature_stats = input.load_feature_stats(hyper_params)

    # all the numerical feature including the input and constructed ones
    numeric_feature_names = set(metadata.INPUT_NUMERIC_FEATURE_NAMES + metadata.CONSTRUCTED_NUMERIC_FEATURE_NAMES)

    # create t.feature_column.numeric_column columns without scaling
    numeric_columns = {feature_name: tf.feature_column.numeric_column(feature_name, normalizer_fn=None)
                           for feature_name in numeric_feature_names}

    # all the categorical feature with identity including the input and constructed ones
    categorical_feature_names_with_identity = metadata.INPUT_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY
    categorical_feature_names_with_identity.update(metadata.CONSTRUCTED_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY)

    # create tf.feature_column.categorical_column_with_identity columns
    categorical_columns_with_identity = \
        {item[0]: tf.feature_column.categorical_column_with_identity(item[0], item[1])
         for item in categorical_feature_names_with_identity.items()}

    # create tf.feature_column.categorical_column_with_vocabulary_list columns
    categorical_columns_with_vocabulary = \
        {item[0]: tf.feature_column.categorical_column_with_vocabulary_list(item[0], item[1])
         for item in metadata.INPUT_CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.items()}

    # create tf.feature_column.categorical_column_with_hash_bucket columns
    categorical_columns_with_hash_bucket = \
        {item[0]: tf.feature_column.categorical_column_with_hash_bucket(item[0], item[1])
         for item in metadata.INPUT_CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET.items()}

    feature_columns = {}

    if numeric_columns is not None:
        feature_columns.update(numeric_columns)

    if categorical_columns_with_identity is not None:
        feature_columns.update(categorical_columns_with_identity)

    if categorical_columns_with_vocabulary is not None:
        feature_columns.update(categorical_columns_with_vocabulary)

    if categorical_columns_with_hash_bucket is not None:
        feature_columns.update(categorical_columns_with_hash_bucket)

    # add extended feature definitions before returning the feature_columns list
    return extend_feature_columns(feature_columns, hyper_params)


# **************************************************************************
# YOU MAY NOT TO CHANGE THIS FUNCTION TO DEFINE WIDE AND DEEP COLUMNS
# **************************************************************************


def get_deep_and_wide_columns(feature_columns, use_indicators=True, use_wide_columns=True):
    """Creates deep and wide feature column lists.

    given a list of feature_column, each feature_column is categorised as either:
    1) dense, if the column is tf.feature_column._NumericColumn or feature_column._EmbeddingColumn,
    2) categorical, if the column is tf.feature_column._VocabularyListCategoricalColumn or
    tf.feature_column._BucketizedColumn, or
    3) sparse, if the column is tf.feature_column._HashedCategoricalColumn or tf.feature_column._CrossedColumn.

    if use_indicators=True, then categorical_columns are converted into indicator_column(s), and used as dense features
    in the deep part of the model. if use_wide_columns=True, then categorical_columns are used as sparse features
    in the wide part of the model.

    deep_columns = dense_columns + indicator_columns
    wide_columns = categorical_columns + sparse_columns

    Args:
        feature_columns: [tf.feature_column] - A list of tf.feature_column objects.
        use_indicators: bool - if True, then categorical_columns are converted into tf.feature_column.indicator_column
        use_wide_columns: bool - if True, categorical_columns are treated wide columns

    Returns:
        [tf.feature_column],[tf.feature_column]: deep and wide feature_column lists.
    """
    dense_columns = list(
        filter(lambda column: isinstance(column, feature_column._NumericColumn) |
                              isinstance(column, feature_column._EmbeddingColumn),
               feature_columns)
    )

    categorical_columns = list(
        filter(lambda column: isinstance(column, feature_column._VocabularyListCategoricalColumn) |
                              isinstance(column, feature_column._IdentityCategoricalColumn) |
                              isinstance(column, feature_column._BucketizedColumn),
               feature_columns)
    )

    sparse_columns = list(
        filter(lambda column: isinstance(column, feature_column._HashedCategoricalColumn) |
                              isinstance(column, feature_column._CrossedColumn),
               feature_columns)
    )

    indicator_columns = []

    if use_indicators:
        indicator_columns = list(
            map(lambda column: tf.feature_column.indicator_column(column),
                categorical_columns)
        )

    deep_columns = dense_columns + indicator_columns
    wide_columns = sparse_columns + (categorical_columns if use_wide_columns else None)

    return deep_columns, wide_columns
