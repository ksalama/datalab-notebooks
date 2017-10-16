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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers
import tensorflow.contrib.learn as tflearn
from tensorflow.contrib import metrics
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

CSV_COLUMNS = 'datetime,dayofweek,hourofday,pickuplon,pickuplat,dropofflon,dropofflat,passengers,fare_amount'.split(',')
SCALE_COLUMNS = ['pickuplon','pickuplat','dropofflon','dropofflat','passengers']
LABEL_COLUMN = 'fare_amount'
DEFAULTS = [ ['NULL'],['Sun'], [0], [-74.0], [40.0], [-74.0], [40.7], [1.0], [0.0]]
UNUSED_COLUMNS = ['datetime']

INPUT_COLUMN_NAMES = {
    'dayofweek':tf.string,
    'hourofday':tf.int32,
    'pickuplon':tf.float32,
    'pickuplat':tf.float32,
    'dropofflon':tf.float32,
    'dropofflat':tf.float32,
    'passengers':tf.int32
}

# These are the raw input columns, and will be provided for prediction also
INPUT_COLUMNS = [
    # define features
    layers.sparse_column_with_keys('dayofweek', keys=['Sun', 'Mon', 'Tues', 'Wed', 'Thu', 'Fri', 'Sat']),
    layers.sparse_column_with_integerized_feature('hourofday', bucket_size=24),

    # engineered features that are created in the input_fn
    layers.real_valued_column('latdiff'),
    layers.real_valued_column('londiff'),
    layers.real_valued_column('euclidean'),

    # real_valued_column
    layers.real_valued_column('pickuplon'),
    layers.real_valued_column('pickuplat'),
    layers.real_valued_column('dropofflat'),
    layers.real_valued_column('dropofflon'),
    layers.real_valued_column('passengers'),
]

def build_estimator(model_dir, nbuckets, hidden_units):
  """
     Build an estimator starting from INPUT COLUMNS.
     These include feature transformations and synthetic features.
     The model is a wide-and-deep model.
  """

  # input columns
  (dayofweek, hourofday, latdiff, londiff, euclidean, plon, plat, dlon, dlat, pcount) = INPUT_COLUMNS 

  # bucketize the lats & lons
  latbuckets = np.linspace(38.0, 42.0, nbuckets).tolist()
  lonbuckets = np.linspace(-76.0, -72.0, nbuckets).tolist()
  b_plat = layers.bucketized_column(plat, latbuckets)
  b_dlat = layers.bucketized_column(dlat, latbuckets)
  b_plon = layers.bucketized_column(plon, lonbuckets)
  b_dlon = layers.bucketized_column(dlon, lonbuckets)

  # feature cross
  ploc = layers.crossed_column([b_plat, b_plon], nbuckets*nbuckets)
  dloc = layers.crossed_column([b_dlat, b_dlon], nbuckets*nbuckets)
  pd_pair = layers.crossed_column([ploc, dloc], nbuckets ** 4 )
  day_hr =  layers.crossed_column([dayofweek, hourofday], 24*7)

  # Wide columns and deep columns.
  wide_columns = [
      # feature crosses
      dloc, ploc, pd_pair,
      day_hr,

      # sparse columns
      dayofweek, hourofday,

      # anything with a linear relationship
      pcount 
  ]

  deep_columns = [
      # embedding_column to "group" together ...
      layers.embedding_column(pd_pair, 10),
      layers.embedding_column(day_hr, 10),

      # real_valued_column
      plat, plon, dlat, dlon,
      latdiff, londiff, euclidean
  ]

  return tf.contrib.learn.DNNLinearCombinedRegressor(
      model_dir=model_dir,
      linear_feature_columns=wide_columns,
      dnn_feature_columns=deep_columns,
      dnn_hidden_units=hidden_units or [128, 32, 4])

def add_engineered(features):
    # this is how you can do feature engineering in TensorFlow
    lat1 = features['pickuplat']
    lat2 = features['dropofflat']
    lon1 = features['pickuplon']
    lon2 = features['dropofflon']
    latdiff = (lat1 - lat2)
    londiff = (lon1 - lon2)
    # set features for distance with sign that indicates direction
    features['latdiff'] = latdiff
    features['londiff'] = londiff
    dist = tf.sqrt(latdiff*latdiff + londiff*londiff)
    features['euclidean'] = dist
    return features   

def serving_input_fn():
    inputs = {}
    for name, type in INPUT_COLUMN_NAMES.items():
        inputs[name] = tf.placeholder(shape=[None], dtype=type)

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in inputs.items()
    }
    return tf.contrib.learn.InputFnOps(add_engineered(features), None, inputs)


def generate_csv_input_fn(filename, num_epochs=None, batch_size=512, mode=tf.contrib.learn.ModeKeys.TRAIN):
  def _input_fn():
    # could be a path to one file or a file pattern.
    input_file_names = tf.train.match_filenames_once(filename)
    #input_file_names = [filename]

    filename_queue = tf.train.string_input_producer(
        input_file_names, num_epochs=num_epochs, shuffle=True)
    reader = tf.TextLineReader()
    _, value = reader.read_up_to(filename_queue, num_records=batch_size)

    value_column = tf.expand_dims(value, -1)

    columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)

    features = dict(zip(CSV_COLUMNS, columns))
    
    for u in UNUSED_COLUMNS:
        features.pop(u)

    label = features.pop(LABEL_COLUMN)

    return add_engineered(features), label

  return _input_fn




def get_eval_metrics():
  return {
     'rmse': tflearn.MetricSpec(metric_fn=metrics.streaming_root_mean_squared_error),
     'training/hptuning/metric': tflearn.MetricSpec(metric_fn=metrics.streaming_root_mean_squared_error),
  }
