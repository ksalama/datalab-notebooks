import tensorflow as tf
import metadata
import preprocess
from tensorflow.python.feature_column import feature_column


def create_feature_columns():
    """Creates tensorflow feature_column definitions based on the metadata of the features.

    the tensorflow feature_column objects are created based on the data types of the features
    defined in the metadata.py module. Extended featured (if any) are created, based on the base features,
    as the preprocess.extend_feature_columns method is called.

    Returns:
      {string: tf.feature_column}: dictionary of name:feature_column .
    """

    numeric_columns = {feature_name: tf.feature_column.numeric_column(feature_name)
                       for feature_name in metadata.NUMERIC_FEATURE_NAMES}

    categorical_column_with_vocabulary = {
    item[0]: tf.feature_column.categorical_column_with_vocabulary_list(item[0], item[1])
    for item in metadata.CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.items()}

    categorical_column_with_hash_bucket = {
    item[0]: tf.feature_column.categorical_column_with_hash_bucket(item[0], item[1])
    for item in metadata.CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET.items()}

    feature_columns = {}

    if numeric_columns is not None:
        feature_columns.update(numeric_columns)

    if categorical_column_with_vocabulary is not None:
        feature_columns.update(categorical_column_with_vocabulary)

    if categorical_column_with_hash_bucket is not None:
        feature_columns.update(categorical_column_with_hash_bucket)

    # add extended feature definitions before returning the feature_columns list
    return preprocess.extend_feature_columns(feature_columns)


def get_deep_and_wide_columns(feature_columns, embedding_size=0, use_indicators=True):
    """Creates deep and wide feature column lists.

    given a list of feature_column, each feature_column is categorised as either:
    1) dense, if the column is tf.feature_column._NumericColumn,
    2) categorical, if the column is tf.feature_column._VocabularyListCategoricalColumn or tf.feature_column._BucketizedColumn, or
    3) sparse, if the column is tf.feature_column._HashedCategoricalColumn or tf.feature_column._CrossedColumn.

    if use_indicators=True, then  categorical_columns are converted into indicator_column.
    if embedding_size > 0, then sparse_columns are converted to tf.feature_column.embedding_column (using the embedding_size).

    deep_columns = dense_columns + indicator_columns + embedding_columns
    wide_columns = categorical_columns + sparse_columns

    Args:
        feature_columns: [tf.feature_column] - A list of tf.feature_column objects.
        embedding_size: int - if greater than 0, then sparse_columns are converted to tf.feature_column.embedding_column.
        use_indicators: bool - if True, then categorical_columns are converted into tf.feature_column.indicator_column.
    Returns:
        [tf.feature_column],[tf.feature_column]: deep and wide feature_column lists.
    """
    dense_columns = list(
        filter(lambda column: isinstance(column, feature_column._NumericColumn),
               feature_columns
               )
    )

    categorical_columns = list(
        filter(lambda column: isinstance(column, feature_column._VocabularyListCategoricalColumn) |
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

    embedding_columns = []

    if embedding_size > 0:
        embedding_columns = list(
            map(lambda sparse_column: tf.feature_column.embedding_column(sparse_column, dimension=embedding_size),
                sparse_columns)
        )

    deep_columns = dense_columns + indicator_columns + embedding_columns
    wide_columns = categorical_columns + sparse_columns

    return deep_columns, wide_columns
