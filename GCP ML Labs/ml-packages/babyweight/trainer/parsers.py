import tensorflow as tf
import metadata


def parse_csv(rows_string_tensor):
    """Takes the string input tensor and returns a dict of rank-2 tensors.

    Takes a rank-1 tensor and converts it into rank-2 tensor, with respect to its data type
    (inferred from the HEADER_DEFAULTS metadata), as well as removing unused columns

    Example if the data is ['csv,line,1', 'csv,line,2', ..] to
    [['csv,line,1'], ['csv,line,2']] which after parsing will result in a
    tuple of tensors: [['csv'], ['csv']], [['line'], ['line']], [[1], [2]]

    Args:
        rows_string_tensor: rank-1 tensor of type string
    Returns:
        rank-2 tensor of the correct data type
    """

    row_columns = tf.expand_dims(rows_string_tensor, -1)
    columns = tf.decode_csv(row_columns, record_defaults=metadata.HEADER_DEFAULTS)
    features = dict(zip(metadata.HEADERS, columns))

    return features
