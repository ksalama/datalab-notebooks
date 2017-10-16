import tensorflow as tf
import metadata
import parsers
import preprocess
import multiprocessing


def generate_text_input_fn(file_names,
                           mode,
                           parser_fn=parsers.parse_csv,
                           skip_header_lines=0,
                           num_epochs=None,
                           batch_size=200
                           ):
    """Generates an input function for training or evaluation.
    This uses the input pipeline based approach using file name queue
    to read data so that entire data is not loaded in memory.

    Args:
        file_names: [str] - list of text files to read data from.
        mode: tf.contrib.learn.ModeKeys - either TRAIN or EVAL.
            Used to determine whether or not to randomize the order of data.
        parser_fn: A function that parses text files (e.g., csv parser, fixed-width parser, etc.
        skip_header_lines: int set to non-zero in order to skip header lines
          in CSV files.
        num_epochs: int - how many times through to read the data.
          If None will loop through data indefinitely
        batch_size: int - first dimension size of the Tensors returned by
          input_fn
    Returns:
        A function () -> (features, indices) where features is a dictionary of
          Tensors, and indices is a single Tensor of label indices.
    """

    shuffle = True if mode == tf.contrib.learn.ModeKeys.TRAIN else False

    filename_queue = tf.train.string_input_producer(
        file_names, num_epochs=num_epochs, shuffle=shuffle)

    reader = tf.TextLineReader(skip_header_lines=skip_header_lines)

    _, rows = reader.read_up_to(filename_queue, num_records=batch_size)

    features = parser_fn(rows)

    # Remove unused columns
    for col in metadata.UNUSED_FEATURE_NAMES:
        features.pop(col)

    if shuffle:
        features = tf.train.shuffle_batch(
            features,
            batch_size,
            min_after_dequeue=2 * batch_size + 1,
            capacity=batch_size * 10,
            num_threads=multiprocessing.cpu_count(),
            enqueue_many=True,
            allow_smaller_final_batch=True
        )
    else:
        features = tf.train.batch(
            features,
            batch_size,
            capacity=batch_size * 10,
            num_threads=multiprocessing.cpu_count(),
            enqueue_many=True,
            allow_smaller_final_batch=True
        )

    target = get_target(features.pop(metadata.TARGET_NAME))
    return preprocess.process_features(features), target


def parse_label_column(label_string_tensor):
    table = tf.contrib.lookup.index_table_from_tensor(tf.constant(metadata.TARGET_LABELS))
    return table.lookup(label_string_tensor)


def get_target(target):
    if metadata.TASK_TYPE == "classification":
        return parse_label_column(target)
    else:
        return target

