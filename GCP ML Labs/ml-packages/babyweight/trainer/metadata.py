# task type can be either 'classification' or 'regression', based on the target feature in the dataset
TASK_TYPE = 'regression'

# the header (all column names) of the input data file(s)
HEADERS = 'weight_pounds,is_male,mother_age,mother_race,plurality,gestation_weeks,mother_married,cigarette_use,alcohol_use,key'.split(',')

# the default values of all the columns of the input data, to help TF detect the data types of the columns
HEADER_DEFAULTS = [[0.0], ['null'], [0.0], ['null'], [0.0], [0.0], ['null'], ['null'], ['null'], ['nokey']]

# column of type int or float
NUMERIC_FEATURE_NAMES = ["mother_age", "plurality", "gestation_weeks"]

# categorical features with few values (to be encoded as one-hot indicators)
CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY = {
    'mother_race': ['White', 'Black', 'American Indian', 'Chinese',
               'Japanese', 'Hawaiian', 'Filipino', 'Unknown',
               'Asian Indian', 'Korean', 'Samaon', 'Vietnamese'],

    'is_male': ['True', 'False'],
    'mother_married': ['True', 'False'],
    'cigarette_use': ['True', 'False', 'None'],
    'alcohol_use': ['True', 'False', 'None']
}

# categorical features with many values (to be treated using embedding)
CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET = {}

# all the categorical feature names
CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.keys()) \
                            + list(CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET.keys())

# all the feature names to be used in the model
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES

# target feature name (response or class variable)
TARGET_NAME = 'weight_pounds'

# column to be ignores (e.g. keys, constants, etc.)
UNUSED_FEATURE_NAMES = ['key']