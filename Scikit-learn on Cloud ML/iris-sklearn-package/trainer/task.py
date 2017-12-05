import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import argparse
import os


MODEL_FILE = 'model.pkl'
LOCAL_DATA_FILE = 'train.data.csv'

INPUT_FEATURES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
TARGET_FEATURE = 'species'

GSUTIL_COMMAND = 'gsutil cp {} {}'


def download_data_from_gcp(gcs_location):
    os.system(GSUTIL_COMMAND.format(gcs_location,LOCAL_DATA_FILE))


def upload_model_to_gcp(gcs_location):
    os.system(GSUTIL_COMMAND.format(MODEL_FILE,gcs_location))


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )

    parser.add_argument(
        '--train-file',
        help='GCS path to training data',
        required=True
    )

    parser.add_argument(
        '--model-dir',
        help='GCS path where the model file will be saved',
        required=True
    )

    args = parser.parse_args()

    download_data_from_gcp(args.train_file)
    print('Train data downloaded from GCS location {}'.format(args.train_file))

    iris_data = pd.read_csv(LOCAL_DATA_FILE, header=0)
    print('Data loaded in Pandas Dataframe')

    X = iris_data[INPUT_FEATURES]
    y = iris_data[TARGET_FEATURE]

    estimator = tree.DecisionTreeClassifier(max_leaf_nodes=3)
    estimator.fit(X=X, y=y)
    print('Estimator is trained...')

    y_predicted = estimator.predict(X=X)
    print('Estimator performed predictions on training data...')

    accuracy = accuracy_score(y, y_predicted)
    print("Train Accuracy: {}".format(accuracy))

    joblib.dump(estimator, MODEL_FILE)
    print('Model is saved locally to {}'.format(MODEL_FILE))

    upload_model_to_gcp(args.model_dir)
    print('Model file is uploaded to GCS location {}'.format(args.model_dir))





if __name__ == '__main__':
    main()