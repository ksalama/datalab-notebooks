#!/bin/bash

echo "Submitting a Cloud ML Engine job..."

REGION=europe-west1
TIER=BASIC
BUCKET=ksalama-gcs-cloudml

MODEL_NAME="iris_estimator"

PACKAGE_PATH=trainer
TRAIN_FILE=gs://${BUCKET}/data/iris/iris.data.csv
MODEL_DIR=gs://${BUCKET}/ml-models/iris_estimators

CURRENT_DATE=`date +%Y%m%d_%H%M%S`
JOB_NAME=train_${MODEL_NAME}_${CURRENT_DATE}

gcloud ml-engine jobs submit training ${JOB_NAME} \
        --job-dir=${MODEL_DIR}/job_dir \
        --runtime-version=1.2 \
        --region=${REGION} \
        --scale-tier=${TIER} \
        --module-name=trainer.task \
        --package-path=${PACKAGE_PATH} \
        -- \
        --train-file=${TRAIN_FILE} \
        --model-dir=${MODEL_DIR}
