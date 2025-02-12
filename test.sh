#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut -c 1-10)
# Maximum is currently 30g, configurable in your algorithm image settings on grand challenge
MEM_LIMIT="4g"

test_result="Test result: "

for fold in 0
do
for task_name in "Task101_Example_sl_bin_clf" "Task102_Example_sl_mc_clf" "Task103_Example_mednli" "Task104_Example_ml_bin_clf" "Task105_Example_ml_mc_clf" "Task106_Example_sl_reg" "Task107_Example_ml_reg" "Task108_Example_sl_ner" "Task109_Example_ml_ner"
do
    jobname="$task_name-fold$fold"

    echo "=========================================="
    echo "Running test for $jobname..."
    docker volume create dragon_baseline-output-$VOLUME_SUFFIX

    # Do not change any of the parameters to docker run, these are fixed
    docker run --rm \
        --gpus=all \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/test-input/$jobname:/input:ro \
        -v dragon_baseline-output-$VOLUME_SUFFIX:/output/ \
        joeranbosma/dragon_baseline

    docker run --rm \
        -v dragon_baseline-output-$VOLUME_SUFFIX:/output/ \
        python:3.10-slim cat /output/nlp-predictions-dataset.json

    echo ""

    docker run --rm \
        -v dragon_baseline-output-$VOLUME_SUFFIX:/output/ \
        -v $SCRIPTPATH/test-output-expected/$jobname/:/output-expected/ \
        python:3.10-slim python -c "import json, sys; f1 = json.load(open('/output/nlp-predictions-dataset.json')); f2 = json.load(open('/output-expected/nlp-predictions-dataset.json')); sys.exit(f1 != f2);"

    if [ $? -eq 0 ]; then
        echo "Test for $jobname successfully passed..."
        test_result="$test_result $jobname:pass"
    else
        echo "Expected output was not found for $jobname..."
        test_result="$test_result $jobname:fail"
    fi

    docker volume rm dragon_baseline-output-$VOLUME_SUFFIX

done
done

echo "=========================================="
echo "$test_result"
echo "=========================================="
