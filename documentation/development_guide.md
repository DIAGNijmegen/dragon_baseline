# DRAGON Development Guide

Submissions to the DRAGON challenge need to generate predictions for the test set of each task. The predictions should be stored in a specific format, as described in the [dataset convention](/documentation/dataset_convention.md). The predictions are then evaluated on the Grand Challenge platform. To generate the predictions, examples from the training set can be used. These examples can be used to fine-tune a model, provide few-shot examples, or be used in any other way to generate predictions for the test set. For model selection, the validation set can be used. To facilitate the development of algorithms, we provide a [submission template](https://github.com/DIAGNijmegen/dragon_submission) that solves each task using a BERT-like model, and we provide synthetic datasets that can be used for debugging and testing.

Note: the validation set may not be used as training data, such that the robustness of the algorithm can be assessed through the cross-validation setup of the DRAGON benchmark.

## Format for submissions
Submissions to the DRAGON challenge are to be made as Docker containers. Thse Docker containers must encapsulate all **training resources** (e.g., pretrained model weights, fine-tuning strategy, and/or prompting strategy with few-shot examples) and the components needed to **generate predictions** for the test cases. The flow for submissions is shown below. Technically, these containers are [Grand Challenge (GC) algorithms](https://grand-challenge.org/documentation/algorithms/) with standardised input and output data handling.

![DRAGON_benchmark_flowdiagram](DRAGON_benchmark_flowdiagram.png)
*Figure: Evaluation method for the DRAGON benchmark. Challenge participants must provide all resources necessary to process the reports and generate predictions for the test set. Any processing of reports is performed on the Grand Challenge platform, without any interaction with the participant.*


## Setup for development
The DRAGON baseline algorithm provides a common solution to all tasks in the DRAGON benchmark. This algoritm was evaluated on the DRAGON benchmark across five architectures and three pretraining strategies, and some of its strengths and weaknesses are described [here](/README.md#where-does-the-dragon-baseline-perform-well-and-where-does-it-not-perform).

Rather than directly adapting the DRAGON baseline repository, we recommend making a fork of the [DRAGON submission repository](https://github.com/DIAGNijmegen/dragon_submission) as a template for your solution. This brings in everyting of the DRAGON baseline, as well as documentation on how to make code changes. A clear benefit from starting with the [DRAGON submission repository](https://github.com/DIAGNijmegen/dragon_submission) is that code changes made by you stand out from the code in the DRAGON benchmark. This in turn makes it easy to maintain and upgrade to improved versions of the baseline once those come along.

After making the fork, clone it. For the sake of this tutorial, we will assume you put your repositories in the `~/repos` folder (feel free to change this to any other directory). Please replace `{YOURUSERNAME}` with your GitHub username.

```bash
cd ~/repos
git clone https://github.com/{YOURUSERNAME}/dragon_submission
```

This brings in all necessary steps for data loading, validation, preprocessing, training, inference, and storing and verifying the predictions. Every aspect of the baseline approach can be adapted, for which we provide examples in the aforementioned repository.


## Validating the setup
Before implementing your algorithm using this template, we recommend to test whether setup was completed successfully. This will also test whether your hardware setup is suitable. The baseline was tested on [these systems](documentation/system_requirements.md). 


### Working in Docker
If you prefer development in Docker, start with building the Docker container:

```bash
cd ~/repos/dragon_submission
./build.sh
```

If ran successfully, this should result in the Docker container named `joeranbosma/dragon_submission:latest`.

Then, test your setup by training and evaluating on the synthetic datasets. This will fine-tune the `distilbert-base-multilingual-cased` for each of the nine synthetic debugging datasets, so this can take a while. To do so, run:

```bash
cd ~/repos/dragon_submission
./test.sh
```

If you want to adapt the hardware (e.g., run on CPU instead of GPU, or allow more/less RAM, CPU cores, etc.) you can adapt the `test.sh` file.


### Working in an IDE (e.g., Visual Studio Code)
Alternatively, you can develop outside a Docker container. This introduces more differences between the development environment and the submission environment on Grand Challenge, but also allows for more flexibility for interaction. For this route, we strongly recommend that you install everyting in a virtual environment! Pip or anaconda are both fine. Use a recent version of Python! 3.9 or newer is guaranteed to work! For the sake of this tutorial, we will use a conda environment with Python 3.10.

```bash
conda create --name=dragon_submission python=3.10
conda activate dragon_submission
cd ~/repos/dragon_submission
pip install -r requirements.txt
```

This was tested on Ubuntu 22.04. If you have issues installing the `requirements.txt`, you can try the adapted variant below. This should install all requirements, but doesn't fully specify all versions and dependencies:

```bash
cd ~/repos/dragon_submission
pip install torch xformers==0.0.21
pip install -r requirements.in
```

If this was all successful, you can open the repository in an IDE and select the `dragon_submission` environment to run the code in.

To validate the setup works as intended, run the `test.py` script to train on the synthetic datasets. This will fine-tune the `distilbert-base-multilingual-cased` for each of the nine synthetic debugging datasets, so this can take a while.


## Developing
After the setup above you're good to go! The most logical place to start adapting is the `process.py` script.

For more information about submission to the DRAGON benchmark, please check out the [algorithm submission guide](https://dragon.grand-challenge.org/submission/).


### Bringing in your own data
To format your own dataset for usage with the DRAGON benchmark, check out the [dataset convention](/documentation/dataset_convention.md).
