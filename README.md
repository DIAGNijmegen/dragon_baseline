# DRAGON Baseline Algorithm

This repository provides the baseline code for the [DRAGON Challenge](https://dragon.grand-challenge.org).

If you are using DRAGON resources, please cite the following article:

> J. S. Bosma, K. Dercksen, L. Builtjes, R. André, C, Roest, S. J. Fransen, C. R. Noordman, M. Navarro-Padilla, J. Lefkes, N. Alves, M. J. J. de Grauw, L. van Eekelen, J. M. A. Spronck, M. Schuurmans, A. Saha, J. J. Twilt, W. Aswolinskiy, W. Hendrix, B. de Wilde, D. Geijs, J. Veltman, D. Yakar, M. de Rooij, F. Ciompi, A. Hering, J. Geerdink, and H. Huisman on behalf of the DRAGON consortium. The DRAGON benchmark for clinical NLP. *npj Digital Medicine* 8, 289 (2025). [https://doi.org/10.1038/s41746-025-01626-x](https://doi.org/10.1038/s41746-025-01626-x)

Download the citation file for your reference manager: [BibTeX](https://github.com/DIAGNijmegen/dragon/blob/main/citation.bib) | [RIS](https://github.com/DIAGNijmegen/dragon/blob/main/citation.ris)

##

## Installation instructions

We strongly recommend that you install the DRAGON baseline in a virtual environment! Pip or anaconda are both fine. Use a recent version of Python! 3.9 or newer is guaranteed to work!

```bash
pip install dragon_baseline
```

## How to get started as AI developer?
Please see the [dedicated development guide](documentation/development_guide.md) to get started with creating new solutions!


## What is the DRAGON benchmark?

The DRAGON benchmark serves as an extensive resource for testing and advancing clinical NLP algorithms, particularly in the realm of automated data curation. In the context of medical imaging datasets, data curation involves selecting relevant studies, collecting key measurements, and determining the clinical outcomes as labels. Clinical reports are the primary source for these curation tasks. The DRAGON benchmark aims to catalyze the development of algorithms capable of addressing a broad spectrum of data curation tasks and introduces 28 clinically relevant tasks, as detailed [here](https://dragon.grand-challenge.org/tasks/).


## Accessing the reports in the DRAGON benchmark
All data, including clinical reports and associated labels, are stored in a sequestered manner on the Grand Challenge platform. This prevents users from directly accessing or viewing the data, preserving patient privacy by design. While participants cannot directly download or see the data, they do have full functional access for model training and validation through the platform interface. Keeping the test labels hidden helps to mitigate potential biases. To aid the development of solutions we provide synthetic datasets (see the [development guide](documentation/development_guide.md)) for all task types and provide an example case for each of the tasks in the DRAGON manuscript and <a href="https://github.com/DIAGNijmegen/dragon_sample_reports" target="_blank">here</a>.


## What can the DRAGON algorithm do for you?
If you are a domain scientist (radiologist, pathologist, ...) looking to automate your own data curation, the DRAGON algorithm provides an out-of-the-box solution that is all but guaranteed to provide excellent results on your individual dataset. Simply convert your dataset into the DRAGON format and enjoy the power of AI - no expertise required!

If you are an AI researcher developing NLP methods, DRAGON:

* offers a fantastic out-of-the-box applicable baseline algorithm to compete against
* can act as a method development framework to test your contribution on a large number of datasets without having to tune individual pipelines (for example evaluating a new loss function)
* provides a strong starting point for further dataset-specific optimizations. This is particularly used when competing in NLP challenges
* provides a new perspective on the design of NLP methods: maybe you can find better connections between dataset properties and best-fitting NLP pipelines?


## What is the scope of the DRAGON challenge?
The DRAGON benchmark focusses on clinical NLP with "closed questions" (see the eight task types at the top). This means that generative models are out of the scope for the DRAGON challenge.

DRAGON relies on supervised learning, which means that you need to provide training cases for your application. The number of required training cases varies heavily depending on the complexity of the problem. No one-fits-all number can be provided here!


## How does the DRAGON baseline work?
Given a new dataset, DRAGON will systematically analyze the provided training cases and create a 'dataset fingerprint'.

DRAGON configures its pipeline based on a two-step recipe:

* **Fixed parameters** are not adapted. During development of the DRAGON baseline we identified a robust configuration (that is, certain architecture and training properties) that can simply be used all the time. This includes, for example, the loss function and learning rate.
* **Rule-based parameters** use the dataset fingerprint to adapt certain segmentation pipeline properties by following hard-coded heuristic rules. For example, the regression target is transformed with a logarithmic function when the skew of the label distribution is more than one.

## How to get started to use existing DRAGON algorithms?
If you want to use an existing algorithm to annotate new data, you need:
1. Manually annotated training data (provided by you)
2. Manually annotated validation data (provided by you)
3. The data you want to annotate (we call this "test data" because in the context of the benchmark, the algorithm will provide predictions for the test data)

First, prepare the data in the correct dataset convention: please see the [dataset conversion](documentation/dataset_convention.md) for more information.

You can use the algorithm on Grand Challenge or locally. Either way, the algorithm will fit the model to your training data, have it select the model checkpoint based on your validation data, and then produce the model predictions for the "test data". To use the algorithm on Grand Challenge, navigate to the leaderboard and select the algorithm you want [PENDING].

If you prefer to perform these steps on your own hardware, please follow the steps in "How to get started as AI developer?" to learn how to set this up. You can find the GitHub repository of submissions on the leaderboard under the GitHub icon [PENDING].


## Bringing in your own data
To format your own dataset for usage with the algorithms from the DRAGON challenge, check out the [dataset convention](/documentation/dataset_convention.md).


## Where does the DRAGON baseline perform well and where does it not perform?
The DRAGON baseline methods have been evaluated on the <a href="https://dragon.grand-challenge.org/evaluation/test/leaderboard/" target="_blank">DRAGON test set</a>, as prespecified in the <a href="https://zenodo.org/records/10374512" target="_blank">DRAGON Statistical Analysis Plan</a>.

Model performance for the best model (<a href="https://dragon.grand-challenge.org/evaluation/ccb4ad0e-fdfa-49d0-af77-fcc61810c5f6/" target="_blank">RoBERTa large with domain-specific pretraining</a>) was excellent for 10/28 tasks (T1, T2, T4, T5, T10, T15, T16, T19, T20, T21), good for 8/28 tasks (T3, T7, T11, T22, T23, T25, T26, T27), moderate for 6/28 tasks (T9, T13, T17, T18, T24, T28) and poor for 4/28 tasks (T6, T8, T12, T14), based on pre-defined performance thresholds.

For the full performance breakdown of each model, see the [evaluation results](documentation/evaluation_results.md).

## Managed By
Diagnostic Image Analysis Group, Radboud University Medical Center, Nijmegen, The Netherlands

## Contact Information
Joeran Bosma: Joeran.Bosma@radboudumc.nl
