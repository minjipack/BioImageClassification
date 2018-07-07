
# Active Learning (Machine Learning) Approach for Image Classification
It was a course project, [02-450/02-750 Automation of Biological Research: Robotics and Machine Learning](https://sites.google.com/site/automationofbiologicalresearch/image-classification-project).
Implemented **Active(=Query) Learning** approach and Random Learning approach to classify 8 proteins. We can observe two different approaches' performance.


## Aim
To classify bio-image, i.e., protein images based on the subcellular localization patterns, Fluorescent microscopy can reveal the subcellular localization patterns of tagged proteins.


### What is Active Learning?
Active Learning is one kind of Machine Learning Paradigm that requires fewer training dataset to perform as good as supervised machine learning approach with large amounts of data. Sometimes, it even outperforms. Also, it can QUERY the label of data, which will be included as the training dataset that we want to learn from. Because of this property, it is also called as QUERY learning.

I applied Uncertainty Sampling as my query strategy for active learning. For the dataset without noise, my SVM classifier based on active learning approach performed around 95%. But, for the dataset including noise, the performance dropped significantly. In order to deal with the problem, I used recursive feature selection, which helps remove noisy features and only include the features that are most helpful. After applying this, it reached around 90 %.


## Installation

Install `matplotlib`, `NumPy`, and `scikit-learn`.

    $ git clone https://github.com/mpack2018/BioImageClassification.git



## Usage

    $ python rfecexperiment.py

## Tunable parameters
: variables to tune inside `rfecexperiment.py`

- MODE 					: the type of dataset (easy/moderate/difficult) under 
- CLF 					: the classifier.

- INITIAL_SAMPLES_SIZE	: based on your budget, you can tune the size of initial training set
- UNCERTAIN_SAMPLES_ITER	: Based on your budget, you can tune the size of training set to query their true label to oracle.

## Results

SVM on Easy Dataset:
![easy dataset](https://imgur.com/a/AWXetV7)

SVM on Moderate Dataset:
![moderate dataset](https://imgur.com/a/AWXetV7)

SVM on Difficult Dataset:
![difficult dataset](https://imgur.com/a/XtfyuG5)

## Future Work
If people like your project they?ll want to learn how they can use it. To do so include step by step guide to use your project.
With some changes, i.e., data loading, in `run_experiment()` function under `rfecvexperiment.py` file, I can expand this project to other projects which want to apply the same active learning approach. I might update this so that it could be easily used in other projects.
