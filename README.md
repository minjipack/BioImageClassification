# Active Learning (Machine Learning) Approach for Image Classification

It was a project in `Automation of Research/Machine Learning Robotics(02-750)` course.

## Purpose
To classify bio-image, i.e., protein images based on the subcellular localization patterns,
Fluorescent microscopy can reveal the subcellular localization patterns of tagged proteins. 

## Dataset
The data was encoded as vectors representing subcellular localization patterns and labeled with protein names. There were 8 types of protein so it was basically multi-classification problem.

## Approach
I applied Uncertainty Sampling as my query strategy for active learning. For the dataset without noise, my SVM classifier based on active learning approach performed around 95%. But when the dataset including noise, the performance dropped significantly. In order to deal with the problem, I used recursive feature selection, which helps remove noisy feature and only include the features that are most helpful. After applying this, it reached around 90 %.

