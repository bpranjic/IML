# Introduction to Machine Learning ETH 2022
## Task 1
Given a vector x, the goal was to predict a value y as a linear function of a set of feature transformations. Generating the Pseudoinverse for the matrix product to directly get the weights was enough to pass the baseline.

## Task 2 
This task contained 3 subtasks and it was primarily about preprocessing data to handle missing values.
### Subtask 1: (Binary Classifier) Predict whether a certain test will be required based on medical values
### Subtask 2: (Binary Classifier) Predict whether a patient is going to have a sepsis event during his stay at the hospital
### Subtask 3: (Regression Task) Predict the mean value of a vital sign in the remaining stay at the hospital to predict a more general evolution of the patient state

## Task 3
In this task we have an input of triplets of images of food, and the goal is to predict whether image 2 or 3 is closer in taste to image 1. This task is done using Transfer Learning, using the pretrained image classification network InceptionResNetV2 from Keras. To increase accuracy, we double our training data set, by swapping images 2 and 3 and inverting the corresponding true value.

## Task 4
The goal of this task is to find suitable materials for organic semiconductors. Important values are the so called HOMO (highest occupied molecular orbital) and LUMO (lowest unoccupied molecular orbital) and especially the HOMO-LUMO gap, which determines how well a material conducts electricity. We are given only a small set of molecules in the SMILE format and their corresponding HOMO-LUMO gap, and a large set of molecules with their LUMO level. We use the large set to pretrain a model to predict the LUMO level and then use the smaller train set to predict the HOMO level.
