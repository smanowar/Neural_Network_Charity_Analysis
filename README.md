# Neural_Network_Charity_Analysis
## Overview
The purpose of this analysis is to use Machine Learning and Neural Networks to create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup - a charitable organization.  

The analysis was done using Jupyter Notebook's sklearn and tensorflow libraries, and performed on the data set *charity_data.csv*.  

## Results

### Data Preprocessing
Before training and testing our model, data was processed in order properly run the model.

***What variable(s) are considered the target(s) for your model?***

The target variable we are intedending to predict is the *IS_SUCCESSFUL* column.

***What variable(s) are considered to be the features for your model?***

The following columns were used as features for our model:

- AFFILIATION
- CLASSIFICATION
- USE_CASE
- ORGANIZATION
- STATUS
- INCOME_AMT
- SPECIAL_CONSIDERATIONS
- ASK_AMT

***What variable(s) are neither targets nor features, and should be removed from the input data?***

In order to reduce the number of features for our model, we dropped the following columns:
- EIN
- NAME

### Compiling, Training, and Evaluating the Model

***How many neurons, layers, and activation functions did you select for your neural network model, and why?***

Below shows a summary of our Neural Network model:

<p align="center">
<img src=https://github.com/smanowar/Neural_Network_Charity_Analysis/blob/main/images/nn_original.PNG> 
</p>

From this we can see that: 

- 80 neurons were used in the first hidden layer
- 30 neurons were used in the second hidden layer
- and 1 neuron was used in the output layer

2 hidden layers were used, and 2 activation functions were used: relu and sigmoid.

The neural network ran for 50 epochs.

***Were you able to achieve the target model performance?***

The target accuracy for the model was 75%. Unfortunatley the target model performance was not achieved by our model. The initial testing of the model had an accuracy of 0.73.

<p align="center">
<img src=https://github.com/smanowar/Neural_Network_Charity_Analysis/blob/main/images/accuracy_original.PNG> 
</p>
<br><br>
To try and achieve target accuracy we attempted the following steps:

**1. Drop additional features**

To try and increase accuracy, in addition to dropping the *EIN* and *NAME* columns we also dropped the *USE_CASE* column, and increased the neurons to 100 neurons in the first hidden layer and 50 neurons in the second hidden layer.

<p align="left">
<img src=https://github.com/smanowar/Neural_Network_Charity_Analysis/blob/main/images/nn_AddNeurons.PNG> 
</p>

This increased the accuracy of our model to:

<p align="left">
<img src=https://github.com/smanowar/Neural_Network_Charity_Analysis/blob/main/images/accuracy_drop_features.PNG> 
</p>

**2. Adding additional hidden layers**
In our next attempt to try and increase accuracy additional hidden layers were added.

<p align="center">
<img src=https://github.com/smanowar/Neural_Network_Charity_Analysis/blob/main/images/nn_AddHiddenLayers.PNG> 
</p>

This decreased the accuracy of our model to:
<p align="left">
<img src=https://github.com/smanowar/Neural_Network_Charity_Analysis/blob/main/images/accuracy_hidden_layers.PNG> 
</p>

**3. Changing the activation function of hidden layers or output layers**

Lastly, since both Sigmoid function and Tanh function serve similar uses in binary classification, we attempted our model using the tanh function in the output layer.

<p align="center">
<img src=https://github.com/smanowar/Neural_Network_Charity_Analysis/blob/main/images/nn_tanh.PNG> 
</p>

This decreased the accuracy of our model to:

<p align="left">
<img src=https://github.com/smanowar/Neural_Network_Charity_Analysis/blob/main/images/accuracy_tanh.PNG> 
</p>

## Summary
The overall results of our Neural Network model unfortunatley did not reach target accuracy. If we were to try the analysis again using a different model, a model that could solve the classification problem similarly to the Neural Network problem is the Logistic Regression model.

 -"This sigmoid curve is the exact same curve used in the sigmoid activation function of a neural network. In fact, a basic neural network using the sigmoid activation function is effectively a logistic regression model"
-"Datasets with thousands of data points, or datasets with complex features, may overwhelm the logistic regression model, while a deep learning model can evaluate every interaction within and across neurons."
