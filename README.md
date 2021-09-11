# Neural_Network_Charity_Analysis
## Overview
The purpose of this analysis is to use Machine Learning and Neural Networks to create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup - a charitable organization.  

The analysis was done using Jupyter Notebook's Sci-Kit Learn and TensorFlow libraries, and performed on the data set *charity_data.csv*.  

## Results

### Data Preprocessing
Before training and testing our model the data was transformed in order properly run the model.

***What variable(s) are considered the target(s) for your model?***

The target variable we are intedending to predict is the *IS_SUCCESSFUL* column.

***What variable(s) are considered to be the features for your model?***

The following columns were used as features for our model:

- APPLICATION_TYPE—*Alphabet Soup application type*
- AFFILIATION—*Affiliated sector of industry*
- CLASSIFICATION—*Government organization classification*
- USE_CASE—*Use case for funding*
- ORGANIZATION—*Organization type*
- STATUS—*Active status*
- INCOME_AMT—*Income classification*
- SPECIAL_CONSIDERATIONS—*Special consideration for application*
- ASK_AMT—*Funding amount requested*

***What variable(s) are neither targets nor features, and should be removed from the input data?***

In order to reduce the number of features for our model, we dropped the following columns:

- EIN
- NAME

Both columns are identification columns are are unecessary for the analysis.

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

2 hidden layers were used, and 2 activation functions were used: ReLu in the hidden layers and Sigmoid in the output layer.

The neural network ran for 50 epochs.

***Were you able to achieve the target model performance?***

The target accuracy for the model was 75%. Unfortunatley the target model performance was not achieved by our model. The initial testing of the model had an accuracy of 0.73.

<p align="center">
<img src=https://github.com/smanowar/Neural_Network_Charity_Analysis/blob/main/images/accuracy_original.PNG> 
</p>

To try and achieve target accuracy we attempted the following steps:
<br><br>
**1. Drop additional features**

To try and increase accuracy, in addition to dropping the *EIN* and *NAME* columns we also dropped the *USE_CASE* column, and increased the neurons to 100 neurons in the first hidden layer and 50 neurons in the second hidden layer.

<p align="center">
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
The overall results of our Neural Network model unfortunatley did not reach target accuracy. The initial run of our model achieved an accuracy of 73%, and after our three attempts to increase accuracy to 75% the accuracy of the model decresed to 47%. From this we can see that a better way to improve accuracy might include increasing our dataset, increasing the epochs, or using a different machine learning model. 

If we were to try the analysis again using a different model, a model that could solve the classification problem similarly to out Neural Network model is the Random Forest Classifier. The Random Forest algorithm will serve the same purpose as the Neural Network model - predicting whether or not a charitable donation will be successful or not - while also potentially increasing accuracy as the Random Forest algorithm reduces overfitting.
