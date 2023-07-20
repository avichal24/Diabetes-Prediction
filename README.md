
# Diabetes Prediction

This project serves a great role in serving the women to forecast wherther they are prone to diabetes or not, where the data entered by them is making them prone towards this harmful disease.
## About the dataset

The dataset is gathered from Kaggle which is one the most proper site for datasets, wherein what our dataset consistes of is,
1. Pregnancies Undergone
2. Glucose level
3. Blood Pressure level
4. Insulin
5. BMI etc...

All these parameters are there for us to toggle and train or model for prediction.

## About the project

This project basically uses the support vector machine model (SVM) to predict the outcome based on the training and test dataset,

All major part of the project comprises of the data classification between dependent variable and independent variable, because the received dataset was a bit pre sorted and not filtrations were required. 


Additionally what I have done is fistly, I have calculated both the outcomes where the person is prone to diabetes or in safe zone, after that what I have done is, I have made the project to grasp the inputs from the end user and based on these inputs the model provide us with the final outcome.
## Deployment

To deploy this project run

```bash
  # Diabetes Prediction

### Importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### Importing the Dependencies


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm

df = pd.read_csv('/content/diabetes.csv')
df.shape

df.head()

#### Dependent and Independent Variables

x = df.drop(columns='Outcome', axis=1)
y = df['Outcome']

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=30, random_state=0)

print(x.shape, x_train.shape, x_test.shape)

classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)

pred_x = classifier.predict(x_train)

### Accuracy Scores of Training and Test set

print("Acccuracy Score Train Data", accuracy_score(pred_x, y_train))

print("Accuracy score Test Data", accuracy_score(classifier.predict(x_test), y_test))

df.groupby('Outcome').mean()

##Predicting the Values based on input given

input_data_raw = (6.2,150,80,28,120,40,0.65,45)
input_data_con = np.asarray(input_data_raw).reshape(1,-1)

predict = print(classifier.predict(input_data_con))
if predict == 1:
  print('You have higher chance of being diabetic, be aware and change your habits')
else:
  print("You are safe from diabeties, but precaution is always better than cure")

input_data_raw =(3,102,62,15,50,25,0.2,25)
input_data_con = np.asarray(input_data_raw).reshape(1,-1)
input_data = scaler.transform(input_data_con)

predict = print(classifier.predict(input_data))
if predict == 1:
  print('You have higher chance of being diabetic, be aware and change your habits')
else:
  print("You are safe from diabeties, but precaution is always better than cure")

df.head(2)

## Prediction based on the data entered by end user

d1  = input("How many pregancies have you undergone: ")
d2 = input("What is your blood glucose level: ")
d3 = input("What is your current BLood Pressure: ")
d4 = input("Skin thickness: ")
d5 = input("What is your Insulin level as per records: ")
d6 = input("What is your current BMI: ")
d7 = input("Give your Diabetes Pedigree Function: ")
d8 = input("What is your current age: ")

input_data_collected = [d1, d2, d3, d4, d5, d6, d7, d8]
input_dataset_raw = np.asarray(input_data_collected)
input_dataset_con = input_dataset_raw.reshape(1,-1)
input_dataset_final = scaler.transform(input_dataset_con)

prediction = classifier.predict(input_dataset_final)

if prediction == 1:
  print("You are prone to Diabetic kindly go through a proper diagnosis!")

else:
  print("As per the detailes provided now you are safe, but precaution is better than cure.")
```


## Conclusion

Therefore as a result what we have seen that our model has predicted great outcomes which can be very much fruitful for the medical use or personal use and help your society lead a healthy life style.
# Hi, I'm Avichal Srivastava ! 👋

You can reach out to me at: srivastavaavichal007@gmail.com

LinkedIn: www.linkedin.com/in/avichal-srivastava-186865187
## 🚀 About Me
I'm a Mechanical Engineer by education, and love to work with data, I eventually started my coding journey for one of my Drone project, wherein I realized that it is something which makes me feel happy in doing, then I planed ahead to move in the Buiness Analyst or Data Analyst domain. The reason for choosing these domains is because I love maths a lot and all the Machine Learning algorithms are completely based on mathematical intution, So this was about me.

Hope! You liked it, and its just a beginning, many more to come, till then Happy Analysing!!

