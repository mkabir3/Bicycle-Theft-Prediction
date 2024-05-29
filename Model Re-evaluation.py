# -*- coding: utf-8 -*-
"""
@author: harki
"""

###################Loading dataset##################
import pandas as pd
import numpy as np
project_dataset =pd.read_csv('C:/Users/harki/Documents/College/Sem-5/COMP-309/Project/Dataset/Bicycle_Thefts.csv')
################Loading dataset End#################


################Preparing dataset#################
##Removing unknown from status
project_dataset = project_dataset[project_dataset.Status != 'UNKNOWN']
project_dataset = project_dataset[project_dataset.Location_Type != 'Unknown']
##removing 0.0 from cost of bike
project_dataset = project_dataset[project_dataset.Cost_of_Bike != 0.0]
project_dataset.describe()
project_dataset.info()

#### Handling missing values ####
project_dataset['Cost_of_Bike'].fillna(project_dataset['Cost_of_Bike'].mean(),inplace=True)
project_dataset['Cost_of_Bike'].head(30)    # Check top 30 values
print(project_dataset['Cost_of_Bike'])      # Print all values
# Droping the columns
project_dataset = project_dataset.drop(columns="Bike_Colour")
project_dataset = project_dataset.drop(columns="Bike_Model")
project_dataset.isnull()
project_dataset.info()
project_dataset['Status'].unique()
project_dataset.columns.values
project_dataset['Location_Type'].unique()
project_dataset['Location_Type']=np.where(project_dataset['Location_Type'] =='Ttc Street Car', 'Ttc Subway Station', project_dataset['Location_Type'])
project_dataset['Location_Type']=np.where(project_dataset['Location_Type'] =='Ttc Light Rail Transit Station', 'Ttc Subway Station', project_dataset['Location_Type'])
project_dataset['Location_Type']=np.where(project_dataset['Location_Type'] =='Ttc Admin Or Support Facility', 'Ttc Subway Station', project_dataset['Location_Type'])
project_dataset['Location_Type']=np.where(project_dataset['Location_Type'] =='Ttc Subway Train', 'Ttc Subway Station', project_dataset['Location_Type'])
project_dataset = project_dataset[project_dataset.Location_Type != 'Unknown']

# We don't need recovered status objects => Calculating only for the stolen ones
# 1-Conversion of object to bool values (integer)
project_dataset['Status']=(project_dataset['Status']=='RECOVERED').astype(int)
#Check the values of how many bicycles were recovered.
print(project_dataset['Status'].value_counts())
#Check the average of all the numeric columns
pd.set_option('display.max_columns',100)
print(project_dataset.groupby('Status').mean())
#Histogram => Status_Of_Bike by Bike_Type
import matplotlib.pyplot as plt
pd.crosstab(project_dataset.Bike_Type,project_dataset.Status)
pd.crosstab(project_dataset.Bike_Type,project_dataset.Status).plot(kind='bar')
plt.title('Status by Bike Type')
plt.xlabel('Bike_Type')
plt.ylabel('Status')
project_dataset =project_dataset[['Location_Type','Premise_Type','Bike_Type','Cost_of_Bike','Neighbourhood','Status']]
categoricals = []
for col, col_type in project_dataset.dtypes.iteritems():
     if col_type == 'O':
          categoricals.append(col)

# Printing list of categories used
print(categoricals)

project_dataset_ohe = pd.get_dummies(project_dataset, columns=categoricals, dummy_na=False)
print(project_dataset_ohe.head())
print(project_dataset_ohe.columns.values)
print(len(project_dataset_ohe) - project_dataset_ohe.count())
from sklearn import preprocessing
# Get column names first
names = project_dataset_ohe.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(project_dataset_ohe)
scaled_df = pd.DataFrame(scaled_df, columns=names)
print(scaled_df.head())
print(scaled_df['Cost_of_Bike'].describe())
print(scaled_df['Status'].describe())
print(scaled_df.dtypes)
#######################Preparing dataset End########################




################Logistic Regression model training#################
from sklearn.linear_model import LogisticRegression
dependent_variable = 'Status'
# Another way to split the features
x = scaled_df[scaled_df.columns.difference([dependent_variable])]
x.dtypes
y = scaled_df[dependent_variable]
#convert the class back into integer
y = y.astype(int)
# Split the data into train test - 20%
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(x,y, test_size = 0.2)
#build the model
lr = LogisticRegression(solver='lbfgs')
lr.fit(x, y)
# Score the model using 10 fold cross validation
from sklearn.model_selection import KFold
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
logistic_score = np.mean(cross_val_score(lr, trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=10))
print ('The logistic_score of the 10 fold run is: ',logistic_score)

# Testing the model using the training and testing data
# The logistic_score of the 10 fold run is:  0.9879466432991441
testY_predict = lr.predict(testX)
testY_predict.dtype
#print(testY_predict)

# TODO: pickle the logictic model


# Run separately
from sklearn import metrics
labels = y.unique()
print(labels)
logistic_accuracy = metrics.accuracy_score(testY, testY_predict)
print("Accuracy:",logistic_accuracy)
#Let us print the confusion matrix
from sklearn.metrics import confusion_matrix
logistic_confusion_matrix = confusion_matrix(testY, testY_predict, labels)
print("Confusion matrix \n" , logistic_confusion_matrix)
##############Logistic Regression model training End##################




##################Decision tree model training########################
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
for i in range(25,35):
    dt = DecisionTreeClassifier(criterion='entropy',max_depth=i, min_samples_split=20, random_state=123)
    dt.fit(trainX,trainY)
    crossvalidation = KFold(n_splits=10, shuffle=True, random_state=25)
    decision_tree_score = np.mean(cross_val_score(dt, trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
    print(decision_tree_score)
##when i=31 the decision_tree_score is highest
dt = DecisionTreeClassifier(criterion='entropy',max_depth=31, min_samples_split=20, random_state=123)
dt.fit(trainX,trainY)
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=25)
decision_tree_score = np.mean(cross_val_score(dt, trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
print(decision_tree_score) 
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
importance = dt.feature_importances_
# summarize feature importance
# for i,v in enumerate(importance):
# 	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

### Test the model using the testing data
testY_predict = dt.predict(testX)
testY_predict.dtype
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics 
# labels = y.unique()
# print(labels)
dt_accuracy = metrics.accuracy_score(testY, testY_predict)
print("Accuracy:",dt_accuracy)
#Let us print the confusion matrix
from sklearn.metrics import confusion_matrix
dt_matrix = confusion_matrix(testY, testY_predict, labels)
print("Confusion matrix \n" , dt_matrix)
#####################Decision tree model training############################




###############################Flask API#####################################
from flask import Flask, request, jsonify
import traceback
import pandas as pd
import joblib
import sys
# Your API definition
app = Flask(__name__)
@app.route("/predict", methods=['GET','POST']) #use decorator pattern for the route
def predict():
    if lr:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)
            print(query)
            from sklearn import preprocessing
            scaler = preprocessing.StandardScaler()
            # Fit your data on the scaler object
            scaled_df = scaler.fit_transform(query)
            # return to data frame
            query = pd.DataFrame(scaled_df, columns=model_columns)
            print(query)
            prediction = list(lr.predict(query))
            print({'prediction': str(prediction)})
            return jsonify({'prediction': str(prediction)})
            return "Welcome to titanic model APIs!"

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    lr = joblib.load('E:\DWPython\Project2/model_columns.pkl') # Load "model.pkl"  # TODO: replace with the coumns pcikle
    print ('Model loaded')
    model_columns = joblib.load('E:\DWPython\Project2/model_lr2.pkl') # Load "model_columns.pkl"  # TODO: replace with the logistic pcikle
    print ('Model columns loaded')
    app.run(port=port, debug=True)
#############################Flask API End###################################
