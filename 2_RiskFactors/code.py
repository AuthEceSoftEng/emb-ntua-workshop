import numpy as np
import pandas as pd
from sklearn import svm, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def read_data():
    # Read data
    data = pd.read_csv("2_RiskFactors/data/riskFactorData.csv", sep=";")
    return data

def get_statistics(data):
    # Get data statistics
    print("\n")
    print(data[["Year", "BMI", "SBP", "FAT"]].describe())
    print("\n")
    print(data[["Country", "Sex"]].describe())
    print("\n")
    print(data[["BMI", "SBP", "FAT"]].corr())

def knnModel(data):
    # Define year
    year = 2008
    
    # Filter data
    data = data[(data["Year"] == year)]

    # Keep selected columns
    x_train = data[["BMI", "SBP", "FAT"]]
    y_train = data["Sex"].to_list()

    # Define test data
    test_BMI = 23.5
    test_SBP = 126.5
    test_FAT = 4.4
    test_data = [test_BMI, test_SBP, test_FAT]

    for k in range(2, 10):
        # Create kNN classifier
        model = KNeighborsClassifier(n_neighbors=k)
        # Train kNN classifier
        model.fit(x_train, y_train)
        # Predict new sample
        prediction = model.predict([test_data])[0]
        prediction_prob = model.predict_proba([test_data])[0]

        print("\n")
        print(str(k) + "-Nearest Neighbors")
        print("Female: " + str(prediction_prob[0]))
        print("Male: " + str(prediction_prob[1]))
        print("Model Prediction: " + prediction)

def svmModel(data):
    # Keep selected columns
    x_train = data[["BMI", "SBP", "FAT"]]
    y_train = data["Country"].to_list()

    # Define test data
    test_BMI = 23.5
    test_SBP = 126.5
    test_FAT = 4.4
    test_data = [test_BMI, test_SBP, test_FAT]

    # Create SVM classifier
    model = svm.SVC()
    # Train SVM classifier
    model.fit(x_train, y_train)
    # Predict new sample
    prediction = model.predict([test_data])[0]
    print("Model Prediction: " + prediction)

def gaussianModel(data):
    # Keep selected columns
    x_train = data[["BMI", "SBP", "FAT"]]
    y_train = data["AtRisk"].to_list()

    # Define test data
    test_BMI = 26.5
    test_SBP = 136.5
    test_FAT = 5.8
    test_data = [test_BMI, test_SBP, test_FAT]

    # Create Gaussian classifier
    model = GaussianNB()
    # Train Gaussian classifier
    model.fit(x_train, y_train)
    # Predict new sample
    prediction = model.predict([test_data])[0]
    print("Model Prediction: " + prediction)

def decisionTreesModel(data):
    # Keep selected columns
    x_train = data[["BMI", "SBP", "FAT"]]
    y_train = data["AtRisk"].to_list()

    # Define test data
    test_BMI = 25.5
    test_SBP = 136.5
    test_FAT = 4.8
    test_data = [test_BMI, test_SBP, test_FAT]

    # Create Decision Tree classifier
    model = tree.DecisionTreeClassifier()
    # Train Decision Tree classifier
    model.fit(x_train, y_train)
    # Predict new sample
    prediction = model.predict([test_data])[0]
    print("Model Prediction: " + prediction)

# Read data
print("Reading data...")
data = read_data()
print("Reading ended...")

# Get general statistics
get_statistics(data)

### CLASSIFICATION ###

# Get sex by kNN
knnModel(data)

# Get country by SVM
svmModel(data)

# Get risk by NaiveBayes
gaussianModel(data)

# Get risk by DecisionTrees
decisionTreesModel(data)

### CLUSTERING ###

# KMeans Clustering
# TODO

# Hiearchical Clustering
# TODO

# DBSCAN Clustering
# TODO