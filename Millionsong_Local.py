
# coding: utf-8

# In[1]:

from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from test_helper import Test
import numpy as np


# In[2]:

# load data
rawData = sc.textFile('millionsong.txt')


# In[54]:

samplePoints = rawData.take(5)
print samplePoints


# In[57]:

def parsePoint(line):

    tokens = line.split(',')
    label = tokens[0]
    features = tokens[1:]
    return LabeledPoint(label, features)

parsedData = rawData.map(parsePoint)

print parsedData.take(3)


# In[58]:

#Devide rawData into Traning, Validation and Test
weights = [.8, .1, .1]
seed = 50
parsedTrainData, parsedValData, parsedTestData = parsedData.randomSplit(weights, seed)


# In[64]:

# Fit the model with default values
fitModel = LinearRegressionWithSGD.train(parsedTrainData)
print  fitModel


# In[65]:

# Prediction 
testPoint = parsedTrainData.take(1)[0]

print testPoint.label

testPrediction = fitModel.predict(testPoint.features)

print samplePrediction


# In[16]:

# Define a formula to caluculate RMSE(Root Mean Square Error)
def squaredError(label, prediction):
    ##Calculates the the squared error for a single prediction.
    return (label - prediction)**2

def calcRMSE(labelsAndPreds):
    ##Calculates the root mean squared error for an `RDD` of (label, prediction) tuples.
    return np.sqrt(labelsAndPreds
                   .map(lambda (label, prediction): squaredError(label, prediction))
                   .mean())


# In[40]:

#Create new RDD with actual label and predicted label 
labelsAndPreds = parsedValData.map(lambda lp: (lp.label, fitModel.predict(lp.features)))

#Calculation RMSE
rmseValLR1 = calcRMSE(labelsAndPreds)
print rmseValLR1


# In[67]:

##Grid search
# Values to use when training the linear regression model
miniBatchFrac = 1.0  # miniBatchFraction
regType = 'l2'  # regType
useIntercept = True  # intercept

# Seed of minmum RMSE
min_rmseVal = 10**10

modelRMSEs = []

for grid_alpha in [1e-5,1.0,10,100]:
    for grid_numIters in [50,100,500]:
        for grid_reg in [1e-5,0.0, 1.0]:
            model = LinearRegressionWithSGD.train(parsedTrainData,
                                                  iterations=numIters, 
                                                  step=grid_alpha, 
                                                  miniBatchFraction=miniBatchFrac, 
                                                  initialWeights=None, 
                                                  regParam=reg, 
                                                  regType=regType, 
                                                  intercept=useIntercept)
            
            labelsAndPreds = parsedValData.map(lambda lp: (lp.label, model.predict(lp.features)))
            rmseVal = calcRMSE(labelsAndPreds)
            print "best_alpha:{},best_numIter:{}, best_reg:{}, best_rmseVal:{}".format(grid_alpha, grid_numIters, grid_reg, rmseVal)


# In[1]:

#Fitting with figures fixed by grid search 
Final_model = LinearRegressionWithSGD.train(parsedTrainData,
                                           iterations=10, 
                                           step=500, 
                                           miniBatchFraction=miniBatchFrac, 
                                           initialWeights=None, 
                                           regParam = 1e-05, 
                                           regType=regType, 
                                           intercept=useIntercept)


# In[70]:

#Labels comparison between Final_model and actual data. 
Final_testPoint = parsedTrainData.take(1)[0]
print Final_testPoint.label
print Final_testPoint.features

Final_Prediction = Final_model.predict(Final_testPoint.features)
print Final_Prediction

