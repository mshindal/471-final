#!/usr/bin/env python3

import numpy as np
import math
import timeit
from random import uniform, seed
import mpp
from pdb import set_trace as d

def readInDataSet(fileName):
    content = []
    with open(fileName) as f:
        content = f.readlines()
    word = content[0].split()
    data = np.empty((len(content)-1,len(word)),dtype = 'object')    
    for i in range(1,len(content)):
        words = content[i].split(",")
        for x in range(0,len(words)):
            newData = words[x]
            newData = " ".join(newData.split())
            
            data[i-1][x]=newData
    return data
    
def writeDataSet(data, fileName):
    with open(fileName, 'w') as f:
        for i in range(0, len(data)):
            for x in range(0,len(data[0])):
                f.write(str(data[i][x]))
                f.write(" ")
            f.write("\n")
            
            
def removeIncompleteFeatures(data):
    data = np.delete(data,7,axis=1)
    return data
def removeIncompleteSamples(data):
    remove = []
    for x in range(len(data)):
        for i in range(len(data[0])):
            if (data[x][i]=="?"):
                remove.append(x)
    for x in range(len(remove)):
        data = np.delete(data,remove[len(remove)-x-1],axis=0)
    return data
    

    
def interpriteData(data):
    for i in range(0,len(data)):
        for x in range(len(data[i])):
            if (data[i][x] =="low"):
                data[i][x] = "5"
            elif (data[i][x] =="mid"):
                data[i][x] = "10"
            elif (data[i][x] =="high"):
                data[i][x] = "15"
            
            
            elif (data[i][x] =="good"):
                data[i][x] = "7"
            elif (data[i][x] =="excellent"):
                data[i][x] = "14"
                
                
            elif (data[i][x] =="unstable"):
                data[i][x] = "5"
            elif (data[i][x] =="mod-stable"):
                data[i][x] = "10"
            elif (data[i][x] =="stable"):
                data[i][x] = "15"
                
                
            elif (data[i][x] =="I"):
                data[i][x] = "0"
            elif (data[i][x] =="S"):
                data[i][x] = "1"
            elif (data[i][x] =="A"):
                data[i][x] = "2"
            elif (data[i][x] =="?"):
                data[i][x] = "-5"
                
def castData(data):
    newData = np.zeros((len(data),len(data[0])), dtype = float)
    for i in range(0,len(data)):
        for x in range(0,len(data[0])):
            newData[i][x] = float(data[i][x])
    return newData

def calculateAverage(data):
    averages = np.zeros((len(data[0])),dtype = 'float')
    for i in range(0,len(data)):
        for x in range(0,len(data[i])):
            averages[x]+=data[i][x]
    averages /= len(data)
    return averages
def calculateStandardDeviation(data, average):
    std = np.zeros((len(data[0])),dtype='float')
    for i in range(0,len(data)):
        for x in range(0,len(data[i])):
            std[x]+= (data[i][x]-average[x])**2
    std = (std/len(data))**(1/2)
    return std
def normalizeSets(trainingSet, testingSet):
    avg = calculateAverage(trainingSet)
    std = calculateStandardDeviation(trainingSet, avg)
    for i in range(0,len(trainingSet)):
        for x in range(0,len(trainingSet[0])-1):
            trainingSet[i][x] = (trainingSet[i][x]-avg[x])/std[x]
    for i in range(0,len(testingSet)):
        for x in range(0,len(testingSet[0])-1):
            testingSet[i][x] = (testingSet[i][x]-avg[x])/std[x]

def knnEvaluationAtX(xs, trainingData, k,n, numClasses):
    distances = np.zeros((len(trainingData),2),dtype='float')
    for i in range(0,len(trainingData)):
        sum=0
        for x in range(0,len(trainingData[0])-1):
            sum += (xs[x]-trainingData[i][x])**n
       
        
        sum = sum**(1/n)
        
        
        distances[i][0] = sum    
        distances[i][1] = i
    
    classSums = np.zeros((numClasses),dtype = 'float')
    ind = np.argsort(distances[:,0])
    distances = distances[ind]
    for i in range(0,k):
        classSums[trainingData[int(distances[i][1])][len(trainingData[0])-1]]+=1
    predClass = 0
    highestCount =0
    for i in range(0,numClasses):
        if classSums[i] > highestCount:
            predClass = i
            highestCount = classSums[i]
    return predClass
    
def knnEvaluationOnSet(ts, trainingData, k,n, numClasses):
    accuracy = 0
    confusionMatrix = np.zeros((numClasses, numClasses), dtype = float)
    for i in range(0,len(ts)):
        set = knnEvaluationAtX(ts[i], trainingData,k,n, numClasses)
        
        confusionMatrix[ts[i][len(ts[i])-1]][set]+=1
    return confusionMatrix
    
def seperateData(data, numSets):
    numElements = int(len(data)/numSets)
    if (len(data)/numSets)!=numElements:
        print("warning, data cannot be evenly divided, some samples will be discarded")
    
    newData = np.zeros((numSets,numElements,len(data[0])), dtype = float)
    
    np.random.shuffle(data)
    for x in range(numSets):
        for i in range(numElements):
            newData[x][i] = data[x*numElements+i]
    return newData
    

data = readInDataSet("post_opt.data")
# print(data)
#data = removeIncompleteSamples(data)
#data = removeIncompleteFeatures(data)
interpriteData(data)
data = castData(data)
d()
mpp.classify(data)
data = seperateData(data,4)

#
'''
data = np.delete(data, [7], axis=1)
interpriteData(data)
newData = castData(data)
accuracy = knnEvaluationOnSet(newData, newData, 5, 2,3)
print(accuracy)
writeDataSet(data, "interprited.dat")
'''


