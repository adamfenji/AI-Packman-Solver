# naiveBayes.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
    See the project description for the specifications of the Naive Bayes classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1 # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **

    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Outside shell to call your method. Do not modify this method.
        """

        # might be useful in your code later...
        # this is a list of all features in the training set.
        self.features = list(set([ f for datum in trainingData for f in datum.keys() ]))

        if (self.automaticTuning):
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        
        #initialize counts and conditional prob tables that I will need
        labelCounts = util.Counter()
        featureCounts = {label: util.Counter() for label in self.legalLabels}
        conditionalProb = {label: {feature: util.Counter() for feature in self.features} for label in self.legalLabels}

        # Count label occurrences and feature occurrences given each label
        for i in range(len(trainingData)):
            labelCounts[trainingLabels[i]] += 1
            for f, v in trainingData[i].items():
                featureCounts[trainingLabels[i]][f] += v
                conditionalProb[trainingLabels[i]][f][v] += 1

        bestK = None
        bestAccuracy = -1

        for k in kgrid:

            self.k = k
            self.conditionalProb = {label: {feature: util.Counter() for feature in self.features} for label in self.legalLabels}
            self.priorProb = util.Counter()
            
            for l in self.legalLabels: #get prior probabilities
                self.priorProb[l] = labelCounts[l] / sum(labelCounts.values())

            for l in self.legalLabels: #get conditional prob with smoothing
                for f in self.features:
                    for v in [0, 1]:
                        self.conditionalProb[l][f][v] = (conditionalProb[l][f][v] + k) / (sum(conditionalProb[l][f].values()) + 2 * k)

            #formula for checking the accuracy on validation sets
            guesses = self.classify(validationData)
            accuracy = sum([guesses[i] == validationLabels[i] for i in range(len(validationLabels))]) / len(validationLabels)
            #after getting the accuracy value, I need to check if we need to update the best k value
            if accuracy > bestAccuracy or (accuracy == bestAccuracy and (bestK is None or k < bestK)):
                bestK = k
                bestAccuracy = accuracy

        #here, I set the best k value and re computed the probabilities
        self.k = bestK
        self.conditionalProb = {label: {feature: util.Counter() for feature in self.features} for label in self.legalLabels}
        self.priorProb = util.Counter()

        for l in self.legalLabels: #get prior probabilities 
            self.priorProb[l] = labelCounts[l] / sum(labelCounts.values())

        for l in self.legalLabels: #get conditional prob with smoothing
            for f in self.features:
                for v in [0, 1]:
                    self.conditionalProb[l][f][v] = (conditionalProb[l][f][v] + self.k) / (sum(conditionalProb[l][f].values()) + 2 * self.k)

        


    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.

        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):

        logJoint = util.Counter() #initialize a counter

        #for every possible label, do:
        for l in self.legalLabels:
            
            logP = math.log(self.priorProb[l]) #log of the prior probability of the label

            #for every feature in the feature set
            for f in self.features:
                if f in datum: logP += math.log(self.conditionalProb[l][f][datum[f]])
                else: logP += math.log(self.conditionalProb[l][f][0])
            logJoint[l] = logP
        return logJoint

