# classifierAgents.py
# parsons/07-oct-2017
#
# Version 1.0
#
# Some simple agents to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# The agents here are extensions written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import game
import util
import sys
import os
import csv
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

from math import log

# ClassifierAgent
#
# An agent that runs a classifier to decide what to do.
class ClassifierAgent(Agent):

    # Constructor. This gets run when the agent starts up.
    def __init__(self):
        print "Initialising"

    # Take a string of digits and convert to an array of
    # numbers. Exploits the fact that we know the digits are in the
    # range 0-4.
    #
    # There are undoubtedly more elegant and general ways to do this,
    # exploiting ASCII codes.
    def convertToArray(self, numberString):
        numberArray = []
        for i in range(len(numberString) - 1):
            if numberString[i] == '0':
                numberArray.append(0)
            elif numberString[i] == '1':
                numberArray.append(1)
            elif numberString[i] == '2':
                numberArray.append(2)
            elif numberString[i] == '3':
                numberArray.append(3)
            elif numberString[i] == '4':
                numberArray.append(4)

        return numberArray
                
    # This gets run on startup. Has access to state information.
    #
    # Here we use it to load the training data.
    def registerInitialState(self, state):

        # open datafile, extract content into an array, and close.
        self.datafile = open('good-moves.txt', 'r')
        content = self.datafile.readlines()
        self.datafile.close()

        # Now extract data, which is in the form of strings, into an
        # array of numbers, and separate into matched data and target
        # variables.
        self.data = []
        self.target = []
        # Turn content into nested lists
        for i in range(len(content)):
            lineAsArray = self.convertToArray(content[i])
            dataline = []
            for j in range(len(lineAsArray) - 1):
                dataline.append(lineAsArray[j])

            self.data.append(dataline)
            targetIndex = len(lineAsArray) - 1
            self.target.append(lineAsArray[targetIndex])

        # data and target are both arrays of arbitrary length.
        #
        # data is an array of arrays of integers (0 or 1) indicating state.
        #
        # target is an array of imtegers 0-3 indicating the action
        # taken in that state.
            
        # X_train, X_test, y_train, y_test = train_test_split(self.data, self.target, test_size=0.1, random_state=0)
        # classifier = tree.DecisionTreeClassifier().fit(X_train, y_train)
        # ac_score = metrics.accuracy_score(y_test, classifier.predict(X_test))

        rm = RandomForestClassifier().fit(self.data, self.target)
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(self.data, self.target)
        # ac_score2 = metrics.accuracy_score(y_test, rm.predict(X_test))
        # print (ac_score)
        self.sklearn_classifier = rm
        self.classifier = decision_tree

        mc = 0
        mw = 0
        sc = 0
        sw = 0
        for x, y in zip(self.data, self.target):
            if decision_tree.predict(x) == y:
                mc += 1  
            else: 
                mw += 1
            if rm.predict([x]) == y:
                sc += 1  
            else: 
                sw += 1

        
            # mc += 1 if decision_tree.predict(self.data[i]) == self.target[i] else mw += 1
            # sc += 1 if sklearn_classifier.predict([self.data[i]]) == self.target[i] else sw += 1
        
        print (mc, mw)
        print (sc, sw)


        
    # Tidy up when Pacman dies
    def final(self, state):

        print "I'm done!"
        
        # *********************************************
        #
        # Any code you want to run at the end goes here.
        #
        # *********************************************

    # Turn the numbers from the feature set into actions:
    def convertNumberToMove(self, number):
        if number == 0:
            return Directions.NORTH
        elif number == 1:
            return Directions.EAST
        elif number == 2:
            return Directions.SOUTH
        elif number == 3:
            return Directions.WEST

    # Here we just run the classifier to decide what to do
    def getAction(self, state):

        # How we access the features.
        features = api.getFeatureVector(state)
        
        # *****************************************************
        #
        # Here you should insert code to call the classifier to
        # decide what to do based on features and use it to decide
        # what action to take.
        #
        # *******************************************************

        # Get the actions we can try.
        legal = api.legalActions(state)

        move = self.classifier.predict(features)
        move1 = self.sklearn_classifier.predict([features])
        move = self.convertNumberToMove(move)
        # move = self.convertNumberToMove(move[0])
        # getAction has to return a move. Here we pass "STOP" to the
        # API to ask Pacman to stay where they are. We need to pass
        # the set of legal moves to teh API so it can do some safety
        # checking.
        return api.makeMove(move, legal)




class DecisionTreeClassifier(object):
    
    def __init__(self, max_depth=100, random_state=0):
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        self.root = Node(max_depth=self.max_depth, random_state=self.random_state, X=X, y=y)
        self.root.train()
        # children = self.root.children.values()
        # i = 0
        # while children != []:
        #     new_children = []
        #     print i
        #     for child in children:
        #         print "\t", child.depth
        #         for child_n in child.children.values():
        #             new_children.append(child_n)
        #     children = new_children 
        #     i += 1

    def predict(self, X):
        return self.root.predict(X)

    



class Node(object):
    
    FEATURES_VALUES_SET = [0, 1]

    def __init__(self, feature_value=None, max_depth=100, depth=0, random_state=0, X=[], y=[], parent=None):
        # self.labels = {label: [] for label in labels}
        self.feature_value = feature_value
        self.max_depth = max_depth
        self.depth = depth
        self.random_state = random_state
        self.X = np.array(X[:])
        self.y = np.array(y[:])
        self.M = len(X)
        self.parent = parent

        # feature value: elements
        self.children = {}
        self.prediction = None

    # def populate(self, elements, labels):
    #     for elements, label in zip(elements, labels):
    #         self.add_elements(elements, label)
    #     self.elements_count += len(labels)

    # def append_child(self, node):
    #     self.children.append(node)

    # def append_children(self, nodes):
    #     self.children += nodes

    def add_element(self, element, label):
        self.M += 1
        self.X = np.append(self.X, element).reshape(self.M, len(element))
        self.y = np.append(self.y, label)

    # def add_elements(self, label, elements):
    #     self.labels[label] += elements
    #     self.elements_count += len(elements)

    # def get_elements_from_label(self, label):
    #     return self.labels[label]

    def get_entropy(self):
        entropy = 0
        if self.M == 0:
            return entropy

        label_to_elements = self.group_elements_by_label()

        for elements in label_to_elements.values():
            count = float(len(elements)) / self.M
            entropy -= (count * self._log2(count))
        return entropy

    def get_information_gain(self, children):
        self_entropy = self.get_entropy()
        children_entropy = 0
        for child in children:
            children_entropy += child.M * child.get_entropy() / self.M 
        return self_entropy - children_entropy

    def _log2(self, number):
        if number == 0:
            return 0
        return log(number, 2)

    def train(self):
        # there exists just one label
        if len(set(self.y)) == 1:
            self.prediction = self.y[0]
        
        # there is more than one label, but further split is allowed
        elif self.max_depth > self.depth:
            # no more features can be used to split the tree
            if self.depth == 25:
                pass
            if self.y.size == 0:
                self.prediction = self.parent._get_max_label()
            elif self.X.size == 0:
                self.prediction = self._get_max_label()
            else:
                best_information_gain = -1
                best_children = []
                best_feature_index = None
                for feature_index in range(len(self.X[0])):
                    children = self.split_node_by_feature_at_index(feature_index)
                    information_gain = self.get_information_gain(children.values())
                    if information_gain > best_information_gain:
                        best_information_gain = information_gain
                        best_children = children
                        best_feature_index = feature_index
                self.children = best_children
                self.feature_index_used_to_split = best_feature_index
                for child in self.children.values():
                    child.train()
        
        # the tree has reached the max_depth, prediction will be
        # based on the current knowledge
        else:
            self.prediction = self._get_max_label()
        


    def split_node_by_feature_at_index(self, feature_index):
        feature_value_to_node = {}
        for feature_value in self.FEATURES_VALUES_SET:
            new_node = Node(max_depth=self.max_depth, 
                            depth=(self.depth+1), 
                            random_state=self.random_state,
                            feature_value=feature_value,
                            parent=self)
            feature_value_to_node[feature_value] = new_node

        X_by_feature = self.X[:, feature_index]

        for i in range(self.M):
            node = feature_value_to_node[int(X_by_feature[i])]
            X = np.delete(self.X[i], feature_index)
            y = self.y[i]
            node.add_element(X, y)

        return feature_value_to_node


    def group_elements_by_label(self):
        label_to_elements = {label: [] for label in set(self.y)}
        for element, label in zip(self.X, self.y):
            label_to_elements[label].append(element)
        return label_to_elements


    def predict(self, X):
        if self.prediction != None:
            # the node is a leaf
            return self.prediction
        else:
            x = X[self.feature_index_used_to_split]
            next_node = self.children[x]
            skimmed_X = np.delete(X, self.feature_index_used_to_split)
            child_prediction = next_node.predict(skimmed_X)
            if child_prediction == None:
                child_prediction = self._get_max_label()
            return child_prediction

    def _get_max_label(self):
        label_to_elements = self.group_elements_by_label()
        max_count = -1
        prediction_label = None
        for label, elements in label_to_elements.items():
            current_count = len(elements)
            if current_count > max_count:
                max_count = current_count
                prediction_label = label
        return prediction_label