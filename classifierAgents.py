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

from math import log, ceil, sqrt
from collections import Counter

# ClassifierAgent
#
# An agent that runs a classifier to decide what to do.
class ClassifierAgent(Agent):

    counter = 0

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
        self.classifier = RandomForestClassifier(n_estimators=100)
        self.classifier.fit(self.data, self.target)

        
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
        move = self.convertNumberToMove(move)
        
        # getAction has to return a move. Here we pass "STOP" to the
        # API to ask Pacman to stay where they are. We need to pass
        # the set of legal moves to teh API so it can do some safety
        # checking.
        return api.makeMove(move, legal)




class Node(object):

    # the set of values that each feature may assume
    FEATURES_VALUES_SET = {0, 1}

    def __init__(self, 
                 feature_value, 
                 depth,
                 parent, 
                 X,
                 y,
                 max_depth=100):
        '''
            The node object used inside Random Tree Classifiers.
            

            Parameters
            ----------
            feature_value : {0, 1}
                It refers to the value of the feature used to make 
                the split in the parent node. In other words, each 
                element of X in this node used to have a feature 
                (while was still contained in the parent node, and
                then removed after the split) with value equal to 
                feature_value. As the features that each node deals 
                with only have 2 possible values (0 or 1), the 
                feature_value shall assume either of these values.  

            
            depth : integer
                The current depth of this node. This is compared to 
                the max depth to stop growing the tree if necessary.

            
            parent : Node
                The reference to the parent node.

            
            X : np.array
                The input features assigned to this node for the 
                training. The elements of X must be in turn vectors.
                The node expects to receive an X which does not include
                features already used for a split in the ancestor nodes.

            
            y : np.array
                The input labels assigned to this node for the
                training with respect to vector X.

            
            max_depth : integer, optional (default=100)
                The max depth a node can reach before stop splitting
                (i.e. growing).


            Attributes
            ----------
            feature_index_used_to_split : integer
                The index of the feature used to make the node split.
                Notice that this index refers to the current node, not
                the parent.


            prediction : {0, 1, `None`}
                During the training, each node is assigned a prediction, 
                i.e., the label that most likely represents an input
                datapoint which reaches this a node. It defaults to `None`,
                but should be updated after the training for this node is 
                complete. If after the training the prediction is still `None`, 
                it means that this node is not able to make a good prediction
                and the tree shall be further traversed to get a good prediction.


            children : dict
                The children generated by a split from this node. 
                The dict-keys are the values that a feature may 
                assume (0 or 1). The respective dict-values are 
                the nodes which contain all the Xs which feature-value 
                equals the dict-key. 

                e.g.
                    Say the feature at index 0 was used to split the 
                    node:

                    parent_node.X = [
                        [0, 1, 0, 1],
                        [0, 1, 0, 1],
                        [1, 1, 1, 1],
                        [1, 1, 0, 0]
                    ]

                    parent_node.children = {
                        0 : child_node_0,
                        1 : child_node_1
                    }

                    child_node_0.X = [
                        [1, 0, 1],
                        [1, 0, 1]
                    ],

                    child_node_1 = [
                        [1, 1, 1],
                        [1, 0, 0]
                    ]
        '''        
        self.max_depth = max_depth
        self.depth = depth
        self.parent = parent
        self.X = np.array(X)
        self.y = np.array(y)

        self.feature_index_used_to_split = None
        self.prediction = None
        self.children = {}


    def get_entropy(self):
        '''
            Calculate and return the entropy of the current node.


            Return
            ------
            The entropy value of this node.
        '''
        entropy = 0
        if len(self.X) == 0:
            return entropy

        counter = Counter(self.y)
        for label_count in counter.itervalues():
            # If the count a label is 0, such a label does does not
            # increase the final entropy, hence, we do not consider it.
            if label_count > 0:
                label_probability = float(label_count) / len(self.X)
                entropy -= (label_probability * log(label_probability, 2))

        return entropy


    def get_information_gain(self, children):
        '''
            Calculate the entropy of this node and the ones
            of its children, then return the information gain.


            Return
            ------
            The information gain achieved by the split
            operated by this node.
        '''
        self_entropy = self.get_entropy()
        children_entropy = 0
        for child in children:
            children_entropy += len(child.X) * child.get_entropy() / len(self.X) 
        return self_entropy - children_entropy


    def fit(self):
        '''
            Train this node based on the input vector X (and
            the respective labels y) assigned to this node.
            
            If given such data, the node is able to estimate 
            with no doubt the label of an unseen data which
            reaches this node, or if the training is no longer 
            possible, stop the training and assign to this node
            a prediction.

            If training is still possible, generate the children
            nodes and recursively train them.
        '''
        if len(set(self.y)) == 1:
            # there exists just one label in this node,
            # so the prediction is such label
            self.prediction = self.y[0]
        
        elif self.max_depth == self.depth:
            # the tree has reached the max_depth, so the
            # prediction is based on the current labels
            self.prediction = self.get_most_frequent_label()

        elif len(self.X) == 0:
            # no samples are contained in this node after the
            # split, so prediction is based on parent labels
            self.prediction = self.parent.get_most_frequent_label() 

        elif len(self.X[0]) == 0:
            # all the features have been used and more than 
            # one label is still present in this node, so
            # prediction is based on the current labels
            self.prediction = self.get_most_frequent_label()

        else:
            # training is still possible, so
            # split the node into 2 children and train them
            self.split_node_by_feature()
            for child in self.children.values():
                child.fit()
        

    def split_node_by_feature(self):
        '''
            Randomly select N features from the unused ones
            (i.e all the features that are contained in this 
            node's X) where N is defined as  
            N = ceil( sqrt(total unused features) ) 
                    < like suggested in [1] > 
            Using only these N features, find the one
            which, if used to split this node, maximises the 
            information gain. Store the index of such a feature in 
            `feature_index_used_to_split`, use it to generate the 
            children, and store the latter.


            References
            ----------
            [1] Book: "Hands-On Machine Learning with R" 
                by Bradley Boehmke & Brandon Greenwell,
                Jan 02, 2020 on Chapter 11 Random Forests
                from website https://bradleyboehmke.github.io/HOML/random-forest.html

        '''
        best_information_gain = -1
        best_children = []
        best_feature_index = None

        # generate a random sample of features to consider
        # for the split
        n_features = len(self.X[0])
        features_used_to_split = int( ceil( sqrt(n_features) ) )
        random_features_indexes = np.random.choice(n_features, size=features_used_to_split, replace=False)

        # identify the split with the best information gain
        for feature_index in random_features_indexes:
            children = self.generate_children(feature_index)
            information_gain = self.get_information_gain(children.values())
            if information_gain > best_information_gain:
                best_information_gain = information_gain
                best_children = children
                best_feature_index = feature_index

        self.children = best_children
        self.feature_index_used_to_split = best_feature_index


    def generate_children(self, feature_index):
        '''
            Given the index of a feature belonging to
            vector X, generate one new node for each 
            value that such a feature may assume (as
            we are only dealing with binary values in our
            classifier, the number of nodes generated this
            way will be 2).

            Store the nodes as dicts (in the variable 
            called `feature_value_to_node`), where the key is 
            the value of the feature, and the value of
            such a key is the node that contains all the
            X elements which value for the said feature 
            is equal to the key of the entry.

            Notice that, upon a split, the feature used for 
            such a split will be removed from the X in new 
            node, as each node's X only contains features not 
            yet used for a split.

            Look at the example for a better understanding.
                

            Parameter
            ---------
            feature_index : integer
                The index of the feature to use to split the 
                node.


            Return
            ------
            A dict having as keys either value of {0, 1}, and
            the respective value for each key k is the node
            containing all the elements of X which value for 
            the feature at index `feature_index` is equal to the key k.
            The X of the children nodes will not include
            the feature used for the split (i.e the feature
            at index `feature_index`) as the X of any node
            only includes those features that have not been
            used for a split in any ancestors. 
            
            Look at the example for a better understanding.


            Example
            -------
            self.X = [
                [0, 1, 0, 1],
                [0, 1, 0, 1],
                [1, 1, 1, 1],
                [1, 1, 0, 0]
            ]

            feature_index = 0
            feature_value_to_node = {
                0 : child_node_0,
                1 : child_node_1
            }

            child_node_0.X = [
                [1, 0, 1],
                [1, 0, 1]
            ],
            
            child_node_1 = [
                [1, 1, 1],
                [1, 0, 0]
            ]
        '''
        
        feature_values = self.X[:, feature_index]
        feature_value_to_node = {}
        
        # for each value that a feature may assume, create a new node
        # and save an entry (value -> node) in the dict feature_value_to_node
        for feature_value in self.FEATURES_VALUES_SET:
            rows = self.X[:, feature_index] == feature_value
            new_node_X = np.delete(self.X[rows], feature_index, axis=1)
            new_node_y = self.y[rows]
            new_node = Node(max_depth=self.max_depth, 
                            depth=(self.depth+1),
                            feature_value=feature_value,
                            parent=self,
                            X=new_node_X,
                            y=new_node_y)
            feature_value_to_node[feature_value] = new_node

        return feature_value_to_node


    def predict(self, X):
        '''
            Predict the class for an input vector X.

            If this node is associated with a label (prediction),
            return it, otherwise, keep traversing the tree removing
            from X the feature used to reach the next (child) node.


            Parameters
            ----------
            X : list
                The vector of input samples.


            Returns
            -------
            The predicted label for the input vector X.
        '''
        if self.prediction != None:
            # the node is a leaf
            return self.prediction
        else:
            next_node_key = X[self.feature_index_used_to_split]
            next_node = self.children[next_node_key]

            # remove the feature used to reach this node
            new_X = np.delete(X, self.feature_index_used_to_split)
            return next_node.predict(new_X)


    def get_most_frequent_label(self):
        '''
            Based on the labels assigned to this node, find 
            the label that occurs more often. If more than 
            one label occurs with the same max frequency,
            randomly return one of them. 

            Notice: the label is randomly picked, but using
            a random seed, as such, the method is replicable.
        '''
        labels_counter = Counter(self.y)
        most_frequent_label_count = labels_counter.most_common(1)[0][-1] 
        most_frequent_labels = [
            label 
            for label, count in labels_counter.iteritems() 
            if count == most_frequent_label_count
        ]
        return np.random.choice(most_frequent_labels)


    
class Root(Node):

    def __init__(self, max_depth=100):
        '''
            A special type of node which is always
            the first in a tree. This can in fact be considered
            as a Tree in a Random Forest.


            Parameter
            ---------
            max_depth : integer, optional (default=100)
                The max depth a node can reach before stopping
                to grow.
        '''
        super(Root, self).__init__(
            feature_value=None,
            depth=0, 
            parent=None,
            max_depth=max_depth,
            X=[],
            y=[]
        )


    def fit(self, X, y):
        '''
            Override of Node.fit method.
            As the root object is the only node which does not 
            receive the vectors X and y upon creation, this method
            will set such parameters for it, and then invoke the
            parent method fit.


            Parameter
            ---------
            X : numpy.array
                The feature vector to use for the training.
                The elements of X must be in turn vectors.

            
            y : numpy.array
                The vector of the labels for the input vector X. 
        '''
        self.X = X
        self.y = y

        super(Root, self).fit()



class RandomForestClassifier(object):

    def __init__(self, 
                 max_depth=100, 
                 random_state=0, 
                 n_estimators=100, 
                 probabilistic=False):
        '''
            A random forest classifier. This class also acts as 
            a converter for the trees since it converts all the 
            lists object into numpy arrays.
            

            Parameters
            ----------
            max_depth : integer, optional (default=100)
                The maximum depth of each tree.

            
            random_state : integer, optional (default=0)
                The number used to set the random seed.

            
            n_estimators : integer, optional (default=5)
                The number of trees in the forest.

            
            probabilistic : boolean, optional (default=`False`)
                The way the random forest decide on which prediction to 
                consider correct. If probabilistic is set to `True`, each 
                vote (NOT prediction) has equal probability of being picked 
                as the final prediction. If `False`, the most voted prediction 
                is used instead.
        '''
        self.max_depth = max_depth
        np.random.seed(random_state)
        self.n_estimators = n_estimators
        self.probabilistic = probabilistic


    def fit(self, X, y):
        '''
            Generate as many random trees as specified
            by the instance attribute `n_samples`. For
            each instance, bootstrap the elements. More
            specifically, the bootstrapped vector should 
            have as many samples as the elements in X (as
            reccommended in the reference [1]).


            Parameters
            ----------
            X : list
                The feature vector to use for the training.
                The elements of X must be in turn vectors.

            
            y : list
                The vector of the labels for the input vector X.


            References
            ----------
            [1] Article: "Understanding Random Forest" by Tony Yiu, 
                Jun 12, 2019 on website https://towardsdatascience.com
        '''
        # map input array to numpy arrays for the nodes
        X = np.array(X)
        y = np.array(y)

        self.trees = []

        # zip X and y together for bootstrapping
        Xy_pairs = np.array(zip(X, y))

        for i in range(self.n_estimators):
            
            bootstrapped_data = self.bootstrap(Xy_pairs)
            # unpack bootstrapped data
            bootstrapped_X = np.array([datapoint[0] for datapoint in bootstrapped_data])
            bootstrapped_y = bootstrapped_data[:, 1]
            tree = Root(max_depth=self.max_depth)
            tree.fit(bootstrapped_X, bootstrapped_y)
            self.trees.append(tree)


    def predict(self, X):
        '''
            Predict class for feature vector X.

            If the instance attribute probabilistic is set to `True`, 
            each vote (NOT prediction) has equal probability of being 
            picked as the final prediction. If `False`, the most voted 
            prediction is used instead.


            Parameters
            ----------
            X : list
                The vector of input samples.


            Returns
            -------
            The predicted label for the input vector X.
        '''
        X = np.array(X)
        predictions = [tree.predict(X) for tree in self.trees]

        if self.probabilistic:
            # prediction is taken randomly among all the votes
            # of the trees.
            max_prediction = np.random.choice(predictions)
        else:
            # the final prediction is the most voted one (in case
            # of tie, a random one is selected with equal probability)

            predictions_counter = Counter(predictions)
            # find how many times the most frequent prediction occurs
            most_frequent_prediction_count = predictions_counter.most_common(1)[0][-1] 
            # more than one prediction could appear with the most frequency
            # so we include all of them and then randomly select one
            most_frequent_predictions = [
                prediction 
                for prediction, count in predictions_counter.iteritems() 
                if count == most_frequent_prediction_count
            ]
            max_prediction = np.random.choice(most_frequent_predictions)
        return max_prediction

    
    def bootstrap(self, bootstrapable_list, n_samples=None):
        '''
            A data bootstrapper, i.e, a random selector of 
            elements in a list with repetitions allowed. 
            Given an input list `bootstrapable_list`, return 
            n elements from it where n is defined by the 
            optional integer parameter `n_samples`. If 
            `n_samples` is not passed, return as many samples 
            as the elements of `bootstrapable_list`.


            Parameters
            ----------
            bootstrapable_list : numpy.array
                The list to be bootstrapped.
            

            n_samples : integer, optional (default=None)
                The length of the final bootstrapped list to
                be returned. If n_samples is not specified, 
                the bootstrapped list will have the same length 
                as the input list `bootstrapable_list`.


            Returns
            -------
            A bootstrapped list derived from the input list 
            bootstrapable_list, and with length equal to the 
            input parameter n_samples.
        '''
        if n_samples == None:
            n_samples = len(bootstrapable_list)
        random_indexes = np.random.randint(len(bootstrapable_list), size=n_samples)
        bootstrapped_list = bootstrapable_list[random_indexes] 
        return bootstrapped_list