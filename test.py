from math import log
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class DecisionTreeClassifier(object):
    
    def __init__(self, max_depth=10, random_state=0):
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        self.root = Node(max_depth=self.max_depth, random_state=self.random_state, X=np.array(X), y=np.array(y))
        self.root.train()

    def predict(self, X):
        return self.root.predict(X)

    



class Node(object):
    
    FEATURES_VALUES_SET = [0, 1]

    def __init__(self, feature_value=None, max_depth=10, depth=0, random_state=0, X=[], y=[]):
        # self.labels = {label: [] for label in labels}
        self.feature_value = feature_value
        self.max_depth = max_depth
        self.depth = depth
        self.random_state = random_state
        self.X = X[:]
        self.y = y[:]
        self.M = len(X)

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
        self.X.append(element)
        self.y.append(label)
        self.M += 1

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
            # no more features can be used to split
            if self.X.shape[1] > 0:
                # labels exist
                if self.y != []:
                    self.prediction = self._get_max_label()
                else:
                    self.prediction = None

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
        
        breakpoint = NotImplemented 


    def split_node_by_feature_at_index(self, feature_index):
        feature_value_to_node = {}
        for feature_value in self.FEATURES_VALUES_SET:
            new_node = Node(max_depth=self.max_depth, 
                            depth=self.depth+1, 
                            random_state=self.random_state,
                            feature_value=feature_value)
            feature_value_to_node[feature_value] = new_node

        X_by_feature = self.X[:, feature_index]

        for i in range(self.M):
            node = feature_value_to_node[int(X_by_feature[i])]
            X = np.delete(self.X[i], feature_index)
            y = self.y[i]
            node.depth = 3
            node.add_element(X, y)

        return feature_value_to_node


    def group_elements_by_label(self):
        label_to_elements = {label: [] for label in set(self.y)}
        for element, label in zip(self.X, self.y):
            label_to_elements[label].append(element)
        return label_to_elements


    def predict(self, X):
        if self.prediction:
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
            ## TODO re-implement
            current_count = len(elements)
            if current_count > max_count:
                max_count = current_count
                prediction_label = label
        return prediction_label

Xy = ["10100101000000000000000001","10100100000000000000000001","10100100000000000000000001","10100100000000000000000001","01101000000000000000000000","01011000000000000000000000","00001101000000000000000000","01011000000000000000000000","00011100000000000000000000","01011000000000000000000000","00001101000000000000000001","10100100000000000000000001","00101100000000000000000001","11000010000000000000000002","01010010000000000000000002","00000111000000000000000002","01010010000000000000000002","01100001000000000000000003","10000011000000000000000003","10100000000000000000000003","00000001000000000000000001","10100000000000000000000001","10000010000000000000000001","01100000000000000000000003","10000010000000000000000003","10100000000000000000000003","00000001000000000000000003","10100001000000000000000003","10100001000000000000000003","10100001000000000000000003","10100001000000000000000003","10100001000000000000000003","10100001000000000000000003","00001011000000000000000002","01010010000000000000000002","00110100000000000000000001","10100100000000000000000001","10100000000000000000000001","10100000000000000000000001","10100000000000000000000001","10100000000000000000000001","10100000000000000000000001","01100000000000000000000000","01010000000000000000000000","00000000000000000000000000","01010000000000000000000000","00010100000000000000000000","01010000000000000000000000","00001001000000000000000000","01011000000000000000000000","11000001000000000000000003","10100001000000000000000003","10100001000000000000000003","10100001000000000000000003","10100001000000000000000003","10100001000000000000000003","10100001000000000000000003","10010010000000000000000002","01010010000001000000000002","00000111000000000000000002","01010010000000000000000002","01000011000000000000000002","01010000000000000000000002","00000001000000000000000002","01010000000000000000000002","00110000000000000000000001","10100000000000000000000001","10100000000000000000000001","10100000000000000000000001","10100000000000000000000001","10100000000000000000000001","10100000000000000000000001","01100000000000000000000000","01010000000000000000000000","00000000000000000000000001","10100000000000000000000001","10000010000000000000000001","01100000000000000000000003","10000010000000000000000002","01010010000000000000000002","00110100000000000000000001","10100100000000000000000001","10100000000000000000000001","01101000000000000000000000","01011000000000000000000000","01011000000000000000000000","01011000000000000000000000","01001001000000000000000000","01011000000000000000000000","01011000000000000000000000","01011000000000000000000000","11000001000000000000000003","10100001000000000000000003","10100001000000000000000003","10010010000000000000000002","01010000000000000000000002","00100000000000000000000003","10100000000000000000000003","00000001000000000000000003","10100001000000000000000003","10100001000000000000000003","10000001000000000000000003","10000001000000000000000003","10100001000000000000000003","10100000000000000000000003","00000001000000000000000003","10100001000000000000000003","00101001000000000000000000","01011000000000000000000000","11000001000000000000000003","10100001000000000000000003","10100000000000000000000003","10010010000000000000000002","01010010000000000000000002","01010010000000000000000012","01010010000000000000000002","00010110000000000000000002","01010010000000000000000002","01010010000000000000000002","01010010000000000000000002","00110100000000000000000001","10100100000000000000000001","10100100000000000000000001","01101000000000000000000000","01011000000000000000000000","10000101000000001000000003"]
Xy = [map(lambda l: int(l), list(line)) for line in Xy]
X = np.array(Xy)[:,:-1]
y = np.array(Xy)[:, -1]


tree = DecisionTreeClassifier()
tree.fit(X, y)

sk_tree = DecisionTreeClassifier()
sk_tree.fit(X, y)


for i in range(100):
    print X[i]
    print tree.predict(X[i]), sk_tree.predict(X[i])