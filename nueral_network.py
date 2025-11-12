"""This program creates a FFBP nueral network. """
import collections
import math

import numpy
import numpy as np
from enum import Enum
import random
from collections import deque
from abc import ABC, abstractmethod
import copy
import json


class DataMismatchError(Exception):
    """ Label and example lists have different lengths"""


class NNData:
    """ Maintain and dispense examples for use by a Neural
    Network Application
    """

    class Order(Enum):
        """Indicate whether data will be shuffled for each new epoch"""
        RANDOM = 0
        SEQUENTIAL = 1

    class Set(Enum):
        """ Indicate which set should be accessed or manipulated """
        TRAIN = 0
        TEST = 1

    @staticmethod
    def percentage_limiter(train_factor):
        """ Ensure that percentage is bounded between 0 and 1 """
        return min(1, max(train_factor, 0))

    def __init__(self, features=None, labels=None, train_factor=.9):
        self._train_factor = NNData.percentage_limiter(train_factor)
        if features is None:
            features = []
        if labels is None:
            labels = []
        self._features = []
        self._labels = []
        self._train_indices = []
        self._test_indices = []
        self._train_pool = deque()
        self._test_pool = deque()
        try:
            self.load_data(features, labels)
        except (ValueError, DataMismatchError):
            raise DataMismatchError

    def split_set(self, new_train_factor=None):
        """This method creates lists of indicies to be used indirectly
        by features and labels.
        """
        if new_train_factor is not None:
            self._train_factor = NNData.percentage_limiter(new_train_factor)
        if self._features is None:
            self._features = []
        numberofexamples = len(self._features)
        trainindexlist = list(range(len(self._features)))
        trainingexamples = numberofexamples * self._train_factor
        train_indices = random.sample(trainindexlist, k=int(trainingexamples))
        self._train_indices = train_indices
        self._test_indices = \
            list(set(trainindexlist).difference(set(self._train_indices)))
        random.shuffle(self._train_indices)
        random.shuffle(self._test_indices)
        self.prime_data()

    def prime_data(self, target_set=None, order=None):
        """This method takes the target_set and assigns it to either
        training pool or testing pool."""
        if target_set is NNData.Set.TEST:
            if order is NNData.Order.RANDOM:
                randomlist = random.sample(self._test_indices,
                                           k=len(self._test_indices))
                self._test_pool = deque(randomlist)
                random.shuffle(self._test_pool)
            else:
                self._test_pool = deque(self._test_indices)
        elif target_set is NNData.Set.TRAIN:
            if order is NNData.Order.RANDOM:
                randomlist2 = random.sample(self._train_indices,
                                            k=len(self._train_indices))
                self._train_pool = deque(randomlist2)
                random.shuffle(self._train_pool)
            else:
                self._train_pool = deque(self._train_indices)
        elif target_set is None:
            if order is NNData.Order.RANDOM:
                randomlist = random.sample(self._test_indices,
                                           k=len(self._test_indices))
                random.shuffle(randomlist)
                self._test_pool = deque(randomlist)
                randomlist2 = random.sample(self._train_indices,
                                            k=len(self._train_indices))
                random.shuffle(randomlist2)
                self._train_pool = deque(randomlist2)
            else:
                self._test_pool = deque(self._test_indices)
                self._train_pool = deque(self._train_indices)

    def get_one_item(self, target_set=None):
        """This method uses the target set to return a tuple of features
        and labels
        """
        if target_set is None:
            try:
                return_value = self._train_pool.popleft()
                return (self._features[return_value],
                        self._labels[return_value])
            except IndexError:
                return None
        elif target_set == NNData.Set.TEST:
            try:
                return_value = self._test_pool.popleft()
                return (self._features[return_value],
                        self._labels[return_value])
            except IndexError:
                return None
        elif target_set == NNData.Set.TRAIN:
            try:
                return_value = self._train_pool.popleft()
                return (self._features[return_value],
                        self._labels[return_value])
            except IndexError:
                return None

    def pool_is_empty(self, target_set=None):
        """This method returns true if either the train pool or
        testing pool is empty and false otherwise.
        """
        if target_set is NNData.Set.TRAIN:
            if self._train_pool == deque():
                return True
            else:
                return False
        elif target_set is None:
            if self._train_pool == deque():
                return True
            else:
                return False
        elif target_set is NNData.Set.TEST:
            if self._test_pool == deque():
                return True
            else:
                return False

    def number_of_samples(self, target_set=None):
        """This method uses the target_set to return the correct
        number of items in the pool.
        """
        if target_set is NNData.Set.TEST:
            numberoftest = len(self._test_pool)
            return numberoftest
        elif target_set is NNData.Set.TRAIN:
            return len(self._train_pool)
        else:
            return len(self._train_pool) + len(self._test_pool)

    def load_data(self, features: list = None, labels: list = None):
        """ Load feature and label data, with some checks to ensure
        that data is valid
        """
        if features is None or labels is None:
            self._features = None
            self._labels = None
            return
        if len(features) != len(labels):
            self._features = None
            self._labels = None
            self.split_set()
            raise DataMismatchError("Label and example lists have "
                                    "different lengths")
        if len(features) > 0:
            if not (isinstance(features[0], list)
                    and isinstance(labels[0], list)):
                self._features = None
                self._labels = None
                self.split_set()
                raise ValueError("Label and example lists must be "
                                 "homogeneous numeric lists of lists")
        try:
            self._features = np.array(features, dtype=float)
            self._labels = np.array(labels, dtype=float)
        except ValueError:
            self._features = None
            self._labels = None
            self.split_set()
            raise ValueError("Label and example lists must be homogeneous "
                             "and numeric lists of lists")
        self.split_set()


class LayerType(Enum):
    """This class creates layers to the neural network"""
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class Node:
    """This class creates nodes for the doublylinked list."""

    def __init__(self, data=None):
        self.data = data
        self.next = None
        self.prev = None


class MultiLinkNode(ABC):
    """
    This class is the parent class for nodes in the neural network.
    """

    class Side(Enum):
        """This class creates Enums for the direction of flow."""
        UPSTREAM = 0
        DOWNSTREAM = 1

    def __init__(self):
        self._reporting_nodes = {MultiLinkNode.Side.UPSTREAM: 0,
                                 MultiLinkNode.Side.DOWNSTREAM: 0}
        self._reference_value = {MultiLinkNode.Side.UPSTREAM: 0,
                                 MultiLinkNode.Side.DOWNSTREAM: 0}
        self._neighbors = {MultiLinkNode.Side.UPSTREAM: [],
                           MultiLinkNode.Side.DOWNSTREAM: []}

    def __str__(self):
        print(f"Node ID: {id(self)}, "
              f"Upstream: {self._neighbors[MultiLinkNode.Side.UPSTREAM]}, "
              f"Downstream: {self._neighbors[MultiLinkNode.Side.DOWNSTREAM]}")

    @abstractmethod
    def _process_new_neighbor(self, node, side):
        pass

    def reset_neighbors(self, nodes: list, side: Side):
        """ This method takes a list of nodes and either resets or set
        the nodes connected to them depending on the side.
        """
        self._neighbors[side] = nodes.copy()
        for node in nodes:
            self._process_new_neighbor(node, side)
        self._reference_value[side] = (1 << len(nodes)) - 1
        self._reporting_nodes[side] = 0


class Neurode(MultiLinkNode):
    """This class inherates from the MultiLinkNode class to create
    class that other Neurodes can inherate from.
    """

    def __init__(self, node_type, learning_rate=.05):
        self._value = 0
        self._node_type = node_type
        self._learning_rate = learning_rate
        self._weights = {}
        super().__init__()

    @property
    def value(self):
        """This property is a getter for the self._value."""
        return self._value

    @property
    def node_type(self):
        """This property is a getter for self._node_type."""
        return self._node_type

    @property
    def learning_rate(self):
        """This propert is a getter for self._learning_rate."""
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, new_learning_rate: float):
        """This setter establishes the learning rate. """
        self._learning_rate = new_learning_rate

    def _process_new_neighbor(self, node, side):
        """This method assigns a random float to a node."""
        if side == MultiLinkNode.Side.UPSTREAM:
            value = random.random()
            self._weights[node] = value

    def _check_in(self, node, side):
        """This method will update reporting_nodes, compare it to the
        reference value, if they are equal return true.
        """
        if side is MultiLinkNode.Side.UPSTREAM:
            nodeindex = \
                self._neighbors[MultiLinkNode.Side.UPSTREAM].index(node)
            self._reporting_nodes[MultiLinkNode.Side.UPSTREAM] = \
                self._reporting_nodes[MultiLinkNode.Side.UPSTREAM] | \
                2 ** nodeindex
            if self._reporting_nodes[MultiLinkNode.Side.UPSTREAM] == \
                    self._reference_value[MultiLinkNode.Side.UPSTREAM]:
                self._reporting_nodes[MultiLinkNode.Side.UPSTREAM] = 0
                return True
            else:
                return False
        elif side is MultiLinkNode.Side.DOWNSTREAM:
            nodeindex = \
                self._neighbors[MultiLinkNode.Side.DOWNSTREAM].index(node)
            self._reporting_nodes[MultiLinkNode.Side.DOWNSTREAM] = \
                self._reporting_nodes[MultiLinkNode.Side.DOWNSTREAM] | \
                2 ** nodeindex
            if self._reporting_nodes[MultiLinkNode.Side.DOWNSTREAM] == \
                    self._reference_value[MultiLinkNode.Side.DOWNSTREAM]:
                self._reporting_nodes[MultiLinkNode.Side.DOWNSTREAM] = 0
                return True
            else:
                return False

    def get_weight(self, node):
        """This method returns the weight of a specific node. """
        for item in self._weights:
            if node == item:
                return self._weights[item]


class FFNeurode(Neurode):
    """This class creates feedforward objects and inherates from the
    Neurode class.
    """

    def __init__(self, my_type):
        super().__init__(my_type)

    @staticmethod
    def _sigmoid(value):
        """This method returns the sigmoid value of a Neurode. """
        sigmoidvalue = (1 / (1 + np.exp(-value)))
        return sigmoidvalue

    def _calculate_value(self):
        """This method calculates a value for the Neurode."""
        itemlist = []
        for item in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            weight = self._weights[item]
            value = item.value
            weighttimesvalue = weight * value
            itemlist.append(weighttimesvalue)
        self._value = self._sigmoid(sum(itemlist))

    def _fire_downstream(self):
        """This method calls data_ready_upstream on downstream neighbors
        passing self as an argument.
        """
        for item in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            item.data_ready_upstream(self)

    def data_ready_upstream(self, node):
        """
        This method checks in upstream neurodes, gives them a value,
        and calls fire_downstream.
        """
        if self._check_in(node, MultiLinkNode.Side.UPSTREAM) is True:
            self._calculate_value()
            self._fire_downstream()

    def set_input(self, input_value):
        """This method assigns a value to the neurode and lets the calls
        data_ready_upstream on downstream neighbors to let them know it
        has data.
        """
        self._value = input_value
        for item in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            item.data_ready_upstream(self)


class BPNeurode(Neurode):
    """This class establishes the Back propagation Neurode and inherits
    from the Neurode class.
    """

    def __init__(self, my_type):
        super().__init__(my_type)
        self._delta = 0

    @property
    def delta(self):
        """Returns self._delta"""
        return self._delta

    @staticmethod
    def _sigmoid_derivative(value):
        """Returns the sigmoid derivative of a value."""
        result = value * (1 - value)
        return result

    def _calculate_delta(self, expected_value=None):
        """Calculates delta for neurodes to adjust weights. """
        if self._node_type == LayerType.OUTPUT:
            self._delta = (expected_value - self.value) * \
                          self._sigmoid_derivative(self.value)
        else:
            weightedsum = 0
            for item in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
                weightedsum += item.get_weight(self) * item.delta
                self._delta = \
                    weightedsum * self._sigmoid_derivative(self.value)

    def data_ready_downstream(self, node):
        """This method checks in the neurode, and if all the downstream
        neighbors are checked in proceeds to calculate their delta and
        update the weights.
        """
        if self._check_in(node, MultiLinkNode.Side.DOWNSTREAM) is True:
            self._calculate_delta()
            self._fire_upstream()
            self._update_weights()

    def set_expected(self, expected_value):
        """This method sets an expected value to be used to calculate
        delta on a neurode.
        """
        self._calculate_delta(expected_value)
        for item in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            item.data_ready_downstream(self)

    def adjust_weights(self, node, adjustment):
        """This method adjusts the weight of a neurode. """
        self._weights[node] += adjustment

    def _update_weights(self):
        """This method updates the weight of a neurode. """
        for item in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            adjustment = self.value * item.delta * item.learning_rate
            item.adjust_weights(self, adjustment)

    def _fire_upstream(self):
        """This method calls data_ready_downstream on upstream neighbors
        to let them know data is ready.
        """
        for item in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            item.data_ready_downstream(self)


class FFBPNeurode(FFNeurode, BPNeurode):
    """This class inherits from the FFNeurode and BPNeurode class. """
    pass


class DoublyLinkedList:
    """This class creates a doublylinked list of nodes."""

    def __init__(self):
        self._head = None
        self._tail = None
        self._curr = None

    class EmptyListError(Exception):
        """This exception catches empty lists."""
        pass

    def move_forward(self):
        """This method allows the client to move foward in the
        doublylinked list.
        """
        if self._head is None:
            raise DoublyLinkedList.EmptyListError
        if self._curr is None:
            return None
        elif self._curr is not None:
            if self._curr.next is None:
                raise IndexError
            else:
                self._curr = self._curr.next

    def move_back(self):
        """This method allows the client to move backward in the
        doublylinked list.
        """
        if self._head is None:
            raise DoublyLinkedList.EmptyListError
        if self._curr is None:
            return None
        elif self._curr is not None:
            if self._curr.prev is None:
                raise IndexError
            else:
                self._curr = self._curr.prev

    def reset_to_head(self):
        """
        This method returns the self._curr to the head of the list.
        """
        if self._head is None:
            raise DoublyLinkedList.EmptyListError
        self._curr = self._head
        if self._curr is None:
            return None
        else:
            return self._curr.data

    def reset_to_tail(self):
        """This method moves self._curr to the tail of the list. """
        if self._head is None:
            raise DoublyLinkedList.EmptyListError
        self._curr = self._tail
        if self._curr is None:
            return None
        else:
            return self._curr.data

    def add_to_head(self, data):
        """This method adds a node to the head of the list."""
        new_node = Node(data)
        new_node.next = self._head
        if self._head:
            self._head.prev = new_node
        self._head = new_node
        if self._tail is None:
            self._tail = new_node
        self.reset_to_head()

    def add_after_cur(self, data):
        """This method adds a node after self._curr. """
        if not self._curr:
            raise DoublyLinkedList.EmptyListError
        new_node = Node(data)
        new_node.prev = self._curr
        new_node.next = self._curr.next
        if self._curr.next:
            self._curr.next.prev = new_node
        self._curr.next = new_node
        if self._tail == self._curr:
            self._tail = new_node

    def remove_from_head(self):
        """This method removes a node from the head. """
        if self._head is None:
            raise DoublyLinkedList.EmptyListError
        ret_val = self._head.data
        self._head = self._head.next
        self._head.prev = None
        self.reset_to_head()
        return ret_val

    def remove_after_cur(self):
        """This method removes a node after self._curr. """
        if self._curr is None or self._curr.next is None:
            return None
        if self._head is None:
            raise DoublyLinkedList
        ret_val = self._curr.next.data
        if self._curr == self._tail:
            raise IndexError
        else:
            self._curr.next = self._curr.next.next
            if self._curr.next is not None:
                self._curr.next.prev = self._curr
            return ret_val

    def get_current_data(self):
        """This method returns the data from self._curr. """
        if self._head is None:
            raise DoublyLinkedList.EmptyListError
        else:
            return self._curr.data


class LayerList(DoublyLinkedList):
    """This class creates a layered list of neurodes. """

    def __init__(self, inputs: int, outputs: int, neurode_type: type(Neurode)):
        super().__init__()
        self._inputs = []
        self._outputs = []
        self._neurode_type = neurode_type
        for k in range(inputs):
            self._inputs.append(neurode_type(LayerType.INPUT))
        for k in range(outputs):
            self._outputs.append(neurode_type(LayerType.OUTPUT))
        self.add_to_head(self._outputs)
        self.add_to_head(self._inputs)
        self.helper_method()

    def helper_method(self):
        """This method connects the current node's neurodes with their
        downstream and upstream  neightbors.
        """
        curr_lc = [item for item in self._curr.data]
        curr_next_lc = [item for item in self._curr.next.data]
        for item in curr_lc:
            item.reset_neighbors(curr_next_lc, MultiLinkNode.Side.DOWNSTREAM)
        for item in curr_next_lc:
            item.reset_neighbors(curr_lc, MultiLinkNode.Side.UPSTREAM)

    def add_layer(self, num_nodes: int):
        """This method adds a hidden layer after the current node."""
        if self._curr is self._tail:
            raise IndexError
        num_nodes_list = []
        for k in range(num_nodes):
            num_nodes_list.append(self._neurode_type(LayerType.HIDDEN))
        self.add_after_cur(num_nodes_list)
        curr_lc = [item for item in self._curr.data]
        curr_next_lc = [item for item in self._curr.next.next.data]
        for item in curr_lc:
            item.reset_neighbors(num_nodes_list, MultiLinkNode.Side.DOWNSTREAM)
        for item in curr_next_lc:
            item.reset_neighbors(num_nodes_list, MultiLinkNode.Side.UPSTREAM)
        for item in num_nodes_list:
            item.reset_neighbors(curr_lc, MultiLinkNode.Side.UPSTREAM)
        for item in num_nodes_list:
            item.reset_neighbors(curr_next_lc, MultiLinkNode.Side.DOWNSTREAM)

    def remove_layer(self):
        """This method removes a hidden layer after the current node."""
        if self._curr.next is self._tail:
            raise IndexError
        self.remove_after_cur()
        self.helper_method()

    @property
    def input_nodes(self):
        """Returns self._inputs."""
        return self._inputs

    @property
    def output_nodes(self):
        """Returns self._outputs"""
        return self._outputs


class FFBPNetwork:
    """This class creates the FFBP network."""

    class EmptySetException(Exception):
        """This method creates an exception for an empty set. """
        pass

    def __init__(self, num_inputs: int, num_outputs: int):
        self.my_dataset = NNData()
        self.my_layers = LayerList(num_inputs, num_outputs, FFBPNeurode)
        self._inputs = num_inputs
        self._outputs = num_outputs

    def add_hidden_layer(self, num_nodes: int, position=0):
        """This method adds hidden layers to the network. """
        if position > 0:
            for k in range(position):
                self.my_layers.move_forward()
            self.my_layers.add_layer(num_nodes)
            self.my_layers.reset_to_head()
        else:
            self.my_layers.add_layer(num_nodes)

    def train(self, data_set: NNData, epochs=1000, verbosity=2,
              order=NNData.Order.RANDOM):
        """This method trains the network. """
        if data_set.pool_is_empty(NNData.Set.TRAIN):
            raise self.EmptySetException
        for epoch in range(epochs):
            data_set.prime_data(NNData.Set.TRAIN, order)
            errorbyepoch = []
            while not data_set.pool_is_empty(NNData.Set.TRAIN):
                item = data_set.get_one_item(NNData.Set.TRAIN)
                features = item[0]
                labels = item[1]
                node_value_orig = []
                for i, feature in enumerate(features):
                    self.my_layers.input_nodes[i].set_input(feature)
                for i, label in enumerate(labels):
                    node_value = self.my_layers.output_nodes[i].value
                    node_value_orig.append(node_value)
                    error = label - node_value
                    errorsquared = error ** 2
                    running_error = errorsquared
                    errorbyepoch.append(running_error)
                    self.my_layers.output_nodes[i].set_expected(label)
                if epoch % 1000 == 0 and verbosity > 1:
                    print(f"Sample {features} expected {labels} produced "
                          f"{node_value_orig}")
            rmse = (math.sqrt(sum(errorbyepoch) / len(errorbyepoch)))
            if epoch % 100 == 0 and verbosity > 0:
                print(f"{epoch} RMSE:{rmse}")
        print(f"Final EPOCH RMSE:{rmse}")

    def test(self, data_set: NNData, order=NNData.Order.SEQUENTIAL):
        """This method tests the network."""
        if data_set.pool_is_empty(NNData.Set.TEST):
            raise self.EmptySetException
        data_set.prime_data(NNData.Set.TEST, order)
        errorbytest = []
        while not data_set.pool_is_empty(NNData.Set.TEST):
            item = data_set.get_one_item(NNData.Set.TEST)
            features = item[0]
            labels = item[1]
            node_value_orig = []
            for i, feature in enumerate(features):
                self.my_layers.input_nodes[i].set_input(feature)
            for i, label in enumerate(labels):
                node_value = self.my_layers.output_nodes[i].value
                node_value_orig.append(node_value)
                error = label - node_value
                errorsquared = error ** 2
                running_error = errorsquared
                errorbytest.append(running_error)
                self.my_layers.output_nodes[i].set_expected(label)
            print(f"Sample {features} expected {labels} produced "
                  f"{node_value_orig}")
            rmse = (math.sqrt(sum(errorbytest) / len(errorbytest)))
        print(f"TEST RMSE:{rmse}")


class MutltiTypeEncoder(json.JSONEncoder):
    """This Class creates a json encoder for the NNData class."""
    def default(self, o):
        """This method takes objects not identified by json and puts
        them into a json acceptable form. """
        if isinstance(o, NNData):
            return {"__NNData__": o.__dict__}
        if isinstance(o, collections.deque):
            return {"__deque__": list(o)}
        if isinstance(o, numpy.ndarray):
            return {"__NDarray__": numpy.ndarray.tolist(o)}
        else:
            return json.JSONEncoder.default(self, o)


def multi_type_decoder(o):
    """This functions creates a JSON decoder for the NNData class. """
    if "__deque__" in o:
        return collections.deque(o["__deque__"])
    if "__NDarray__" in o:
        return list(o["__NDarray__"])
    if "__NNData__" in o:
        item = o["__NNData__"]
        return_obj = NNData(item["_features"], item["_labels"],
                            item["_train_factor"])
        return_obj._train_indices = item["_train_indices"]
        return_obj._test_indices = item["_test_indices"]
        return_obj._train_pool = item["_train_pool"]
        return_obj._test_pool = item["_test_pool"]
        return return_obj
    else:
        return o


def xor_encoded_decoded():
    """This functions loads the XOR dataset into a NNData object that is
    encoded into a JSON string and then decoded into a python object and
    trains it using the Neural Network."""
    XOR_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    XOR_Y = [[0], [1], [1], [0]]
    xor_data = NNData(XOR_X, XOR_Y, 1)
    json_str = json.dumps(xor_data, cls=MutltiTypeEncoder)
    decoded_data = json.loads(json_str, object_hook=multi_type_decoder)
    decoded_data.percentage_limiter(0.7)
    print("XOR DATASET:  ")
    print()
    print(decoded_data.__dict__)
    network = FFBPNetwork(2, 1)
    network.add_hidden_layer(3)
    network.train(decoded_data, 3001, order=NNData.Order.RANDOM)


def sin_decoded():
    """
    This function decodes Sine data into a python object and trains it
    using the Neural Network.
    """
    with open('sindecodedfile', 'r') as file:
        content = file.read()
    datadec = json.loads(content, object_hook=multi_type_decoder)
    print()
    print("SINE DATASET:  ")
    print()
    print(datadec.__dict__)
    network = FFBPNetwork(1, 1)
    network.add_hidden_layer(3)
    network.train(datadec, 3001, order=NNData.Order.RANDOM)


if __name__ == '__main__':
    xor_encoded_decoded()
    sin_decoded()

r"""
---Sample Run---
XOR DATASET:  

{'_train_factor': 1, '_features': array([[0., 0.],
       [0., 1.],
       [1., 0.],
       [1., 1.]]), '_labels': array([[0.],
       [1.],
       [1.],
       [0.]]), '_train_indices': [3, 2, 0, 1], '_test_indices': [], '_train_pool': deque([3, 2, 0, 1]), '_test_pool': deque([])}
Sample [1. 0.] expected [1.] produced [0.6619322133115649]
Sample [0. 0.] expected [0.] produced [0.6180383615026861]
Sample [1. 1.] expected [0.] produced [0.6811040903738702]
Sample [0. 1.] expected [1.] produced [0.6422368535689077]
0 RMSE:0.5215741787017535
100 RMSE:0.50210852918958
200 RMSE:0.5006245882619005
300 RMSE:0.5004933167619617
400 RMSE:0.5004448280524268
500 RMSE:0.5004007582197075
600 RMSE:0.5003524092474894
700 RMSE:0.5002976310348201
800 RMSE:0.500244781480909
900 RMSE:0.500182322439564
Sample [0. 0.] expected [0.] produced [0.4919871187551936]
Sample [1. 1.] expected [0.] produced [0.5137921077393643]
Sample [0. 1.] expected [1.] produced [0.499541165008061]
Sample [1. 0.] expected [1.] produced [0.5060607509460401]
1000 RMSE:0.5001171568427177
1100 RMSE:0.5000470097935158
1200 RMSE:0.49996934307674273
1300 RMSE:0.49988396444869454
1400 RMSE:0.4997920562660181
1500 RMSE:0.4996860097450087
1600 RMSE:0.49956398840438443
1700 RMSE:0.49943359652777214
1800 RMSE:0.4992796576264247
1900 RMSE:0.4991091867757412
Sample [1. 1.] expected [0.] produced [0.5216209233731839]
Sample [0. 1.] expected [1.] produced [0.504359402362306]
Sample [1. 0.] expected [1.] produced [0.5098864205161061]
Sample [0. 0.] expected [0.] produced [0.48756596071440694]
2000 RMSE:0.4989188001491879
2100 RMSE:0.49869912514594
2200 RMSE:0.49844743740727265
2300 RMSE:0.498161380158913
2400 RMSE:0.49783292685747926
2500 RMSE:0.49745770433063763
2600 RMSE:0.4970288130790172
2700 RMSE:0.49653908564085325
2800 RMSE:0.4959786952017232
2900 RMSE:0.4953380260589477
Sample [0. 0.] expected [0.] produced [0.47533937031300116]
Sample [0. 1.] expected [1.] produced [0.5159695631002745]
Sample [1. 1.] expected [0.] produced [0.5317863885394496]
Sample [1. 0.] expected [1.] produced [0.5146571533196138]
3000 RMSE:0.49461788854492283
Final EPOCH RMSE:0.49461788854492283

SINE DATASET:  

{'_train_factor': 0.2, '_features': array([[0.  ],
       [0.01],
       [0.02],
       [0.03],
       [0.04],
       [0.05],
       [0.06],
       [0.07],
       [0.08],
       [0.09],
       [0.1 ],
       [0.11],
       [0.12],
       [0.13],
       [0.14],
       [0.15],
       [0.16],
       [0.17],
       [0.18],
       [0.19],
       [0.2 ],
       [0.21],
       [0.22],
       [0.23],
       [0.24],
       [0.25],
       [0.26],
       [0.27],
       [0.28],
       [0.29],
       [0.3 ],
       [0.31],
       [0.32],
       [0.33],
       [0.34],
       [0.35],
       [0.36],
       [0.37],
       [0.38],
       [0.39],
       [0.4 ],
       [0.41],
       [0.42],
       [0.43],
       [0.44],
       [0.45],
       [0.46],
       [0.47],
       [0.48],
       [0.49],
       [0.5 ],
       [0.51],
       [0.52],
       [0.53],
       [0.54],
       [0.55],
       [0.56],
       [0.57],
       [0.58],
       [0.59],
       [0.6 ],
       [0.61],
       [0.62],
       [0.63],
       [0.64],
       [0.65],
       [0.66],
       [0.67],
       [0.68],
       [0.69],
       [0.7 ],
       [0.71],
       [0.72],
       [0.73],
       [0.74],
       [0.75],
       [0.76],
       [0.77],
       [0.78],
       [0.79],
       [0.8 ],
       [0.81],
       [0.82],
       [0.83],
       [0.84],
       [0.85],
       [0.86],
       [0.87],
       [0.88],
       [0.89],
       [0.9 ],
       [0.91],
       [0.92],
       [0.93],
       [0.94],
       [0.95],
       [0.96],
       [0.97],
       [0.98],
       [0.99],
       [1.  ],
       [1.01],
       [1.02],
       [1.03],
       [1.04],
       [1.05],
       [1.06],
       [1.07],
       [1.08],
       [1.09],
       [1.1 ],
       [1.11],
       [1.12],
       [1.13],
       [1.14],
       [1.15],
       [1.16],
       [1.17],
       [1.18],
       [1.19],
       [1.2 ],
       [1.21],
       [1.22],
       [1.23],
       [1.24],
       [1.25],
       [1.26],
       [1.27],
       [1.28],
       [1.29],
       [1.3 ],
       [1.31],
       [1.32],
       [1.33],
       [1.34],
       [1.35],
       [1.36],
       [1.37],
       [1.38],
       [1.39],
       [1.4 ],
       [1.41],
       [1.42],
       [1.43],
       [1.44],
       [1.45],
       [1.46],
       [1.47],
       [1.48],
       [1.49],
       [1.5 ],
       [1.51],
       [1.52],
       [1.53],
       [1.54],
       [1.55],
       [1.56],
       [1.57]]), '_labels': array([[0.        ],
       [0.00999983],
       [0.01999867],
       [0.0299955 ],
       [0.03998933],
       [0.04997917],
       [0.05996401],
       [0.06994285],
       [0.07991469],
       [0.08987855],
       [0.09983342],
       [0.1097783 ],
       [0.11971221],
       [0.12963414],
       [0.13954311],
       [0.14943813],
       [0.15931821],
       [0.16918235],
       [0.17902957],
       [0.18885889],
       [0.19866933],
       [0.2084599 ],
       [0.21822962],
       [0.22797752],
       [0.23770263],
       [0.24740396],
       [0.25708055],
       [0.26673144],
       [0.27635565],
       [0.28595223],
       [0.29552021],
       [0.30505864],
       [0.31456656],
       [0.32404303],
       [0.33348709],
       [0.34289781],
       [0.35227423],
       [0.36161543],
       [0.37092047],
       [0.38018842],
       [0.38941834],
       [0.39860933],
       [0.40776045],
       [0.4168708 ],
       [0.42593947],
       [0.43496553],
       [0.44394811],
       [0.45288629],
       [0.46177918],
       [0.47062589],
       [0.47942554],
       [0.48817725],
       [0.49688014],
       [0.50553334],
       [0.51413599],
       [0.52268723],
       [0.5311862 ],
       [0.53963205],
       [0.54802394],
       [0.55636102],
       [0.56464247],
       [0.57286746],
       [0.58103516],
       [0.58914476],
       [0.59719544],
       [0.60518641],
       [0.61311685],
       [0.62098599],
       [0.62879302],
       [0.63653718],
       [0.64421769],
       [0.65183377],
       [0.65938467],
       [0.66686964],
       [0.67428791],
       [0.68163876],
       [0.68892145],
       [0.69613524],
       [0.70327942],
       [0.71035327],
       [0.71735609],
       [0.72428717],
       [0.73114583],
       [0.73793137],
       [0.74464312],
       [0.75128041],
       [0.75784256],
       [0.76432894],
       [0.77073888],
       [0.77707175],
       [0.78332691],
       [0.78950374],
       [0.79560162],
       [0.80161994],
       [0.8075581 ],
       [0.8134155 ],
       [0.81919157],
       [0.82488571],
       [0.83049737],
       [0.83602598],
       [0.84147098],
       [0.84683184],
       [0.85210802],
       [0.85729899],
       [0.86240423],
       [0.86742323],
       [0.87235548],
       [0.8772005 ],
       [0.88195781],
       [0.88662691],
       [0.89120736],
       [0.89569869],
       [0.90010044],
       [0.90441219],
       [0.9086335 ],
       [0.91276394],
       [0.91680311],
       [0.9207506 ],
       [0.92460601],
       [0.92836897],
       [0.93203909],
       [0.935616  ],
       [0.93909936],
       [0.9424888 ],
       [0.945784  ],
       [0.94898462],
       [0.95209034],
       [0.95510086],
       [0.95801586],
       [0.96083506],
       [0.96355819],
       [0.96618495],
       [0.9687151 ],
       [0.97114838],
       [0.97348454],
       [0.97572336],
       [0.9778646 ],
       [0.97990806],
       [0.98185353],
       [0.98370081],
       [0.98544973],
       [0.9871001 ],
       [0.98865176],
       [0.99010456],
       [0.99145835],
       [0.99271299],
       [0.99386836],
       [0.99492435],
       [0.99588084],
       [0.99673775],
       [0.99749499],
       [0.99815247],
       [0.99871014],
       [0.99916795],
       [0.99952583],
       [0.99978376],
       [0.99994172],
       [0.99999968]]), '_train_indices': [1, 8, 15, 17, 21, 24, 34, 39, 41, 44, 47, 48, 49, 53, 54, 56, 61, 66, 69, 80, 82, 83, 87, 90, 97, 110, 120, 136, 145, 146, 148], '_test_indices': [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 18, 19, 20, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 40, 42, 43, 45, 46, 50, 51, 52, 55, 57, 58, 59, 60, 62, 63, 64, 65, 67, 68, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 81, 84, 85, 86, 88, 89, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 137, 138, 139, 140, 141, 142, 143, 144, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157], '_train_pool': deque([83, 146, 120, 34, 136, 49, 66, 24, 80, 148, 48, 47, 39, 53, 82, 69, 44, 61, 1, 15, 41, 17, 97, 90, 145, 110, 8, 87, 56, 54, 21]), '_test_pool': deque([7, 155, 127, 143, 141, 121, 62, 151, 156, 93, 63, 105, 144, 113, 100, 103, 154, 84, 94, 18, 123, 134, 92, 68, 67, 35, 116, 19, 98, 75, 76, 22, 43, 132, 33, 71, 65, 95, 89, 51, 124, 91, 23, 50, 101, 16, 46, 111, 58, 102, 138, 106, 78, 4, 13, 14, 119, 99, 125, 149, 152, 153, 27, 0, 85, 131, 5, 117, 157, 57, 142, 10, 88, 72, 42, 3, 37, 147, 40, 29, 81, 36, 30, 133, 79, 55, 77, 130, 20, 150, 25, 45, 12, 11, 129, 9, 2, 114, 59, 115, 96, 38, 139, 86, 32, 118, 107, 52, 122, 126, 108, 112, 70, 6, 60, 135, 64, 73, 31, 28, 128, 109, 137, 104, 26, 140, 74])}
Sample [0.48] expected [0.46177918] produced [0.7675491365357682]
Sample [0.97] expected [0.82488571] produced [0.7879426509711237]
Sample [0.53] expected [0.50553334] produced [0.7694228575189858]
Sample [1.2] expected [0.93203909] produced [0.7961171251944528]
Sample [0.9] expected [0.78332691] produced [0.7850521518965088]
Sample [0.82] expected [0.73114583] produced [0.7818001314268685]
Sample [0.39] expected [0.38018842] produced [0.7626753963592439]
Sample [1.36] expected [0.9778646] produced [0.8011157789663653]
Sample [0.15] expected [0.14943813] produced [0.7507644368157305]
Sample [0.49] expected [0.47062589] produced [0.7661479204963393]
Sample [0.41] expected [0.39860933] produced [0.761988325384723]
Sample [0.24] expected [0.23770263] produced [0.7533386804263791]
Sample [0.47] expected [0.45288629] produced [0.763426325807497]
Sample [0.34] expected [0.33348709] produced [0.7568899623951812]
Sample [0.69] expected [0.63653718] produced [0.7720137629118596]
Sample [0.83] expected [0.73793137] produced [0.7776259503383122]
Sample [0.01] expected [0.00999983] produced [0.7398882992188239]
Sample [1.45] expected [0.99271299] produced [0.7987891123996812]
Sample [0.56] expected [0.5311862] produced [0.765312343509488]
Sample [1.48] expected [0.99588084] produced [0.7996957895924174]
Sample [0.87] expected [0.76432894] produced [0.7783941939040718]
Sample [0.54] expected [0.51413599] produced [0.7643617483151072]
Sample [0.21] expected [0.2084599] produced [0.7485745391674129]
Sample [1.1] expected [0.89120736] produced [0.7858925494277023]
Sample [0.17] expected [0.16918235] produced [0.7460048707065058]
Sample [0.8] expected [0.71735609] produced [0.7735339297823269]
Sample [0.66] expected [0.61311685] produced [0.7675652933655347]
Sample [0.08] expected [0.07991469] produced [0.7404301623029464]
Sample [0.44] expected [0.42593947] produced [0.7565585342206976]
Sample [1.46] expected [0.99386836] produced [0.7953338258375197]
Sample [0.61] expected [0.57286746] produced [0.7639483829677777]
0 RMSE:0.33678216283693047
100 RMSE:0.2727548508428339
200 RMSE:0.2606941591175312
300 RMSE:0.22742587219289526
400 RMSE:0.1836758151336231
500 RMSE:0.1499583662167728
600 RMSE:0.12619392976348515
700 RMSE:0.10896714232403937
800 RMSE:0.09598513778709368
900 RMSE:0.08587718784618942
Sample [1.45] expected [0.99271299] produced [0.8490508114091273]
Sample [0.17] expected [0.16918235] produced [0.25310760182962744]
Sample [0.9] expected [0.78332691] produced [0.7375668842631549]
Sample [1.36] expected [0.9778646] produced [0.8393638534243417]
Sample [0.87] expected [0.76432894] produced [0.7267475678750136]
Sample [0.54] expected [0.51413599] produced [0.5435465640139671]
Sample [0.53] expected [0.50553334] produced [0.5361206656317793]
Sample [0.21] expected [0.2084599] produced [0.28247376098360133]
Sample [0.47] expected [0.45288629] produced [0.49029156582032224]
Sample [1.46] expected [0.99386836] produced [0.850175334412515]
Sample [0.01] expected [0.00999983] produced [0.1551214246069125]
Sample [0.41] expected [0.39860933] produced [0.44249632652777227]
Sample [0.66] expected [0.61311685] produced [0.6228116119853857]
Sample [1.48] expected [0.99588084] produced [0.8520411658923365]
Sample [0.97] expected [0.82488571] produced [0.7613530275634978]
Sample [0.82] expected [0.73114583] produced [0.7061843354605751]
Sample [0.49] expected [0.47062589] produced [0.5061871124983075]
Sample [0.48] expected [0.46177918] produced [0.49833618405861674]
Sample [0.34] expected [0.33348709] produced [0.3854788772181934]
Sample [0.69] expected [0.63653718] produced [0.640553465892881]
Sample [0.08] expected [0.07991469] produced [0.1937984909071593]
Sample [0.8] expected [0.71735609] produced [0.6967049104098703]
Sample [1.2] expected [0.93203909] produced [0.8156654118915855]
Sample [0.39] expected [0.38018842] produced [0.4263662461156456]
Sample [0.44] expected [0.42593947] produced [0.46656773427077153]
Sample [0.15] expected [0.14943813] produced [0.23890329050008782]
Sample [1.1] expected [0.89120736] produced [0.7954347145158198]
Sample [0.56] expected [0.5311862] produced [0.5572912873118734]
Sample [0.24] expected [0.23770263] produced [0.3049988447933376]
Sample [0.61] expected [0.57286746] produced [0.5910798341239639]
Sample [0.83] expected [0.73793137] produced [0.7097493880909711]
1000 RMSE:0.0778088930520648
1100 RMSE:0.07124732866039575
1200 RMSE:0.06583341388177145
1300 RMSE:0.061316124762241186
1400 RMSE:0.057511134763281964
1500 RMSE:0.05428179633205407
1600 RMSE:0.05151965359393352
1700 RMSE:0.049151385537882675
1800 RMSE:0.047104570106043915
1900 RMSE:0.0453283099610718
Sample [0.48] expected [0.46177918] produced [0.474461058587563]
Sample [0.24] expected [0.23770263] produced [0.24563658509374997]
Sample [0.08] expected [0.07991469] produced [0.13359925357526423]
Sample [0.21] expected [0.2084599] produced [0.22104561075567236]
Sample [0.15] expected [0.14943813] produced [0.17673134077479102]
Sample [0.41] expected [0.39860933] produced [0.4051637779015347]
Sample [1.45] expected [0.99271299] produced [0.8964770006318412]
Sample [1.1] expected [0.89120736] produced [0.8404050815029179]
Sample [0.9] expected [0.78332691] produced [0.7743883887775057]
Sample [0.01] expected [0.00999983] produced [0.09940300061559018]
Sample [0.39] expected [0.38018842] produced [0.38557481130296695]
Sample [0.44] expected [0.42593947] produced [0.43514961614788505]
Sample [0.49] expected [0.47062589] produced [0.4841068676867293]
Sample [0.17] expected [0.16918235] produced [0.190782177312306]
Sample [0.82] expected [0.73114583] produced [0.7359653874706356]
Sample [0.61] expected [0.57286746] produced [0.5928798226725567]
Sample [0.54] expected [0.51413599] produced [0.5311386147278668]
Sample [0.8] expected [0.71735609] produced [0.7249129570903596]
Sample [0.53] expected [0.50553334] produced [0.5217428758147769]
Sample [0.34] expected [0.33348709] produced [0.33623178777640256]
Sample [1.46] expected [0.99386836] produced [0.8973640160649079]
Sample [0.56] expected [0.5311862] produced [0.5493329326921458]
Sample [0.97] expected [0.82488571] produced [0.8014085719474836]
Sample [0.87] expected [0.76432894] produced [0.7606725029215519]
Sample [1.2] expected [0.93203909] produced [0.8619837636101655]
Sample [0.69] expected [0.63653718] produced [0.655157735301886]
Sample [0.83] expected [0.73793137] produced [0.741157424031298]
Sample [1.48] expected [0.99588084] produced [0.8993546599320944]
Sample [0.47] expected [0.45288629] produced [0.4646682302919565]
Sample [0.66] expected [0.61311685] produced [0.6329607757494807]
Sample [1.36] expected [0.9778646] produced [0.8866183808871271]
2000 RMSE:0.043779901492235415
2100 RMSE:0.042425445930589306
2200 RMSE:0.041235392613588134
2300 RMSE:0.04018556510359276
2400 RMSE:0.03925515189651427
2500 RMSE:0.038429940878188684
2600 RMSE:0.03769269693964635
2700 RMSE:0.03703292055425386
2800 RMSE:0.03644030269002384
2900 RMSE:0.03590570824935928
Sample [0.17] expected [0.16918235] produced [0.1750353236713801]
Sample [0.01] expected [0.00999983] produced [0.08776690476581019]
Sample [1.36] expected [0.9778646] produced [0.904232227235721]
Sample [1.48] expected [0.99588084] produced [0.9171129318575756]
Sample [0.24] expected [0.23770263] produced [0.22921803423171536]
Sample [0.69] expected [0.63653718] produced [0.659405674317437]
Sample [1.45] expected [0.99271299] produced [0.9142945228478052]
Sample [0.97] expected [0.82488571] produced [0.8162988560940532]
Sample [0.39] expected [0.38018842] produced [0.3720633476600976]
Sample [0.56] expected [0.5311862] produced [0.5458142029667594]
Sample [0.82] expected [0.73114583] produced [0.7464198223580063]
Sample [0.53] expected [0.50553334] produced [0.5162218578315232]
Sample [1.1] expected [0.89120736] produced [0.8568617581417807]
Sample [0.15] expected [0.14943813] produced [0.16146218398193782]
Sample [0.66] expected [0.61311685] produced [0.6354058444078179]
Sample [0.87] expected [0.76432894] produced [0.7728406016341175]
Sample [0.08] expected [0.07991469] produced [0.11994829904570985]
Sample [0.49] expected [0.47062589] produced [0.47552406053241003]
Sample [1.46] expected [0.99386836] produced [0.9152146919522626]
Sample [1.2] expected [0.93203909] produced [0.8792871867308525]
Sample [0.8] expected [0.71735609] produced [0.7347733473286937]
Sample [0.21] expected [0.2084599] produced [0.2048286412975813]
Sample [0.41] expected [0.39860933] produced [0.3926450925692042]
Sample [0.34] expected [0.33348709] produced [0.32136552086423376]
Sample [0.61] expected [0.57286746] produced [0.5923783098209717]
Sample [0.9] expected [0.78332691] produced [0.7871871686465798]
Sample [0.44] expected [0.42593947] produced [0.4238337821049011]
Sample [0.54] expected [0.51413599] produced [0.5260905977737986]
Sample [0.83] expected [0.73793137] produced [0.751863507348304]
Sample [0.48] expected [0.46177918] produced [0.4652479813190546]
Sample [0.47] expected [0.45288629] produced [0.45489127965927084]
3000 RMSE:0.03542151098203645
Final EPOCH RMSE:0.03542151098203645

Process finished with exit code 0
"""