#!/usr/bin/python

#
# author: yanchen036@gmail.com
# date: 2013-07-02
#

from sets import set

class Node:
    def __init__(self, tree):
        # if is real, left_child is < split_val, right_child >= split_val;
        # if is not real, left_child is split_val, others in right_node;
        self.left_child = None
        self.right_child = None
        self.split_val = None
        self.depth = 0

        # a list, which element is index of sample list
        self.sample_indices = None
        # a list, which elemnt is index of label list
        self.label_indices = None
        # residual value list
        self.residual = None

        # leaf node attribute 
        self.is_leaf = False
        self.predict_value = 0.0

        # tree the node belonged
        self.tree = tree

    def calculate_loss(self, residual_left, residual_right, left_label_indices, right_label_indices):
        left_mean = 0.0
        right_mean = 0.0
        for idx in left_label_indices:
            left_mean += self.tree.forest.labels[idx]
        for idx in right_label_indices:
            right_mean += self.tree.forest.labels[idx]
        left_mean /= len(left_label_indices)
        right_mean /= len(right_label_indices)
        loss = 0.0
        for i in range(0, len(left_label_indices):
            loss += (residual_left[i] - self.tree.forest.learning_rate * left_mean) * (residual_left[i] - self.tree.forest.learning_rate * left_mean)
        for i in range(0, len(right_label_indices):
            loss += (residual_right[i] - self.tree.forest.learning_rate * right_mean) * (residual_right[i] - self.tree.forest.learning_rate * right_mean)

        return loss

    # for a specific feature, all possible value in current samples
    def get_possible_value(self, fea_idx):
        fea_set = set()
        for idx in self.sample_indices:
            fea_set.add(self.tree.forest.samples[idx][fea_idx])
        return fea_set

    # calculate the predict value
    def calc_predict_value(self):
        pred_val = 0.0
        for idx in self.label_indices:
            pred_val += self.tree.forest.labels[idx]
        self.predict_value = self.learning_rate * pred_val

    # after train a single tree, most of the info in node could be deleted
    def clean_up(self):
        self.sample_indices = None
        self.label_indices = None
        self.residual = None

    def split(self):
        left_sample_indices = list()
        left_label_indices = list()
        left_residual = list()
        right_sample_indices = list()
        right_label_indices = list()
        right_residual = list()

        # find split point
        for fea_idx in range(0, len(self.tree.forest.samples[0])):
            possible_value_set = self.get_possible_value(fea_idx)
            min_loss = 1e10
            opt_split_val = None
            opt_left_sample_indices = None
            opt_left_label_indices = None
            opt_left_residual = None
            opt_right_sample_indices = None
            opt_right_label_indices = None
            opt_right_residual = None

            left_sample_indices.clear()
            left_label_indices.clear()
            left_residual.clear()
            right_sample_indices.clear()
            right_label_indices.clear()
            right_residual.clear()

            for split_value in possible_value_set:
                for i in range(0, len(self.sample_indices)):
                    sample_idx = self.sample_indices[i]
                    if self.tree.forest.samples[sample_idx][fea_idx] == split_value:
                        left_sample_indices.append(self.sample_indices[i])
                        left_label_indices.append(self.label_indices[i])
                        left_residual.append(self.residual[i])
                    else:
                        right_sample_indices.append(self.sample_indices[i])
                        right_label_indices.append(self.label_indices[i])
                        right_residual.append(self.residual[i])
                loss = self.calculate_loss(left_residual, right_residual, left_label_indices, right_label_indices)
                if loss < min_loss:
                    min_loss = loss
                    opt_split_val = split_value
                    opt_left_sample_indices = left_sample_indices
                    opt_left_label_indices = left_label_indices
                    opt_left_residual = left_residual
                    opt_right_sample_indices = right_sample_indices
                    opt_right_label_indices = right_label_indices
                    opt_right_residual = right_residual

        # create child
        self.split_val = opt_split_val

        self.left_child = Node(self.tree)
        self.left_child.sample_indices = opt_left_sample_indices
        self.left_child.label_indices = opt_left_label_indices
        self.left_child.residual = opt_left_residual
        self.left_child.depth = self.depth + 1

        self.right_child = Node(self.learning_rate)
        self.right_child.sample_indices = opt_right_sample_indices
        self.right_child.label_indices = opt_right_label_indices
        self.right_child.residual = opt_right_residual
        self.right_child.depth = self.depth + 1

class Tree:
    def __init__(self, forest):
        # forest which this tree belonged
        self.forest = None
        self.root = None
        self.leaf_nodes = list()

    def train_a_single_tree(self, residual):
        self.root = Node(self)
        self.root.sample_indices = range(0, len(self.forest.samples) - 1)
        self.root.label_indices = range(0, len(self.forest.labels) - 1)
        self.root.residual = residual
        self.root.depth = 1

        # dfs
        node_stack = list()
        node_stack.append(self.root)
        while len(node_stack) > 0:
            cur_node = node_stack.pop(len(node_stack) - 1)
            if (cur_node.depth >= self.forest.depth_restrict):
                cur_node.is_leaf = True
                cur_node.calc_predict_value()
                self.leaf_nodes.append(cur_node)
            else:
                cur_node.split()
                cur_node.clean_up()
                node_stack.append(cur_node.right_child)
                node_stack.append(cur_node.left_child)

    # contribution of current tree
    def additive_score(self):
        score = list([0.0] * len(self.forest.samples))
        for node in self.leaf_nodes:
            for idx in node.sample_indices:
                score[idx] += node.predict_value
        return score

class Forest:
    def __init__(self, tree_num, restrict_depth, learning_rate, is_real):
        self.tree_num = tree_num
        self.restrict_depth = restrict_depth
        self.learning_rate = learning_rate
        self.is_real = is_real
        self.residual = None

        # all trainging set, each element is a feature list
        self.samples = list()
        self.labels = list()
