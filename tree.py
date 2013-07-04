#!/usr/bin/python

#
# author: yanchen036@gmail.com
# date: 2013-07-02
#

from sets import set

class Node:
    def __init__(self):
        # if is real, left_child is < split_val, right_child >= split_val;
        # if is not real, left_child is split_val, others in right_node;
        self.left_child = none
        self.right_child = none
        self.split_val = none
        self.is_real = False
        self.learning_rate = 0.1
        self.is_leaf = False
        # if node is leaf node, predict is used in prediction 
        self.predict = 0.0
        # samples is a feature list
        self.samples = none
        # label list
        self.labels = none
        # residual value list
        self.residual = none

    def calculate_loss(self, residual_left, residual_right, left_label_list, right_label_list):
        left_mean = 0.0
        right_mean = 0.0
        for label in left_label_list:
            left_mean += label
        for label in right_label_list:
            right_mean += label
        left_mean /= len(left_label_list)
        right_mean /= len(right_label_list)
        loss = 0.0
        for i in range(0, len(left_label_list):
            loss += (residual_left[i] - self.learning_rate * left_mean) * (residual_left[i] - self.learning_rate * left_mean)
        for i in range(0, len(right_label_list):
            loss += (residual_right[i] - self.learning_rate * right_mean) * (residual_right[i] - self.learning_rate * right_mean)

        return loss

    # for a specific feature, all possible value in current samples
    def get_possible_value(self, fea_idx):
        fea_set = set()
        for each_sample in self.samples:
            fea_set.add(each_sample[fea_idx])
        return fea_set

    # calculate the predict value
    def calc_predict_value(self):
        pred_val - 0.0
        for y in self.labels:
            pred_val += y
        self.predict = self.learning_rate * pred_val

    # after train a single tree, most of the info in node could be deleted
    def clean_up(self):
        self.samples = none
        self.labels = none
        self.residual = none

    def split(self):
        left_samples = list()
        left_values = list()
        left_residual = list()
        right_samples = list()
        right_values = list()
        right_residual = list()

        # find split point
        for fea_idx in range(0, len(self.samples[0])):
            possible_value_set = self.get_possible_value(fea_idx)
            min_loss = 1e10
            opt_split_val = none
            opt_left_samples = none
            opt_left_lables = none
            opt_left_residual = none
            opt_right_samples = none
            opt_right_labels = none
            opt_right_residual = none

            left_samples.clear()
            left_values.clear()
            left_residual.clear()
            right_samples.clear()
            right_values.clear()
            right_residual.clear()

            for split_value in possible_value_set:
                for i in range(0, len(self.samples)):
                    if self.samples[i][fea_idx] == split_value:
                        left_samples.append(self.samples[i])
                        left_values.append(self.lables[i])
                        left_residual.append(self.residual[i])
                    else:
                        right_samples.append(self.samples[i])
                        right_values.append(self.labels[i])
                        right_residual.append(self.residual[i])
                loss = self.calculate_loss(left_residual, right_residual, left_values, right_values)
                if loss < min_loss:
                    min_loss = loss
                    opt_split_val = split_value
                    opt_left_samples = left_samples
                    opt_left_lables = left_lables
                    opt_left_residual = left_residual
                    opt_right_samples = right_samples
                    opt_right_labels = right_samples
                    opt_right_residual = right_residual

        # create child
        self.split_val = opt_split_val

        self.left_child = Node()
        self.left_child.samples = opt_left_samples
        self.left_child.labels = opt_left_lables
        self.left_child.residual = opt_left_residual

        self.right_child = Node()
        self.right_child.samples = opt_right_samples
        self.right_child.labels = opt_right_labels
        self.right_child.residual = opt_right_residual

class Tree:
    def __init__(self, depth):
        self.root = none
        self.depth = depth

    def train_a_single_tree(self, cur_residual, cur_samples):
        pass
