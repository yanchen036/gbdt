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
        # samples is a feature list
        self.samples = none
        # label list
        self.labels = none

    def calculate_loss(self, cur_residual, left_label_list, right_label_list):
        left_mean = 0.0
        right_mean = 0.0
        for label in left_label_list:
            left_mean += label
        for label in right_label_list:
            right_mean += label
        left_mean /= len(left_label_list)
        right_mean /= len(right_label_list)
        d

    # for a specific feature, all possible value in current samples
    def get_possible_value(self, fea_idx):
        fea_set = set()
        for each_sample in self.samples:
            fea_set.add(each_sample[fea_idx])
        return fea_set

    def split(self, cur_residual):
        left_sample_list = list()
        left_value_list = list()
        right_sample_list = list()
        right_value_list = list()
        for fea_idx in range(0, len(self.samples[0])):
            possible_value_set = self.get_possible_value(fea_idx)
            min_loss = 1e10
            split_fea_val = none
            left_sample_list.clear()
            left_value_list.clear()
            right_sample_list.clear()
            right_sample_list.clear()

            for split_value in possible_value_set:
                for i in range(0, len(self.samples)):
                    if self.samples[i][fea_idx] == split_value:
                        left_sample_list.append(self.samples[i])
                        left_value_list.append(self.lables[i])
                    else:
                        right_sample_list.append(self.samples[i])
                        right_value_list.append(self.labels[i])
                loss = self.calculate_loss(cur_residual, left_value_list, right_value_list)

class Tree:
    def __init__(self, depth):
        self.root = none
        self.depth = depth

    def train_a_single_tree(self, cur_residual, cur_samples):
        pass
