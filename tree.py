#!/usr/bin/python

#
# author: yanchen036@gmail.com
# date: 2013-07-02
#

class Node:
    def __init__(self):
        # if is real, left_child is < split_val, right_child >= split_val;
        # if is not real, left_child is split_val, others in right_node;
        self.left_child = none
        self.right_child = none
        self.split_val = none
        self.is_real = False
        self.samples = none

    def split(self, cur_residual, cur_samples):
        pass

class Tree:
    def __init__(self, depth):
        self.root = none
        self.depth = depth

    def train_a_single_tree(self, cur_residual, cur_samples):
        pass
