
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import numpy as np
import pickle
import math
def bagging_predict(trees, sample):
    predictions = [predict(tree, sample) for tree in trees]
    sumx = 0
    sumy = 0
    for t in predictions:
        sumx += t[0]
        sumy += t[1]
    sumx /= len(predictions)
    sumy /= len(predictions)
    return [sumx, sumy]

def calc_val(img, index_vector):
    thresh = img[index_vector[0]] - img[index_vector[1]]
    return thresh


def predict(node, sample):
    thresh = node['value']
    nums = node['index']
    samp = sample['hog']
    if calc_val(samp, nums) < thresh:
        if isinstance(node['left'], dict):
            return predict(node['left'], sample)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], sample)
        else:
            return node['right']


def to_terminal(group):
    outcomes = [[sample["centers"][1][0] - sample["centers"][0][0], sample["centers"][1][1] - sample["centers"][0][1]]
                for sample in group]
    sumx = 0
    sumy = 0
    for t in outcomes:
        sumx += t[0]
        sumy += t[1]
    sumx /= len(outcomes)
    sumy /= len(outcomes)
    return [sumx, sumy]


def split(node, max_depth, n_features, depth):
    # print(depth)
    left, right = node['groups']
    del (node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) == 1:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, n_features)
        split(node['left'], max_depth, n_features, depth + 1)
    if len(right) == 1:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, n_features, depth + 1)


def subsample(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


def random_forest(train, test, max_depth, sample_size, n_trees, n_features, loadModel):  # max_depth=200, n_trees=4, n_features=20
    trees = list()
    test = [train[test]]
    if loadModel:
        with open("forest.model", 'rb') as pickle_file:
            trees = pickle.load(pickle_file)
    else:
        for i in range(n_trees):
            print(f"Building tree {i}...")

            sample = subsample(train, sample_size)
            tree = build_tree(sample, max_depth, n_features)
            trees.append(tree)
    # predictions = [bagging_predict(trees, row) for row in test]
    return (trees)


# Build a decision tree
def build_tree(train, max_depth, n_features):
    root = get_split(train, n_features)
    split(root, max_depth, n_features, 1)
    return root


# Select the best split point for a dataset
def get_split(dataset, n_features):
    b_index, b_value, b_score, b_groups = 999, 999, 999999999999, None
    features = list()
    while len(features) < n_features:  # generate features
        dims = np.random.randint(96, size=2)
        feature = list(dims)
        if feature not in features:
            features.append(feature)
    threshs = np.random.random(size=20)
    # print('features ready')
    for f in features:
        counter = 0
        for thresh in threshs:
            groups = test_split(f, thresh, dataset)
            score = L_score(groups)
            if score < b_score:
                b_index, b_value, b_score, b_groups = f, thresh, score, groups
            counter+=1
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


def test_split(index, value, dataset):
    left, right = list(), list()
    val = value
    for row in dataset:
        if calc_val(row["hog"], index) < val:
            left.append(row)
        else:
            right.append(row)
    return left, right




def L_score(groups):
    total = []
    for branch in groups:
        dS_arr_x = []
        dS_arr_y = []
        for sample in branch:
            dS_arr_x.append(sample["centers"][1][0] - sample["centers"][0][0])
            dS_arr_y.append(sample["centers"][1][1] - sample["centers"][0][1])
        if len(dS_arr_x) == 0: x_div = 999
        else: x_div = len(dS_arr_x)
        if len(dS_arr_y) == 0: y_div = 999
        else: y_div = len(dS_arr_y)

        midx = sum(dS_arr_x) / x_div
        midy = sum(dS_arr_y) / y_div
        errx = 0
        for dS_x in dS_arr_x:
            errx += abs(dS_x - midx)
        erry = 0
        for dS_y in dS_arr_y:
            erry += abs(dS_y - midy)
        total.append(errx * errx + erry * erry)
    return sum(total)
print("Loadnig training_set.npy...")
sample = 4050
d_set = np.load("training_set2.npy", allow_pickle=True)
for i in range(11):
    print(f'Forest: {i}')
    sample = 0
    trees = random_forest(d_set, [sample], 200, 0.25, 4, 20, loadModel=False)
    pickle.dump(trees, open(f"forest.model{i}", "wb"))


# data = np.load("training_set2.npy", allow_pickle=True)
sample = d_set[sample]
# print(sample)
# print(sample['centers'])
# print("OK!")


def fitting(a,b,e_inter):
    init_center = [a,b]
    radius = 0.1 * e_inter
    angle_ranges = [[-45,45],[135,235]]
    # N =


def calc_boundary_coord(ang, radius):
    y = radius * math.sin(ang)
    x = radius * math.cos(ang)
    return x,y

predicted_delta = 0
for i in range(1,11):
    with open(f"forest.model{i}", 'rb') as pickle_file:
        trees = pickle.load(pickle_file)
    delta = np.array(bagging_predict(trees, sample))
    sample['centers'][1][1] -= 0.1 * delta[1]
    sample['centers'][1][0] -= 0.1 * delta[0]
    # print(
# print(predicted_delta)
print(sample['centers'])
