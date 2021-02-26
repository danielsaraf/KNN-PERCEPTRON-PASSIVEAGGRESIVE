import sys
import numpy as np
import random

K = 7
PERC_EPOCS = 20
ETA = 0.005
PA_EPOCS = 28
PER_MIN_VALUE = -2
PER_MAX_VALUE = 2
PA_MIN_VALUE = -0.1
PA_MAX_VALUE = 0.1


def closer_vec(closest_k, test_vec, dist):
    dist_arr = []
    for idx, vec in enumerate(closest_k):
        dist_arr.append(np.linalg.norm(np.array(closest_k[idx][1]) - np.array(test_vec)))
    if max(dist_arr) < dist:
        return False
    else:
        del closest_k[dist_arr.index(max(dist_arr))]
        return True


def get_knn_yhat(closest_k, train_y):
    values = []
    for vec in closest_k:
        values.append(train_y[vec[0]])
    return max(set(values), key=values.count)


def normalize(new_train_list, new_test_list):
    normalize_train = []
    normalize_test = []
    transpose_train = np.transpose(new_train_list)
    transpose_test = np.transpose(new_test_list)
    for idx, vec in enumerate(transpose_train):
        maxi = np.max(vec)
        mini = np.min(vec)
        if maxi - mini == 0:
            continue
        train_norm_vector = (vec - mini) / (maxi - mini)
        normalize_train.append(train_norm_vector)
        test_norm_vector = (transpose_test[idx] - mini) / (maxi - mini)
        normalize_test.append(test_norm_vector)
    normalize_train = np.transpose(normalize_train)
    normalize_test = np.transpose(normalize_test)
    return normalize_train, normalize_test


def fix_nominal(test):
    new_list = list()
    for vec in test:
        temp_list = list(vec.tolist())
        if b"W" in temp_list:
            temp_list.remove(b"W")
            temp_list.append(1)
            temp_list.append(0)
        elif b"R" in temp_list:
            temp_list.remove(b"R")
            temp_list.append(0)
            temp_list.append(1)
        new_list.append(temp_list)

    return new_list


def fix_nomial_and_normalize(train_x, test_x):
    new_train_list = fix_nominal(train_x)
    new_test_list = fix_nominal(test_x)
    return normalize(new_train_list, new_test_list)


def add_bais(list_train_x):
    new_list = []
    bias_arr = np.array([1])
    for vec in list_train_x:
        new_list.append(np.concatenate((vec, bias_arr.T), axis=0))
    return np.array(new_list)


def calc_tau(w_y, w_y_hat, x):
    loss = max(0, (1.0 - np.dot(w_y, x) + np.dot(w_y_hat, x)))
    norm = np.linalg.norm(x)
    if norm == 0:
        return 1
    return loss / (2 * (norm ** 2))


def main():
    type_sample = {'names': ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'), "formats": (
        np.float, np.float, np.float, np.float, np.float, np.float, np.float, np.float, np.float, np.float, np.float,
        '|S1')}
    train_x, train_y, test_x = sys.argv[1], sys.argv[2], sys.argv[3]
    test_x = np.loadtxt(test_x, delimiter=',', dtype=type_sample, skiprows=0)
    train_x = np.loadtxt(train_x, delimiter=',', dtype=type_sample, skiprows=0)
    train_y = np.loadtxt(train_y, dtype=int)
    if len(train_x) == 0:
        print("please enter some training set")
        exit(1)
    list_train_x, list_test_x = fix_nomial_and_normalize(train_x, test_x)
    zip_list = list(zip(list_train_x, train_y))
    random.shuffle(zip_list)
    list_train_x, train_y = zip(*zip_list)
    list_train_x = np.array(list_train_x)
    train_y = np.array(train_y)
    list_bais_train_x = add_bais(list_train_x)
    list_bais_test_x = add_bais(list_test_x)

    # knn
    knn_yhats = list()
    for test_vec in list_test_x:
        closest_k = []
        test_vec_clone = test_vec
        test_vec_clone = np.delete(test_vec_clone, 9, 0)
        test_vec_clone = np.delete(test_vec_clone, 0, 0)
        for idx, train_vec in enumerate(list_train_x):
            train_vec_clone = train_vec
            train_vec_clone = np.delete(train_vec_clone, 9, 0)
            train_vec_clone = np.delete(train_vec_clone, 0, 0)
            dist = np.linalg.norm(np.array(train_vec_clone) - np.array(test_vec_clone))
            if len(closest_k) < K or closer_vec(closest_k, test_vec_clone, dist):
                closest_k.append([idx, train_vec_clone])
        knn_yhats.append(get_knn_yhat(closest_k, train_y))

    # perceptron
    perceptron_yhats = list()
    w = np.random.uniform(PER_MIN_VALUE, PER_MAX_VALUE, (3, len(list_bais_train_x[0])))
    for i in range(PERC_EPOCS):
        for x, y in zip(list_bais_train_x, train_y):
            y_hat = np.argmax(np.dot(w, x))
            if y != y_hat:
                val = ETA * x
                w[y, :] += val
                w[y_hat, :] -= val
    for x in list_bais_test_x:
        v = np.argmax(np.dot(w, x))
        perceptron_yhats.append(np.argmax(np.dot(w, x)))

    # pa
    pa_yhats = list()
    w = np.random.uniform(PA_MIN_VALUE, PA_MAX_VALUE, (3, len(list_bais_train_x[0])))
    for i in range(PA_EPOCS):

        zip_list2 = list(zip(list_bais_train_x, train_y))
        random.shuffle(zip_list2)
        list_bais_train_x, train_y = zip(*zip_list2)
        list_bais_train_x = np.array(list_bais_train_x)
        train_y = np.array(train_y)

        for x, y in zip(list_bais_train_x, train_y):
            w_temp = np.delete(w, y, 0)
            y_hat = np.argmax(np.dot(w_temp, x))
            if y <= y_hat:
                y_hat = y_hat + 1
            tau = calc_tau(w[y, :], w[y_hat, :], x)
            val = tau * x
            w[y, :] += val
            w[y_hat, :] -= val

    for x in list_bais_test_x:
        pa_yhats.append(np.argmax(np.dot(w, x)))

    for i in range(len(test_x)):
        print(f"knn: {knn_yhats[i]}, perceptron: {perceptron_yhats[i]}, pa: {pa_yhats[i]}")


main()
