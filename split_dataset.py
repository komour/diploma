import numpy as np
from sklearn.model_selection import train_test_split
import os


def delete_idx_from_list(lst, idxes):
    cnt = 0
    for idx in idxes:
        lst.pop(idx - cnt)
        cnt += 1
    return lst


def filt(x, y, c):
    assert len(x) == len(y)
    x0 = []
    x1 = []
    y0 = []
    y1 = []
    pos_idxes = []
    for i in range(len(x)):
        if c[i] == 0:
            x0.append(x[i])
            y0.append(y[i])
        else:
            pos_idxes.append(i)
            x1.append(x[i])
            y1.append(y[i])
    assert len(x0) == len(y0)
    assert len(x1) == len(y1)
    assert len(x0) + len(x1) == len(x)
    assert len(y0) + len(y1) == len(y)

    return x0, x1, y0, y1, pos_idxes


# order of splitting - 5th class, 3rd class, 1st class, 2nd class, 4th class
# 4 2 0 1 3
# train - 1600 -- 0.61680802
# val - 400 -- 0.154202
# test - 594 -- 0.22898998
# first partition -- 0.61680802, 0.38319198
# second -- 0.40241449, 0.59758551
def main():
    first_partition = 0.61680802
    second_partition = 0.40241449
    f = open('images-onehot.txt', 'r')
    X = []
    Y = []
    lines = f.readlines()
    f.close()
    c1 = []
    c2 = []
    c3 = []
    c4 = []
    c5 = []
    positive = [0, 0, 0, 0, 0]
    negative = [0, 0, 0, 0, 0]
    for line in lines:
        name, label = line.split(' ')
        c1.append(int(label[0]))
        c2.append(int(label[1]))
        c3.append(int(label[2]))
        c4.append(int(label[3]))
        c5.append(int(label[4]))
        X.append(name)
        Y.append(label)
        for i in range(len(label[:-1])):
            if label[i] == '0':
                negative[i] += 1
            else:
                positive[i] += 1
    pos_weight = [-1, -1, -1, -1, -1]
    cl = [c1, c2, c3, c4, c5]
    for i in range(5):
        pos_weight[i] = negative[i] / positive[i]
    print(positive, negative)
    # print(len(list(filter(lambda x: x == 0, cl[4]))))
    # print(pos_weight)
    train_X = []
    train_Y = []
    val_X = []
    val_Y = []
    test_X = []
    test_Y = []

    for i in range(5):
        if i == 0:
            cur = 4
        elif i == 1:
            cur = 2
        elif i == 2:
            cur = 0
        elif i == 3:
            cur = 1
        else:
            cur = 3
        x0, x1, y0, y1, pos_idxes = filt(X, Y, cl[cur])
        for label in y1:
            assert label[cur] == '1'
        for label in y0:
            assert label[cur] == '0'
        for j in range(len(cl)):
            cl[j] = delete_idx_from_list(cl[j], pos_idxes)
        X = x0
        Y = y0
        xt, xp, yt, yp = train_test_split(x1, y1, train_size=0.61880802)
        xv, xte, yv, yte = train_test_split(xp, yp, train_size=0.40701449)

        train_X += xt
        train_Y += yt

        val_X += xv
        val_Y += yv

        test_X += xte
        test_Y += yte
        assert len(xt) + len(xv) + len(xte) == len(x1) == len(yt) + len(yv) + len(yte) == len(y1)

    xt, xp, yt, yp = train_test_split(X, Y, train_size=0.61680802)
    xv, xte, yv, yte = train_test_split(xp, yp, train_size=0.40241449)
    for label in yt:
        assert label == '00000\n' or label == '00000'
    assert len(xt) + len(xv) + len(xte) == len(X) == len(yt) + len(yv) + len(yte) == len(Y)

    train_X += xt
    train_Y += yt

    val_X += xv
    val_Y += yv

    test_X += xte
    test_Y += yte
    print(len(train_X) + len(val_X) + len(test_X), len(train_X), len(val_X), len(test_X))
    print(len(train_Y) + len(val_Y) + len(test_Y), len(train_Y), len(val_Y), len(test_Y))
    f_train = open('images_onehot_train.txt', 'w')
    f_val = open('images_onehot_val.txt', 'w')
    f_test = open('images_onehot_test.txt', 'w')
    for i in range(len(train_X)):
        f_train.write(train_X[i] + ' ' + train_Y[i])
    for i in range(len(val_X)):
        f_val.write(val_X[i] + ' ' + val_Y[i])
    for i in range(len(test_X)):
        f_test.write(test_X[i] + ' ' + test_Y[i])


if __name__ == "__main__":
    main()
