import os


def main():
    root_dir = 'data/'
    train_labels = os.path.join(root_dir, 'train', 'images_onehot_train_milia.txt')
    val_labels = os.path.join(root_dir, 'val', 'images_onehot_val.txt')

    f_train = open(train_labels, 'r')
    lines = f_train.readlines()
    f_train.close()
    positive = [0, 0, 0, 0, 0, 0, 0]
    negative = [0, 0, 0, 0, 0, 0, 0]
    for line in lines:
        _, label = line.split(' ')
        for i in range(len(label[:-1])):
            if label[i] == '0':
                negative[i] += 1
            else:
                positive[i] += 1
    pos_weight = [-1, -1, -1, -1, -1, -1, -1]
    for i in range(7):
        pos_weight[i] = (negative[i] / positive[i]) if positive[i] != 0 else 0
    print(positive, negative)
    print(pos_weight)


if __name__ == "__main__":
    main()
