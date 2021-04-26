import random


def main():
    amount_of_images = 100
    train_file = 'data/train/images_onehot_train.txt'
    train_vis_file = 'vis_train.txt'
    val_file = 'data/val/images_onehot_val.txt'
    val_vis_file = 'vis_val.txt'
    f = open(train_file, 'r')
    f2 = open(train_vis_file, 'w')
    lines = f.readlines()
    chosen_images = random.sample(lines, amount_of_images)
    for line in chosen_images:
        f2.write(f'{list(line.split())[0]}\n')


if __name__ == "__main__":
    main()
