from custom_dataset import DatasetISIC2018

if __name__ == '__main__':
    label_file = 'images-onehot.txt'
    root_dir = 'images/256ISIC2018_Task1-2_Training_Input'
    dataset = DatasetISIC2018(label_file, root_dir)
    for i in range(len(dataset)):
        sample = dataset[i]
        print(sample['label'], sample['image'].shape, end='\n\n')
