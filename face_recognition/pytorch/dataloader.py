import os
from torch.utils.data import Dataset
from PIL import Image

class lfwCustom(Dataset):
    def __init__(self, data_dir, transforms=None, train=True):
        self.train = train
        self.data_dir = data_dir
        self.img_list = os.listdir(data_dir)

        if transforms:
            self.transforms = transforms

        len_img = len(self.img_list)
        len_for_test = len_img // 10
        self.train_set = self.img_list[len_for_test:]
        self.test_set = self.img_list[:len_for_test]

        self.train_label_list = list(filename[:-9] for filename in self.train_set)
        self.test_label_list = list(filename[:-9] for filename in self.test_set)

        self.class_list = list(set(i[:-9] for i in self.img_list))

    def __len__(self):
        if self.train:
            return len(self.train_set)
        else:
            return len(self.test_set)

    def class_to_idx(self, class_name):
        return self.class_list.index(class_name)

    def __getitem__(self, idx):
        if self.train:
            return self.transforms(Image.open(os.path.join(self.data_dir, self.train_set[idx]))),\
                self.class_to_idx(self.train_label_list[idx])
        else:
            return self.transforms(Image.open(os.path.join(self.data_dir, self.test_set[idx]))), \
                self.class_to_idx(self.test_label_list[idx])