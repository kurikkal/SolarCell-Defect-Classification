from torch.utils.data import Dataset
import torch
from skimage.io import imread
from skimage.color import gray2rgb
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class SolarDataset(Dataset):

    def __init__(self,data,mode):
        super().__init__()
        self.data = data
        self.mode =mode
        if self.mode == 'train':
            self.transforms = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.RandomVerticalFlip(),
                tv.transforms.RandomRotation(degrees=30),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])
        else:
            self.transforms = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data.iloc[index]['filename']
        is_cracked = torch.tensor(self.data.iloc[index]['crack'], dtype=torch.float32)
        is_inactive = torch.tensor(self.data.iloc[index]['inactive'], dtype=torch.float32)

        img = imread(img_path)
        img = gray2rgb(img)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, torch.stack([is_cracked, is_inactive])


