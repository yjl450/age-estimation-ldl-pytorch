import argparse
import better_exceptions
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from imgaug import augmenters as iaa
import torchvision.transforms as transforms
import math


def normal_sampling(mean, label_k, std=2):
    return math.exp(-(label_k-mean)**2/(2*std**2))/(math.sqrt(2*math.pi)*std)


class FaceDataset(Dataset):
    def __init__(self, data_dir, data_type, dataset, img_size=224, augment=False, age_stddev=1.0, label = False, gender = False, crop = True):
        assert(data_type in ("train", "valid", "test"))
        csv_path = Path(data_dir).joinpath(f"{dataset}_{data_type}_align.csv")
        img_dir = Path(data_dir)
        self.img_size = img_size
        self.label = label
        self.gen = gender
        self.augment = augment
        self.crop = crop
        self.age_stddev = age_stddev
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])


        self.x = []
        self.y = []
        # self.std = []
        self.rotate = []
        self.boxes = []
        df = pd.read_csv(str(csv_path))
        if self.gen:
            self.gender = []

        for _, row in df.iterrows():
            img_name = row["photo"]
            img_path = img_dir.joinpath(img_name)
            assert(img_path.is_file())
            self.x.append(str(img_path))
            self.y.append(row["age"])
            self.rotate.append(row["deg"])
            if self.gen:
                if row["gender"] == "M" or row["gender"] == "male":
                    self.gender.append(0)
                if row["gender"] == "F" or row["gender"] == "female":
                    self.gender.append(1)
            self.boxes.append(
                [row["box1"], row["box2"], row["box3"], row["box4"]])
            # self.std.append(row["apparent_age_std"])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img_path = self.x[idx]
        age = self.y[idx]

        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        # img.show()
        img = img.rotate(
            self.rotate[idx], resample=Image.BICUBIC, expand=True)  # Alignment
        # size = img.size        
        if self.crop:
            img = img.crop(self.boxes[idx])
        # img.show()
        img = self.transform(img)
        # print(img.shape)
        if self.label:
            label = [normal_sampling(int(age), i) for i in range(101)]
            label = [i if i > 1e-15 else 1e-15 for i in label]
            label = torch.Tensor(label)
            if self.gen:
                return img, int(age), label, self.gender[idx]
            return img, int(age), label
        else:
            if self.gen:
                return img, int(age), self.gender[idx]
            return img, int(age)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    print(args)
    dataset = FaceDataset(args.data_dir, "train", args.dataset)
    print("train dataset len: {}".format(len(dataset)))
    dataset = FaceDataset(args.data_dir, "valid", args.dataset)
    print("valid dataset len: {}".format(len(dataset)))

if __name__ == '__main__':
    main()
