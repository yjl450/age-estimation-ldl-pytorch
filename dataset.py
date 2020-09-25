import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import math
import albumentations as A
from albumentations.pytorch import ToTensorV2


def normal_sampling(mean, label_k, std=2):
    return math.exp(-(label_k-mean)**2/(2*std**2))/(math.sqrt(2*math.pi)*std)

def expand_bbox(size, bbox_list, ratio = 0.2):
    width = bbox_list[2] - bbox_list[0]
    height = bbox_list[3] - bbox_list[1]
    new_bbox = []
    expand = [-(width*ratio)/2, -(height*ratio)/2, (width*ratio)/2, (height*ratio)/2]
    for ind, coor in enumerate(bbox_list):
        to_append = coor + expand[ind]
        to_append = 0 if to_append < 0 else to_append
        to_append = size[ind % 2] if to_append > size[ind % 2] else to_append
        new_bbox.append(int(to_append))
    return new_bbox



class FaceDataset(Dataset):
    def __init__(self, data_dir, data_type, dataset, img_size=224, augment=False, age_stddev=1.0, label=False, gender=False, expand=0.0):
        assert(data_type in ("train", "valid", "test"))
        csv_path = Path(data_dir).joinpath(f"{dataset}_{data_type}_align.csv")
        img_dir = Path(data_dir)
        self.img_size = img_size
        self.label = label
        self.gen = gender
        self.race_dic = {"A":0, "B": 1, "H": 2, "O": 4, "W": 4}
        self.augment = augment
        self.expand = expand
        self.age_stddev = age_stddev
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
        # self.transform = A.Compose([
        #     A.Resize(img_size, img_size),
        #     A.Normalize(
        #         mean=[0.485, 0.456, 0.406],
        #         std=[0.229, 0.224, 0.225],
        #     ),
        #     ToTensorV2()
        # ])
        self.transform_aug = A.Compose([
            A.HorizontalFlip(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.RandomGamma(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.Resize(img_size, img_size, always_apply=True),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                always_apply=True
            ),
            ToTensorV2()
        ])

        self.x = []
        self.y = []
        # self.std = []
        self.rotate = []
        self.boxes = []
        self.race = []
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
                if row["race"] is not None:
                    self.race.append(self.race_dic[row["race"]])
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
        img = img.rotate(
            self.rotate[idx], resample=Image.BICUBIC, expand=True)  # Alignment
        # size = img.size
        if self.expand > 0:
            img = img.crop(expand_bbox(img.size, self.boxes[idx], ratio= self.expand))
        else:
            img = img.crop(self.boxes[idx])
        # img = self.transform(img)

        image_np = np.array(img)
        if self.augment:
            augmented = self.transform_aug(image = image_np)
            img = augmented["image"]
        else:
            img = self.transform(img)
        

        # if self.gen:
            # gen_vec = torch.zeros(2, dtype=torch.long)
            # gen_vec[self.gender[idx]] = 1
            # if len(self.race) > 0:
            #     race_vec = torch.zeros(5)
            #     race_vec[self.race[idx]] = 1

        if self.label:
            label = [normal_sampling(int(age), i) for i in range(101)]
            label = [i if i > 1e-15 else 1e-15 for i in label]
            label = torch.Tensor(label)
            if self.gen:
                return img, int(age), label, self.gender[idx], self.race[idx]
            return img, int(age), label
        else:
            if self.gen:
                return img, int(age), self.gender[idx], self.race[idx]
            return img, int(age)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    print(args)
    # dataset = FaceDataset(args.data_dir, "train", args.dataset)
    # print("train dataset len: {}".format(len(dataset)))
    dataset = FaceDataset(args.data_dir, "valid", args.dataset, augment=True)
    print("valid dataset len: {}".format(len(dataset)))
    print(dataset[0])


if __name__ == '__main__':
    main()