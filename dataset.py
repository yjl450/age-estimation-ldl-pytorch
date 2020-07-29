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


class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.OneOf([
                iaa.Sometimes(
                    0.25, iaa.AdditiveGaussianNoise(scale=0.1 * 255)),
                iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0)))
            ]),
            iaa.Affine(
                rotate=(-20, 20), mode="edge",
                scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}
            ),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
            iaa.GammaContrast((0.3, 2)),
            iaa.Fliplr(0.5),
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.aug.augment_image(img)
        return img


class FaceDataset(Dataset):
    def __init__(self, data_dir, data_type, dataset, img_size=224, augment=False, age_stddev=1.0):
        assert(data_type in ("train", "valid", "test"))
        csv_path = Path(data_dir).joinpath(f"{dataset}_{data_type}_align.csv")
        img_dir = Path(data_dir)
        self.img_size = img_size
        self.augment = augment
        self.age_stddev = age_stddev
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

        # if augment:
        #     self.transform = ImgAugTransform()
        # else:
        #     self.transform = lambda i: i

        self.x = []
        self.y = []
        # self.std = []
        self.rotate = []
        self.boxes = []
        df = pd.read_csv(str(csv_path))

        for _, row in df.iterrows():
            img_name = row["photo"]
            img_path = img_dir.joinpath(img_name)
            assert(img_path.is_file())
            self.x.append(str(img_path))
            self.y.append(row["age"])
            self.rotate.append(row["deg"])
            self.boxes.append(
                [row["box1"], row["box2"], row["box3"], row["box4"]])
            # self.std.append(row["apparent_age_std"])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img_path = self.x[idx]
        age = self.y[idx]

        # if self.augment:
        #     age += np.random.randn() * self.std[idx] * self.age_stddev
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        # img.show()
        img = img.rotate(
            self.rotate[idx], resample=Image.BICUBIC, expand=True)  # Alignment
        img = img.crop(self.boxes[idx])
        # img.show()
        img = self.transform(img)
        # print(img.shape)
        return img, age #np.clip(round(age), 0, 100)


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
