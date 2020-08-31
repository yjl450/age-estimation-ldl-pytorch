# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
from facenet_pytorch import MTCNN
from PIL import Image
import math
import pandas as pd
import csv
from PIL import ImageDraw
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm


# %%
mtcnn = MTCNN(device="cuda")


# %%
def angel(t1, t2):
    x = t2[0]-t1[0]
    y = (t2[1]-t1[1])
    return math.degrees(math.atan2(y, x))


# %%
path = ""
mode = ["train", "test"]
for i in mode:
    f = open(path + "megaage_asian_{}_align.csv".format(i), "w", newline='')
    writer = csv.writer(f)

    with open(path + 'megaage_asian_{}.csv'.format(i),'r') as csvfile:
        header = next(csvfile).split(",")
        header[-1] = "box4"
        writer.writerow(header)
        reader = csv.reader(csvfile)
        # pbar = tqdm(total = 55134)
        for row in tqdm(reader):
            # pbar.update(1)
            img = Image.open(path + row[0])

            # display(img)

            boxes, probs, landmarks = mtcnn.detect(img, landmarks=True) # Get eye position for alignment
            deg = angel(landmarks[0][0], landmarks[0][1])
            img = img.rotate(deg, resample=Image.BICUBIC, expand=True)
            boxes, probs, landmarks = mtcnn.detect(img, landmarks=True) # Get bounding box
            # img1 = ImageDraw.Draw(img)
            # img1.rectangle(boxes, outline="red", width=2)
            # img1.point(landmarks[0][0], fill="red") #left eye
            # img1.point(landmarks[0][1], fill="green") #right eye
            # img1.point(landmarks[0][2], fill="blue")
            # img1.point(landmarks[0][3], fill="yellow")
            # img1.point(landmarks[0][4], fill="orange")
            # display(img)
            writer.writerow(row+[str(deg)]+list(boxes[0]))

            # break
    # pbar.close
    f.close()
print("done")


# %%


