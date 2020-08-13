import argparse
import better_exceptions
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pretrainedmodels
import pretrainedmodels.utils
from model import get_model
from dataset import FaceDataset, expand_bbox, normal_sampling
from defaults import _C as cfg
from datetime import datetime
from matplotlib import pyplot as plt
import loss as L
import pandas as pd
from PIL import Image

def get_group(age):
    if 0 <= age <= 5:
        return 0
    if 6 <= age <= 10:
        return 1
    if 11 <= age <= 20:
        return 2
    if 21 <= age <= 30:
        return 3
    if 31 <= age <= 40:
        return 4
    if 41 <= age <= 60:
        return 5
    if 61 <= age:
        return 6

class FaceVal(FaceDataset):
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
        else:
            augmented = self.transform(image = image_np)
        img = augmented["image"]
        # if torch.isnan(img).any() or torch.isinf(img).any():
        #     print(img_path[idx])
        if self.label:
            label = [normal_sampling(int(age), i) for i in range(101)]
            label = [i if i > 1e-15 else 1e-15 for i in label]
            label = torch.Tensor(label)
            if self.gen:
                return img, int(age), label, self.gender[idx], img_path
            return img, int(age), label, img_path
        else:
            if self.gen:
                return img, int(age), self.gender[idx], img_path
            return img, int(age), img_path


def get_args():
    model_names = sorted(name for name in pretrainedmodels.__dict__
                         if not name.startswith("__")
                         and name.islower()
                         and callable(pretrainedmodels.__dict__[name]))
    parser = argparse.ArgumentParser(description=f"available models: {model_names}",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str,
                        required=True, help="Data root directory")
    parser.add_argument("--dataset", type=str,
                        required=True, help="Dataset name")
    parser.add_argument("--resume", type=str, required=True, default=None,
                        help="Resume from checkpoint if any")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoint", help="Checkpoint directory")
    parser.add_argument("--tensorboard", type=str,
                        default=None, help="Tensorboard log directory")
    parser.add_argument('--multi_gpu', action="store_true",
                        help="Use multi GPUs (data parallel)")
    parser.add_argument('--ldl', action="store_true",
                        help="Use KLDivLoss + L1 Loss")
    parser.add_argument('--expand', type=float, default=0, help="expand the crop area [0, 1)")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    return args


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def validate(validate_loader, model, criterion, epoch, device, group_count, gender_count="False"):
    model.eval()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()
    preds = []
    gt = []
    group_mae = torch.zeros(7)
    gender_mae = torch.zeros(2)
    to_count = False
    error = []
    if sum(group_count) == 0:
        to_count = True
    with torch.no_grad():
        with tqdm(validate_loader) as _tqdm:
            for i, pack in enumerate(_tqdm):
                x = pack[0]
                y = pack[1]
                path = pack[-1]
                if gender_count != "False":
                    gender = pack[2]
                if to_count:
                    for ind, p in enumerate(y):
                        group_count[get_group(p.item())] += 1
                        if gender_count != "False":
                            if gender[ind]:
                                gender_count[1] += 1
                            else:
                                gender_count[0] += 1                        
                x = x.to(device)
                y = y.to(device)

                # compute output
                outputs = model(x)
                pred_ages = F.softmax(outputs, dim=-1).cpu().numpy()
                preds.append(pred_ages)

                for ind, age in enumerate(pred_ages):
                    group_mae[get_group(y[ind].item())] += abs(y[ind] - age)
                    if gender_count != "False":
                        gender_mae[gender[ind]] += abs(y[ind] - age)
                    if abs(y[ind] - age) > 3:
                        error.append([path[ind], y[ind], age, abs(y[ind] - age)])

                gt.append(y.cpu().numpy())

                # valid for validation, not used for test
                if criterion is not None:
                    # calc loss
                    loss = criterion(outputs, y)
                    cur_loss = loss.item()

                    # calc accuracy
                    _, predicted = outputs.max(1)
                    correct_num = predicted.eq(y).sum().item()

                    # measure accuracy and record loss
                    sample_num = x.size(0)
                    loss_monitor.update(cur_loss, sample_num)
                    accuracy_monitor.update(correct_num, sample_num)
                    _tqdm.set_postfix(OrderedDict(stage="val", epoch=epoch, loss=loss_monitor.avg),
                                      acc=accuracy_monitor.avg, correct=correct_num, sample_num=sample_num)

    preds = np.concatenate(preds, axis=0)
    gt = np.concatenate(gt, axis=0)
    ages = np.arange(0, 101)
    ave_preds = (preds * ages).sum(axis=-1)
    diff = ave_preds - gt
    mae = np.abs(diff).mean()

    df = pd.DataFrame(error, columns = ["photo", "age", "pred", "error"])

    if gender_count != "False":
        return loss_monitor.avg, accuracy_monitor.avg, mae, (group_mae, gender_mae), df
    else:
        return loss_monitor.avg, accuracy_monitor.avg, mae, (group_mae,), df


def validate_ldl(validate_loader, model, criterion, epoch, device, group_count, gender_count="False"):
    model.eval()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()
    preds = []
    gt = []
    rank = torch.Tensor([i for i in range(101)]).to(device)
    group_mae = torch.zeros(7)
    gender_mae = torch.zeros(2)
    to_count = False
    error = []
    if sum(group_count) == 0:
        to_count = True
    with torch.no_grad():
        with tqdm(validate_loader) as _tqdm:
            for i, pack in enumerate(_tqdm): #(x, y, lbl)
                x = pack[0]
                y = pack[1]
                lbl = pack[2]
                path = pack[-1]
                if gender_count != "False":
                    gender = pack[3]
                if to_count:
                    for ind, p in enumerate(y):
                        group_count[get_group(p.item())] += 1
                        if gender_count != "False":
                            if gender[ind]:
                                gender_count[1] += 1
                            else:
                                gender_count[0] += 1  
                x = x.to(device)
                y = y.to(device)
                lbl = lbl.to(device)

                # compute output
                outputs = model(x)
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(outputs)
                outputs = F.softmax(outputs, dim = 1)
                ages = torch.sum(outputs*rank, dim=1)  # age expectation
                preds.append(ages.cpu().numpy())  # append predicted age
                gt.append(y.cpu().numpy())  # append real age

                for ind, age in enumerate(ages):
                    group_mae[get_group(y[ind].item())] += abs(y[ind] - age)
                    if gender_count != "False":
                        gender_mae[gender[ind]] += abs(y[ind] - age) 
                    if abs(y[ind] - age) > 3:
                        error.append([path[ind], y[ind], age, abs(y[ind] - age)])

                # valid for validation, not used for test
                if criterion is not None:
                    # calc loss
                    loss1 = L.kl_loss(outputs, lbl)
                    loss2 = L.L1_loss(ages, y)
                    loss = loss1 + loss2
                    cur_loss = loss.item()

                    # calc accuracy
                    # _, predicted = outputs.max(1)
                    # correct_num = predicted.eq(y).sum().item()
                    correct_num = (abs(ages - y) < 1).sum().item()

                    # measure accuracy and record loss
                    sample_num = x.size(0)
                    loss_monitor.update(cur_loss, sample_num)
                    accuracy_monitor.update(correct_num, sample_num)
                    _tqdm.set_postfix(OrderedDict(stage="val", epoch=epoch, loss=loss_monitor.avg),
                                      acc=accuracy_monitor.avg, correct=correct_num, sample_num=sample_num)

    preds = np.concatenate(preds, axis=0)
    gt = np.concatenate(gt, axis=0)
    mae = np.abs(preds - gt).mean()

    df = pd.DataFrame(error, columns = ["photo", "age", "pred", "error"])

    if gender_count != "False":
        return loss_monitor.avg, accuracy_monitor.avg, mae, (group_mae, gender_mae), df
    else:
        return loss_monitor.avg, accuracy_monitor.avg, mae, (group_mae,), df


def main():
    args = get_args()

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    start_epoch = 0
    checkpoint_dir = Path(args.checkpoint)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    group = {0:"  0-5", 1:" 6-10", 2:"11-20", 3:"21-30", 4:"31-40", 5:"41-60", 6:"  61-"}
    group_count = torch.zeros(7)
    gender_count = torch.zeros(2)

    # create model
    print("=> creating model '{}'".format(cfg.MODEL.ARCH))
    model = get_model(model_name=cfg.MODEL.ARCH)

    if cfg.TRAIN.OPT == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LR,
                                    momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # optionally resume from a checkpoint
    resume_path = args.resume

    if resume_path:
        if Path(resume_path).is_file():
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location="cpu")
            start_epoch = checkpoint['epoch'] - 1
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, checkpoint['epoch']))
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(resume_path))

    if args.multi_gpu:
        model = nn.DataParallel(model)

    if device == "cuda":
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().to(device)

    gender = False
    if args.dataset == "Morph" or args.dataset == "imdb_wiki":
        gender = True

    val_dataset = FaceVal(args.data_dir, "valid", args.dataset,
                              img_size=cfg.MODEL.IMG_SIZE, augment=False, label=True, gender=gender, expand = args.expand)
    val_loader = DataLoader(val_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.TRAIN.WORKERS, drop_last=False)
    print(len(val_dataset))
    # validate
    if gender:
        if args.ldl:
            val_loss, val_acc, val_mae, maes, df= validate_ldl(val_loader, model, criterion, start_epoch, device, group_count, gender_count)
        else:
            val_loss, val_acc, val_mae, maes, df= validate(val_loader, model, criterion, start_epoch, device, group_count, gender_count)
    else:
        if args.ldl:
            val_loss, val_acc, val_mae, maes, df= validate_ldl(val_loader, model, criterion, start_epoch, device, group_count)
        else:
            val_loss, val_acc, val_mae, maes, df= validate(val_loader, model, criterion, start_epoch, device, group_count)


    print("=> Validation finished")
    print(f"additional opts: {args.opts}")
    print(f"Val MAE: {val_mae:.4f}")
    
    group_mae = maes[0]
    print("Group MAE:")
    for ind, interval in enumerate(group.values()):
        print(interval+":", (group_mae[ind]/group_count[ind]).item())
    
    if gender:
        gender_mae = maes[1]
        for ind, gen in enumerate(["  Male", "Female"]):
            print(gen+":", (gender_mae[ind]/gender_count[ind]).item())
    csv_path = resume_path.split("/")[-1]
    csv_path = csv_path[:-4]
    df.to_csv("csv/"+csv_path+".csv", index=False)


if __name__ == '__main__':
    main()
