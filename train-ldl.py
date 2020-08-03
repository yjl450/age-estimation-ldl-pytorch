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
from dataset import FaceDataset
from defaults import _C as cfg
from datetime import datetime
from matplotlib import pyplot as plt
import loss as L


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
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint if any")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoint", help="Checkpoint directory")
    parser.add_argument("--tensorboard", type=str,
                        default=None, help="Tensorboard log directory")
    parser.add_argument('--multi_gpu', action="store_true",
                        help="Use multi GPUs (data parallel)")
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


def train(train_loader, model, criterion, optimizer, epoch, device):
    model.train()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()
    rank = torch.Tensor([i for i in range(101)]).to(device)

    with tqdm(train_loader) as _tqdm:
        for x, y, lbl in _tqdm:
            x = x.to(device)
            y = y.to(device)
            lbl = lbl.to(device)

            # compute output
            outputs = model(x)
            outputs = F.softmax(outputs, dim = 1)
            ages = torch.sum(outputs*rank, dim=1)

            # calc loss
            # loss = criterion(outputs, y)
            loss1 = L.kl_loss(outputs, lbl)
            loss2 = L.L1_loss(ages, y)
            loss = loss1 + loss2
            cur_loss = loss.item()

            # calc accuracy
            correct_num = (abs(ages - y) < 1).sum().item()

            # measure accuracy and record loss
            sample_num = x.size(0)
            loss_monitor.update(cur_loss, sample_num)
            accuracy_monitor.update(correct_num, sample_num)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _tqdm.set_postfix(OrderedDict(stage="train", epoch=epoch, loss=loss_monitor.avg),
                              acc=accuracy_monitor.avg, correct=correct_num, sample_num=sample_num)

    return loss_monitor.avg, accuracy_monitor.avg


def validate(validate_loader, model, criterion, epoch, device):
    model.eval()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()
    preds = []
    gt = []
    rank = torch.Tensor([i for i in range(101)]).to(device)

    with torch.no_grad():
        with tqdm(validate_loader) as _tqdm:
            for i, (x, y, lbl) in enumerate(_tqdm):
                x = x.to(device)
                y = y.to(device)
                lbl = lbl.to(device)

                # compute output
                outputs = model(x)
                outputs = F.softmax(outputs, dim = 1)
                ages = torch.sum(outputs*rank, dim=1)  # age expectation
                preds.append(ages.cpu().numpy())  # append predicted age
                gt.append(y.cpu().numpy())  # append real age

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
    # ages = np.arange(0, 101)
    # ave_preds = (preds * ages).sum(axis=-1)
    mae = np.abs(preds - gt).mean()

    return loss_monitor.avg, accuracy_monitor.avg, mae


def main():
    args = get_args()

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    start_epoch = 0
    checkpoint_dir = Path(args.checkpoint)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
            start_epoch = checkpoint['epoch']
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
    train_dataset = FaceDataset(args.data_dir, "train", args.dataset, img_size=cfg.MODEL.IMG_SIZE, augment=True,
                                age_stddev=cfg.TRAIN.AGE_STDDEV, label=True)
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.TRAIN.WORKERS, drop_last=False)

    val_dataset = FaceDataset(args.data_dir, "valid", args.dataset,
                              img_size=cfg.MODEL.IMG_SIZE, augment=False, label=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.TRAIN.WORKERS, drop_last=False)

    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.LR_DECAY_RATE,
                       last_epoch=start_epoch - 1)
    best_val_mae = 10000.0
    train_writer = None

    if args.tensorboard is not None:
        opts_prefix = "_".join(args.opts)
        train_writer = SummaryWriter(
            log_dir=args.tensorboard + "/" + opts_prefix + "_train")
        val_writer = SummaryWriter(
            log_dir=args.tensorboard + "/" + opts_prefix + "_val")

    all_train_loss = []
    all_train_accu = []
    all_val_loss = []
    all_val_accu = []

    # range(start_epoch, cfg.TRAIN.EPOCHS):
    for epoch in range(cfg.TRAIN.EPOCHS):
        # train
        train_loss, train_acc = train(
            train_loader, model, criterion, optimizer, epoch, device)

        # validate
        val_loss, val_acc, val_mae = validate(
            val_loader, model, criterion, epoch, device)

        if args.tensorboard is not None:
            train_writer.add_scalar("loss", train_loss, epoch)
            train_writer.add_scalar("acc", train_acc, epoch)
            val_writer.add_scalar("loss", val_loss, epoch)
            val_writer.add_scalar("acc", val_acc, epoch)
            val_writer.add_scalar("mae", val_mae, epoch)

        all_train_loss.append(float(train_loss))
        all_train_accu.append(float(train_acc))
        all_val_loss.append(float(val_loss))
        all_val_accu.append(float(val_mae))

        # checkpoint
        if val_mae < best_val_mae:
            print(
                f"=> [epoch {epoch:03d}] best val mae was improved from {best_val_mae:.3f} to {val_mae:.3f}")
            model_state_dict = model.module.state_dict(
            ) if args.multi_gpu else model.state_dict()
            torch.save(
                {
                    'epoch': epoch + 1,
                    'arch': cfg.MODEL.ARCH,
                    'state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict()
                },
                str(checkpoint_dir.joinpath("epoch{:03d}_{}_{:.5f}_{:.4f}_{}_{}_ldl.pth".format(
                    epoch, args.dataset, val_loss, val_mae, datetime.now().strftime("%Y%m%d"), cfg.MODEL.ARCH)))
            )
            best_val_mae = val_mae
        else:
            print(
                f"=> [epoch {epoch:03d}] best val mae was not improved from {best_val_mae:.3f} ({val_mae:.3f})")

        # adjust learning rate
        scheduler.step()

    print("=> training finished")
    print(f"additional opts: {args.opts}")
    print(f"best val mae: {best_val_mae:.3f}")

    x = np.arange(cfg.TRAIN.EPOCHS)
    plt.xlabel("Epoch")

    plt.ylabel("Train Loss")
    plt.plot(x, all_train_loss)
    plt.savefig("savefig/{}_{}_{}_train_loss.png".format(args.dataset,
                                                         cfg.MODEL.ARCH, datetime.now().strftime("%Y%m%d")))
    plt.clf()

    plt.ylabel("Train Accuracy")
    plt.plot(x, all_train_accu)
    plt.savefig("savefig/{}_{}_{}_train_accu.png".format(args.dataset,
                                                         cfg.MODEL.ARCH, datetime.now().strftime("%Y%m%d")))
    plt.clf()

    plt.ylabel("Validation Loss")
    plt.plot(x, all_val_loss)
    plt.savefig("savefig/{}_{}_{}_val_loss.png".format(args.dataset,
                                                       cfg.MODEL.ARCH, datetime.now().strftime("%Y%m%d")))
    plt.clf()

    plt.ylabel("Validation Accuracy")
    plt.plot(x, all_val_accu)
    plt.savefig("savefig/{}_{}_{}_val_mae.png".format(args.dataset,
                                                      cfg.MODEL.ARCH, datetime.now().strftime("%Y%m%d")))


if __name__ == '__main__':
    main()
