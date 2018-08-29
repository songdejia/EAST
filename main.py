import torch
from torch.autograd import Variable
import os
from torch import nn
from torch.optim import lr_scheduler
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torchvision import transforms
from model import East
from loss import *
from data_utils import custom_dset, collate_fn
import time
from tensorboardX import SummaryWriter
import config as cfg
from utils.init import *
from utils.util import *
from utils.save import *
import torch.backends.cudnn as cudnn
from eval import predict
import zipfile
import glob
import warnings
import numpy as np



def train(train_loader, model, criterion, scheduler, optimizer, epoch):
    print('*'*50)
    print('Epoch {} / {}'.format(epoch + 1, cfg.max_epochs))
    start = time.time()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    for i, (img, score_map, geo_map, training_mask) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if cfg.gpu is not None:
            img, score_map, geo_map, training_mask = img.cuda(), score_map.cuda(), geo_map.cuda(), training_mask.cuda()

        f_score, f_geometry = model(img)
        loss1 = criterion(score_map, f_score, geo_map, f_geometry, training_mask)
        losses.update(loss1.item(), img.size(0))

        # backward
        scheduler.step()
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\n'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\n'.format(epoch, i, len(train_loader), loss=losses))

        save_loss_info(losses, epoch, i, train_loader)


def main():
    warnings.simplefilter('ignore', np.RankWarning)
    # Prepare for dataset
    train_path = os.path.join(cfg.dataroot, 'train')
    train_img = os.path.join(train_path, 'img')
    train_gt  = os.path.join(train_path, 'gt')

    trainset = custom_dset(train_img, train_gt)
    train_loader = DataLoader(trainset, batch_size=cfg.train_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=cfg.num_workers)
    print('load img from {}'.format(train_path))
    print('load gt from {}'.format(train_gt))
    print('Data loader Done')

    # Model
    model = East()
    model = nn.DataParallel(model, device_ids=cfg.gpu_ids)
    model = model.cuda()
    init_weights(model, init_type=cfg.init_type)
    cudnn.benchmark = True
    print('Model initialization Done')

    # init or resume
    if cfg.resume and  os.path.isfile(cfg.checkpoint):
        weightpath = os.path.abspath(cfg.checkpoint)
        print(weightpath)
        print("=> loading checkpoint '{}'".format(weightpath))
        checkpoint = torch.load(weightpath)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(weightpath, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(cfg.checkpoint))
        start_epoch = 0

    criterion = LossFunc()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.94)
    print('Criterion and Optimizer is Done')

    for epoch in range(start_epoch, cfg.max_epochs):

        train(train_loader, model, criterion, scheduler, optimizer, epoch)

        if epoch % cfg.eval_iteration == 0:

            output_txt_dir_path = predict(model, criterion, epoch)
            print('epoch {} prediction'.format(epoch))

            # zip and move to  compute F1
            shutil.make_archive('epoch_{}_submit'.format(epoch),'zip',output_txt_dir_path+'/')

            command = 'mv epoch_{}_submit.zip ./submit'.format(epoch)
            os.system(command)

        if epoch % cfg.save_iteration == 0:
            state = {
                    'epoch' : epoch + 1,
                    'state_dict' : model.state_dict(),
                    'optimizer'  : optimizer.state_dict(),
                    }
            save_checkpoint(state, epoch)
            print('epoch {} saved'.format(epoch))



if __name__ == "__main__":
    main()
