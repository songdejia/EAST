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
from utils.myzip import *
import torch.backends.cudnn as cudnn
from eval import predict
from hmean import compute_hmean
import zipfile
import glob
import warnings
import numpy as np



def train(train_loader, model, criterion, scheduler, optimizer, epoch):
    start = time.time()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    model.train()
    
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
            print('EAST <==> TRAIN <==> Epoch: [{0}][{1}/{2}] Loss {loss.val:.4f} Avg Loss {loss.avg:.4f})\n'.format(epoch, i, len(train_loader), loss=losses))

        save_loss_info(losses, epoch, i, train_loader)


def main():
    hmean = .0
    is_best = False

    warnings.simplefilter('ignore', np.RankWarning)
    # Prepare for dataset
    print('EAST <==> Prepare <==> DataLoader <==> Begin')
    train_root_path = os.path.abspath(os.path.join('./dataset/', 'train'))
    train_img = os.path.join(train_root_path, 'img')
    train_gt  = os.path.join(train_root_path, 'gt')

    trainset = custom_dset(train_img, train_gt)
    train_loader = DataLoader(trainset, batch_size=cfg.train_batch_size_per_gpu*cfg.gpu,
        shuffle=True, collate_fn=collate_fn, num_workers=cfg.num_workers)
    print('EAST <==> Prepare <==> Batch_size:{} <==> Begin'.format(cfg.train_batch_size_per_gpu*cfg.gpu))
    print('EAST <==> Prepare <==> DataLoader <==> Done') 
    

    # test datalodaer
    """
    for i in range(100000):
        for j, (a,b,c,d) in enumerate(train_loader):
            print(i, j,'/',len(train_loader))
    """

    # Model
    print('EAST <==> Prepare <==> Network <==> Begin')
    model = East()
    model = nn.DataParallel(model, device_ids=cfg.gpu_ids)
    model = model.cuda()
    init_weights(model, init_type=cfg.init_type)
    cudnn.benchmark = True
    
    criterion = LossFunc()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.94)
    
    # init or resume
    if cfg.resume and  os.path.isfile(cfg.checkpoint):
        weightpath = os.path.abspath(cfg.checkpoint)
        print("EAST <==> Prepare <==> Loading checkpoint '{}' <==> Begin".format(weightpath))
        checkpoint = torch.load(weightpath)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("EAST <==> Prepare <==> Loading checkpoint '{}' <==> Done".format(weightpath))
    else:
        start_epoch = 0
    print('EAST <==> Prepare <==> Network <==> Done')

    for epoch in range(start_epoch, cfg.max_epochs):

        train(train_loader, model, criterion, scheduler, optimizer, epoch)

        if epoch % cfg.eval_iteration == 0:

            # create res_file and img_with_box
            output_txt_dir_path = predict(model, criterion, epoch)

            # Zip file
            submit_path = MyZip(output_txt_dir_path, epoch)

            # submit and compute Hmean
            hmean_ = compute_hmean(submit_path)

            if hmean_ > hmean:
                is_best = True

            state = {
                    'epoch'      : epoch,
                    'state_dict' : model.state_dict(),
                    'optimizer'  : optimizer.state_dict(),
                    'is_best'    : is_best,
                    }
            save_checkpoint(state, epoch)



if __name__ == "__main__":
    main()
