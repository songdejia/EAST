import torch
import os
import shutil
import datetime
from utils.util import *
# this is for weight
def save_checkpoint(state, epoch, filename='checkpoint.pth.tar'):
    """[summary]

    [description]

    Arguments:
        state {[type]} -- [description] a dict describe some params
        is_best {bool} -- [description] a bool value

    Keyword Arguments:
        filename {str} -- [description] (default: {'checkpoint.pth.tar'})
    """
    print('EAST <==> Save weight - epoch {} <==> Begin'.format(epoch))
    root_dir = os.path.abspath('./')
    weight_dir = os.path.join(root_dir, 'weight')

    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)

    filename = 'epoch_'+str(epoch)+'_checkpoint.pth.tar'
    file_path = os.path.join(weight_dir, filename)
    torch.save(state, file_path)

    if state['is_best']:
        src = file_path
        dst = os.path.join(weight_dir, 'best_model.pth.tar')
        shutil.copyfile(src, dst)
    print('EAST <==> Save weight - epoch {} <==> Done'.format(epoch))

def save_loss_info(losses, epoch, current_batch, loader, path='./log.txt'):
    default_path = os.path.abspath(path)

    dir_path = os.path.dirname(default_path)

    log_loss_path = os.path.join(dir_path, 'result', 'log_loss.txt')
    
    if not os.path.isfile(log_loss_path):
        os.mknod(log_loss_path)

    with open(path, 'a') as f:
        line = 'Epoch: [{0}][{1}/{2}]\t Loss {loss.val:.4f} ({loss.avg:.4f})\n'.format(epoch,current_batch, len(loader), loss = losses)
        f.write(line)





