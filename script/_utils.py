import os
import shutil

import torch


def save_checkpoint(state, is_best, filename='checkpoint.pkl'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.dirname(filename) + '\\checkpoint_best.pkl')


def save_log(log_dir, log):
    # Save log
    f = open(log_dir, 'a')
    f.write(log + '\n')
    f.close()


def make_dirs(path):
    """
    경로(폴더) 가 있음을 확인하고 없으면 새로 생성한다.
    :param path: 확인할 경로
    :return: path
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def adjust_learning_rate(init_lr, optimizer, epoch, n=100):
    """Sets the learning rate to the initial learning rate decayed by 10 every n epochs"""
    init_lr = init_lr * (0.1 ** (epoch // n))
    print('learning rate : ', init_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr


def seconds_to_hours(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)