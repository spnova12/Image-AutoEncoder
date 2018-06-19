"""
실험 1.
"""

from pathlib import Path
import time

from _utils import *
from _dataset import *
from models001 import *
from _transforms import *
import _eval

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
# cudnn.benchmark allows you to enable the inbuilt cudnn auto-tuner
# to find the best algorithm to use for your hardware.

if __name__ == '__main__':
    # ====< Training settings >=========================================================================================
    num_epochs = 20  # number of epochs to train for
    batch_size = 16  # training batch size
    learning_rate = 0.0001
    sample_step = 1000  # how many iteration will the sample be saved?
    input_channel = 1  # if color -> 3, if gray scale -> 1
    """
    -> 주의 사항 : _dataIO.py 의 load_img(.) 가 사용하는 함수를 input_channel 에 맞게 바꿔줘야 한다.
    """
    threads = 4  # number of threads for data loader to use
    seed = 333  # random seed to use
    device = torch.device('cuda:0')

    qf = '20'
    exp_name = 'exp\\exp004_20'


    origin_dir = 'C:\\datasets\\coco\\original'

    test_dir = 'C:\\kdw\\AutoEncoder\\testdata\\ori'

    weight_save = True
    weight_load = True
    sample_save = True
    log_save = True

    # ------------------------------------------------------------------------------------------------------
    parent_dir = Path(__file__).parents[1]

    if weight_save or weight_load:
        dir_saved_netG = make_dirs(f'{parent_dir}\\{exp_name}\\saved_model\\{qf}') + '\\' + 'netG.pkl'

    if sample_save:
        dir_sample = make_dirs(f'{parent_dir}\\{exp_name}\\samples\\{qf}')

    if log_save:
        dir_log = make_dirs(f'{parent_dir}\\{exp_name}')

    if not torch.cuda.is_available():
        raise Exception("No GPU found")
    else:
        print("===> GPU on")

    torch.manual_seed(seed)

    # ====< Loading datasets >==========================================================================================
    print('===> Loading datasets')
    train_set = ImageDataSet(origin_dir,
                                   Compose([
                                       RandomCrop(48),
                                       Color0_255to1_1(),
                                       ToTensor()
                                   ])
                             )

    training_data_loader = DataLoader(dataset=train_set,
                                      num_workers=threads,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      )

    test_batch = folder_to_batch(test_dir,
                                      Compose([
                                          CenterCrop(48),
                                          Color0_255to1_1(),
                                          ToTensor()
                                      ])
                                 )
    # ====< Building model >============================================================================================
    print('===> Building model')
    netG = Auto_encoder().to(device)
    netG.apply(weights_init)

    G_optimizer = torch.optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    start_epoch = 1
    best_psnr = 0
    # optionally resume from a checkpoint
    if weight_load:
        if os.path.isfile(dir_saved_netG):
            print("===> loading checkpoint '{}'".format(dir_saved_netG))
            checkpoint = torch.load(dir_saved_netG)
            start_epoch = checkpoint['epoch']
            best_psnr = checkpoint['best_psnr']
            netG.load_state_dict(checkpoint['state_dict'])
            G_optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("===> no checkpoint found at '{}'".format(dir_saved_netG))

    MSE = nn.MSELoss().to(device)
    BCE = nn.BCELoss().to(device)

    # ====< Training and Evaluation>====================================================================================
    def train_base(epoch):
        for i, batch in enumerate(training_data_loader, 1):
            netG.train()
            input = batch.to(device)
            output = netG(input)
            loss = MSE(output, input)

            G_optimizer.zero_grad()
            loss.backward()
            G_optimizer.step()

            if i % 100 == 0:
                print("===> Epoch[{}/{}]({}/{}): Loss: {:.4f}".format(epoch, num_epochs, i, len(training_data_loader), loss.item()))

            # save the sampled images
            if sample_save:
                if (i) % sample_step == 0:
                    sample_name = '%03d_%04d.png' % (epoch, i)
                    _eval.batch2img(dir_sample + '\\' + sample_name,
                                    test_batch.to(device), netG, tile_shape=(3, 3))


    # 학습 전의 psnr 체크
    # 회손 이미지와 원본 이미지 사이의 psnr
    log = "Avg. PSNR(artifact, origin): {:.4f} dB".format(
        _eval.batch2psnr(test_batch.to(device), test_batch.to(device)))
    print(log)
    if log_save:
        save_log(dir_log + "\\log.txt", log)

    # 학습 안된 모델을 통과한 회손 이미지와 원본 이미지 사이의 psnr
    log = "epoch : 0 Avg. PSNR(recon, origin): {:.4f} dB".format(
        _eval.batch2psnr(test_batch.to(device), test_batch.to(device), netG))
    print(log)

    # 학습 시작
    start_time = time.time()

    for epoch in range(start_epoch, num_epochs+1):
        print(f'epoch {epoch} start -------------------------------------------------------------')
        adjust_learning_rate(init_lr=learning_rate, optimizer=G_optimizer, epoch=epoch)

        # train for one epoch
        train_base(epoch)

        # evaluate on validation set
        if epoch % 1 == 0:
            psnr = _eval.batch2psnr(test_batch.to(device), test_batch.to(device), netG)
            log = "epoch : {} Avg. PSNR(recon, origin): {:.4f} dB (Best : {:.4f} dB)".format(epoch, psnr, best_psnr)
            print(log)

            # Time spent
            end_time = time.time()
            print("Time spent : {}".format(seconds_to_hours(end_time - start_time)))

            # Save log
            if log_save:
                save_log(dir_log +"\\log.txt", log)

            if weight_save:
                # check best psnr
                is_best = psnr > best_psnr
                best_psnr = max(psnr, best_psnr)

                save_checkpoint(
                    {
                    'epoch': epoch,
                    'state_dict': netG.state_dict(),
                    'best_psnr': best_psnr,
                    'optimizer': G_optimizer.state_dict(),
                    },
                    is_best,
                    filename=dir_saved_netG)



