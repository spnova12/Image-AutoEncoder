from math import log10
from _transforms import *
import cv2


def batch2psnr(input, target, model=None):
    """
    batch 형태의 input target 에 대해 psnr 을 구한다.
    :param input: input image 들의 batch (tensor)
    :param target: target image 들의 batch (tensor)
    :param model: model을 할당해 주면, model 을 통해 input 을 복원하고 target 과의 psnr 을 구한다.
    :return: 이미지 batcch 들의 psnr (float)
    """
    with torch.no_grad():
        if model is not None:
            model.eval()
            prediction = model(input)
        else:
            prediction = input
        se = (prediction - target) ** 2
        if torch.sum(se).item() == 0:
            return 100
        # psnr 의 정의 (color 범위가 -1 ~ 1 인 경우) : 10 * log10((max-mean) ** 2 / mse))
        psnr = [10 * log10(4 / torch.mean(x).item()) for x in se]
        psnr = np.mean(psnr)
        # print("===> Avg. PSNR: {:.4f} dB".format(psnr))
        return psnr


def batch2psnr_withnoise():
    pass


def batch2img(name, input, model, tile_shape):
    """
    batch 형태의 input 이미지들을 복원하고 merge 해서 한장으로 만들어준다.
    :param name: merged 이미지가 저장될 경로
    :param input: batch (tensor)
    :param model: model
    :param height: 타일의 세로 개수
    :param width: 타일의 가로 개수
    """
    with torch.no_grad():
        model.eval()
        prediction = model(input)
        result = batch_to_merged_img(prediction, tile_shape)
        result = Color1_1to0_255(result)
        cv2.imwrite(name, result)
