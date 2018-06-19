import cv2
import os
import numpy as np


def load_img(filepath):
    """
    :param filepath: 원하는 영상의 dir
    :return: 1 채널 혹은 3채널의 영상

    예시)
    img = load_grayscale(filepath)
    또는
    img = load_YUV_I420(filepath)
    등..
    """
    img = load_grayscale(filepath)
    return img


def load_BGR(filepath):
    img_BGR = cv2.imread(filepath, cv2.IMREAD_COLOR)
    return img_BGR


def load_grayscale(filepath):
    img_grayscale = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return img_grayscale


def load_YUV_I420(filepath):
    """
    참고 싸이트 :
    http://crynut84.tistory.com/56
    https://en.wikipedia.org/wiki/YUV
    http://blog.dasomoli.org/265/
    https://picamera.readthedocs.io/en/release-1.10/recipes2.html#unencoded-image-capture-yuv-format
    https://raspberrypi.stackexchange.com/questions/28033/reading-frames-of-uncompressed-yuv-video-file?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

    :param filepath: YUV_I420 포맷 영상의 dir
    :return: yuvi420 비디오의 첫번째 프래임의 YUV 채널 중 Y 채널만 반환

    >>> y = load_YUV_I420('0002018_320x480_P420_8b.yuv')
    >>> cv2.imshow('image', y)
    >>> cv2.waitKey(0)
    """
    # yuv420 포맷에서 중 y 만 읽는다.
    img_name = os.path.basename(filepath)
    img_name = os.path.splitext(img_name)[0]
    w, h = img_name.split('_')[1].split('x')
    w, h = int(w), int(h)
    frame_len = w * h
    f = open(filepath, 'rb')
    try:
        raw = f.read(int(frame_len))
        y = np.frombuffer(raw, dtype=np.uint8)
        # cv2 grayscale image shape is height x width
        img_y = y.reshape(h, w)
    except Exception as e:
        print(str(e))
        return None
    return img_y


if __name__ == '__main__':
    y = load_YUV_I420('0002018_320x480_P420_8b.yuv')
    print(y.dtype)
    cv2.imshow('image_', y)
    cv2.waitKey(0)


