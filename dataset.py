import cv2, os
import numpy as np
import seaborn as sns
from sklearn.utils import shuffle

dir_data = "./dataset1"
dir_seg = dir_data + "/annotations_prepped_train/"
dir_img = dir_data + "/images_prepped_train/"

input_height, input_width = 224, 224
output_height, output_width = 224, 224
n_classes = 10


def give_color_to_seg_img(seg, n_classes):
    '''
    seg : (input_width,input_height,3)
    '''

    if len(seg.shape) == 3:
        seg = seg[:, :, 0]
    seg_img = np.zeros((seg.shape[0], seg.shape[1], 3)).astype('float')
    colors = sns.color_palette("hls", n_classes)

    c = 3
    segc = (seg == c)
    seg_img[:, :, 0] += (segc * (colors[c][0]))
    seg_img[:, :, 1] += (segc * (colors[c][1]))
    seg_img[:, :, 2] += (segc * (colors[c][2]))

    return (seg_img)


def getImageArr(path, width, height):
    img = cv2.imread(path, 1)
    img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
    return img


def getSegmentationArr(path, nClasses, width, height):
    seg_labels = np.zeros((height, width, nClasses))
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (width, height))
    img = img[:, :, 0]

    for c in range(nClasses):
        seg_labels[:, :, c] = (img == c).astype(int)
    return seg_labels


def train_test():
    images = os.listdir(dir_img)
    images.sort()
    segmentations = os.listdir(dir_seg)
    segmentations.sort()

    X = []
    Y = []

    for im, seg in zip(images, segmentations):
        X.append(getImageArr(dir_img + im, input_width, input_height))
        Y.append(getSegmentationArr(dir_seg + seg, n_classes, output_width, output_height))

    X, Y = np.array(X), np.array(Y)
    print(X.shape, Y.shape)

    train_rate = 0.85
    index_train = np.random.choice(X.shape[0], int(X.shape[0] * train_rate), replace=False)
    index_test = list(set(range(X.shape[0])) - set(index_train))

    X, Y = shuffle(X, Y)
    X_train, y_train = X[index_train], Y[index_train]
    X_test, y_test = X[index_test], Y[index_test]
    return X_train, y_train, X_test, y_test
