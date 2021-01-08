from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers

import numpy as np
from matplotlib import pyplot as plt

from dataset import train_test, give_color_to_seg_img
from model import FCN8

sgd = optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)

model = FCN8(nClasses=10,
             input_height=224,
             input_width=224)


X_train, y_train, X_test, y_test = train_test()

model = load_model("./FCN_model/model_fcn.hdf5", compile=False)
print(model)
y_pred = model.predict(X_test)
y_predi = np.argmax(y_pred, axis=3)
y_testi = np.argmax(y_test, axis=3)
print(y_testi.shape,y_predi.shape)


def IoU(Yi, y_predi):
    ## mean Intersection over Union
    ## Mean IoU = TP/(FN + TP + FP)

    IoUs = []
    Nclass = int(np.max(Yi)) + 1
    for c in range(Nclass):
        TP = np.sum((Yi == c) & (y_predi == c))
        FP = np.sum((Yi != c) & (y_predi == c))
        FN = np.sum((Yi == c) & (y_predi != c))
        IoU = TP / float(TP + FP + FN)
        print("class {:02.0f}: #TP={:6.0f}, #FP={:6.0f}, #FN={:5.0f}, IoU={:4.3f}".format(c, TP, FP, FN, IoU))
        IoUs.append(IoU)
    mIoU = np.mean(IoUs)
    print("_________________")
    print("Mean IoU: {:4.3f}".format(mIoU))


IoU(y_testi, y_predi)

shape = (224, 224)
n_classes = 10

for i in range(20):
    img_is = (X_test[i] + 1) * (255.0 / 2)
    seg = y_predi[i]
    segtest = y_testi[i]

    fig = plt.figure(figsize=(10, 30))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(img_is / 255.0)
    ax.set_title("original")

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(give_color_to_seg_img(seg, n_classes))
    ax.set_title("predicted class")

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(give_color_to_seg_img(segtest, n_classes))
    ax.set_title("true class")
    plt.show()