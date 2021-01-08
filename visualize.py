from matplotlib import pyplot as plt
from dataset import *


input_height, input_width = 224, 224
output_height, output_width = 224, 224
n_classes=10

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

dir_data = "./dataset1"
dir_seg = dir_data + "/annotations_prepped_train/"
dir_img = dir_data + "/images_prepped_train/"

## seaborn has white grid by default so I will get rid of this.
sns.set_style("whitegrid", {'axes.grid' : False})


ldseg = np.array(os.listdir(dir_seg))
fnm = ldseg[0]
print(fnm)

## read in the original image and segmentation labels
seg = cv2.imread(dir_seg + fnm ) # (360, 480, 3)
img_is = cv2.imread(dir_img + fnm )
print("seg.shape={}, img_is.shape={}".format(seg.shape,img_is.shape))

## Check the number of labels
mi, ma = np.min(seg), np.max(seg)
n_classes = ma - mi + 1
print("minimum seg = {}, maximum seg = {}, Total number of segmentation classes = {}".format(mi,ma, n_classes))

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1,2,1)
ax.imshow(img_is)
ax.set_title("original image")
ax = fig.add_subplot(1,2,2)
ax.imshow(seg)
ax.set_title("segmented image")
plt.show()

ldseg = np.array(os.listdir(dir_seg))
for fnm in ldseg[np.random.choice(len(ldseg), 3, replace=False)]:
    fnm = fnm.split(".")[0]
    seg = cv2.imread(dir_seg + fnm + ".png")  # (360, 480, 3)
    img_is = cv2.imread(dir_img + fnm + ".png")
    seg_img = give_color_to_seg_img(seg, n_classes)

    fig = plt.figure(figsize=(20, 40))
    ax = fig.add_subplot(1, 4, 1)
    ax.imshow(seg_img)

    ax = fig.add_subplot(1, 4, 2)
    ax.imshow(img_is / 255.0)
    ax.set_title("original image {}".format(img_is.shape[:2]))

    ax = fig.add_subplot(1, 4, 3)
    ax.imshow(cv2.resize(seg_img, (input_height, input_width)))

    ax = fig.add_subplot(1, 4, 4)
    ax.imshow(cv2.resize(img_is, (output_height, output_width)) / 255.0)
    ax.set_title("resized to {}".format((output_height, output_width)))
    plt.show()