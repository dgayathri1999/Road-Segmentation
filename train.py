from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from dataset import *
from model import *

images = os.listdir(dir_img)
images.sort()
segmentations = os.listdir(dir_seg)
segmentations.sort()

X = []
Y = []
n_classes =10
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
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


sgd = optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
# Save the checkpoint in the /output folder
filepath = "./FCN_model/model_fcn.hdf5"

# Keep only a single checkpoint, the best over test accuracy.
checkpoint = ModelCheckpoint(filepath,
                            monitor='accuracy',
                            verbose=1,
                            save_best_only=True,
                            mode='max')

hist1 = model.fit(X_train,y_train,
                  validation_data=(X_test,y_test),
                  batch_size=32,epochs=120,verbose=2,callbacks=[checkpoint])
model.save_weights('./FCN_model/model_fcn.h5')
