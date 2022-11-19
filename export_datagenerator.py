import numpy as np
import tensorflow as tf
from skimage.filters import (threshold_sauvola)

from config import *


class ExportDataGenerator(tf.keras.utils.Sequence):
    allImages = {}

    def getImage(self, imgPath, imgSize) -> np.ndarray:
        if imgPath is None:
            img = np.zeros([imgSize[0], imgSize[1]], dtype=float)
            print("Image broken, zeroing")
            return img

        if imgPath in self.allImages:
            return self.allImages[imgPath]

        img = tf.io.read_file(imgPath)
        try:
            img = tf.image.decode_image(img, channels=config.NUM_CHANNELS, dtype=tf.dtypes.float32)
        except:
            print(imgPath)
            img = np.zeros([imgSize[0], imgSize[1], config.NUM_CHANNELS], dtype=float)
            print("Image broken, zeroing")
        self.allImages[imgPath] = img
        return img

    def preprocess(self, imgPath, imgSize, channels, augment=False) -> np.ndarray:
        if imgPath is None:
            img = np.zeros([imgSize[0], imgSize[1]], dtype=float)
            print("Image broken, zeroing")
        else:
            img = tf.io.read_file(imgPath)
        # tf.io.write_file("/tmp/1.png",img)
        try:
            # img = tf.image.decode_image(img, channels=config.NUM_CHANNELS, dtype=tf.dtypes.float32)
            img = tf.image.decode_image(img, channels=channels, dtype=tf.dtypes.float32)
        except:
            print(imgPath)
            img = np.zeros([imgSize[0], imgSize[1], channels], dtype=float)
            # img = np.zeros([imgSize[0], imgSize[1], config.NUM_CHANNELS], dtype=float)
            print("Image broken, zeroing")

        if self.do_binarize_otsu:
            if img.shape[2] > 1:
                img = tf.image.rgb_to_grayscale(img)
            img = img * 255
            img = self.otsu_thresholding(img)
        if self.do_binarize_sauvola:
            if img.shape[2] > 1:
                img = tf.image.rgb_to_grayscale(img)
            img = self.sauvola(img.numpy())

        img = tf.image.resize(img, [imgSize[0], 65536], preserve_aspect_ratio=True)
        img = tf.clip_by_value(img, 0.0, 1.0)

        return img

        # https://colab.research.google.com/drive/1CdVfa2NlkQBga1E9dBwHved36Tk7Bg61#scrollTo=Jw-NU1wbHnWA

    def otsu_thresholding(self, image):
        image = tf.convert_to_tensor(image, name="image")
        # image = tf.squeeze(image)
        rank = image.shape.rank
        if rank != 2 and rank != 3:
            raise ValueError("Image should be either 2 or 3-dimensional.")
        # print (image.shape)

        if image.dtype != tf.int32:
            image = tf.cast(image, tf.int32)

        r, c, detected_channels = image.shape
        hist = tf.math.bincount(image, dtype=tf.int32)

        if len(hist) < 256:
            hist = tf.concat([hist, [0] * (256 - len(hist))], 0)

        current_max, threshold = 0, 0
        total = r * c

        spre = [0] * 256
        sw = [0] * 256
        spre[0] = int(hist[0])

        for i in range(1, 256):
            spre[i] = spre[i - 1] + int(hist[i])
            sw[i] = sw[i - 1] + (i * int(hist[i]))

        for i in range(256):
            if total - spre[i] == 0:
                break

            meanB = 0 if int(spre[i]) == 0 else sw[i] / spre[i]
            meanF = (sw[255] - sw[i]) / (total - spre[i])
            varBetween = (total - spre[i]) * spre[i] * ((meanB - meanF) ** 2)

            if varBetween > current_max:
                current_max = varBetween
                threshold = i

        final = tf.where(image > threshold, 0, 255)
        # final = tf.expand_dims(final, -1)
        return final

    'Generates data for Keras'

    def __init__(self, list_IDs, labels, batch_size=32,
                 shuffle=True,
                 do_binarize_otsu=False,
                 do_binarize_sauvola=False,
                 height=51,
                 width=251,
                 channels=3,
                 sauvola_window=11
                 ):

        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        labelValues = labels.values()
        self.uniqueLabels = set(labelValues)
        self.do_binarize_otsu = do_binarize_otsu
        self.do_binarize_sauvola = do_binarize_sauvola
        self.height = height
        self.width = width
        self.channels = channels
        self.sauvola_window = sauvola_window

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)
        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        X = []
        Y = []
        # X = np.empty((self.batch_size), dtype=object)
        y = np.empty((self.batch_size), dtype=int)
        Z = []
        maxWidthFirst = 0
        maxWidthSecond = 0
        maxHeightFirst = 0
        maxHeightSecond = 0
        # TODO: make this multithreading
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            # X[i,] = np.load('data/' + ID + '.npy')

            # X[i,] = self.preprocess(ID, config.IMG_SHAPE)
            X.append(self.preprocess(ID, (self.height, self.width), self.channels))
            if X[i].shape[0]>maxHeightFirst:
                maxHeightFirst = X[i].shape[0]
            if X[i].shape[1]>maxWidthFirst:
                maxWidthFirst = X[i].shape[1]
            # X[i,] = np.asarray(self.preprocess(ID, config.IMG_SHAPE)).astype('float32')
            # image = self.preprocess(ID, config.IMG_SHAPE)
            # tensor = tf.convert_to_tensor(image, dtype=tf.float32)
            # X[i] =tensor
            # X.append(tensor)
            # Store class
            # y[i] = self.labels['textline_f5c183b3-f824-41ee-ace0-9934917d4c80_8e81ee47-26da-4973-b479-4c0973f0becd.jpg']
            label = self.labels[ID]

        for i, ID in enumerate(list_IDs_temp):
            X[i] = tf.image.resize_with_pad(X[i], maxHeightFirst,maxWidthFirst)

        X = tf.convert_to_tensor(X)
        return X  # , keras.utils.to_categorical(y, num_classes=self.n_classes)


    def getLabel(self, ID):
        return list(self.labels.keys())[ID]

    def sauvola(self, image):
        thresh_sauvola = threshold_sauvola(image, window_size=self.sauvola_window, k=0.2)
        binary_sauvola = np.invert(image > thresh_sauvola)*1

        return tf.convert_to_tensor(binary_sauvola)
