import numpy as np
import tensorflow as tf
import random
import tensorflow_addons as tfa
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)
from pathlib import Path
import lmdb
import pickle
import cv2


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, img_size, list_IDs, labels, batch_size=32, dim=(227, 227),
                 shuffle=True, channels=3, do_binarize_otsu=False, do_binarize_sauvola=False,
                 center_zero=False,
                 augment=False,
                 sauvola_window=11,
                 sauvola_k=0.2,
                 random_crop=False,
                 cache_path=None,
                 use_existing_lmdb=None
                 ):
        """Initialization"""
        if cache_path:
            self.lmdb_dir = Path(cache_path)

        self.img_size = img_size
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        labelValues = labels.values()
        self.uniqueLabels = set(labelValues)
        self.do_binarize_otsu = do_binarize_otsu
        self.do_binarize_sauvola = do_binarize_sauvola
        self.channels = channels
        self.center_zero = center_zero
        self.augment = augment
        self.sauvola_window = sauvola_window
        self.sauvola_k = sauvola_k
        self.random_crop = random_crop
        self.use_existing_lmdb = use_existing_lmdb

        tmp_batch_size = self.batch_size
        self.batch_size = 1
        if not use_existing_lmdb and cache_path:
            print('creating new lmdb')
            self.write_lmdb()
            for i in range(0, self.__len__()):
                image_id = self.list_IDs[i]
                image = self.getImage(image_id, img_size, channels)
                k = self.sauvola_k
                if self.center_zero:
                    image = image/255
                    image = 1 - image
                    # mean = np.mean(image)
                    median = np.median(image)
                    image = image - median
                    image = np.clip(image, 0.0, 1.0)
                    image = 1 - image
                    standard_deviation = np.std(image)
                    k = standard_deviation
                image = np.expand_dims(image, axis=-1)
                if self.do_binarize_sauvola:
                    image = (self.sauvola(image, k, False))
                image = image.astype('float')
                image = self.image_resize(image, height=img_size[0]) / 255
                self.store_single_lmdb(image, i)
            with open(cache_path+'/labels.pkl', 'wb') as f:
                pickle.dump(self.labels, f)
        self.batch_size = tmp_batch_size
        self.env = lmdb.open(str(self.lmdb_dir / f"single_lmdb"), readonly=True)

    def preprocess(self, id, img_path, img_size, channels, augment=False) -> np.ndarray:
        if True:
            image = self.read_single_lmdb(id)
            image = np.expand_dims(image, axis=-1)
            # image = np.expand_dims(image, -1)*255
            # gtImageEncoded = tf.image.encode_png(tf.image.convert_image_dtype(image, dtype=tf.uint8))
            # # gtImageEncoded = tf.image.encode_png(img)
            # tf.io.write_file("/tmp/test-" + str(id) + ".png", gtImageEncoded)
        else:
            image = self.getImage(img_path, img_size, channels)
            k = self.sauvola_k
            if self.center_zero:
                image = image / 255
                image = 1 - image
                # mean = np.mean(image)
                median = np.median(image)
                image = image - median
                image = np.clip(image, 0.0, 1.0)
                image = 1 - image
                standard_deviation = np.std(image)
                k = standard_deviation
                # print('image')
                # print(image)
            image = self.image_resize(image, height=img_size[0]) / 255
            # print('resized')
            # print(image)
            image = image.astype('float')
            if self.do_binarize_sauvola:
                image = (self.sauvola(image, k, False)).astype("uint8")
                # print('after sauvola')
                # print(image)
            # image = np.expand_dims(image, axis=-1)
            # gtImageEncoded = tf.image.encode_png(tf.image.convert_image_dtype(image, dtype=tf.uint8))
            # tf.io.write_file("/tmp/test-" + str(i) + ".png", gtImageEncoded)
        # random crop
        height = image.shape[0]
        width = image.shape[1]
        # print(str(height) + " " + str(width))
        # targetHeight = imgSize[0]
        # targetWidth = imgSize[1]
        # x = random.randint(0, width - targetWidth)
        # y = random.randint(0, height - targetHeight)
        # print (x)
        # print (y)
        # if x >= 0 and y >= 0:
        #     img = img[y:y + targetHeight, x:x + targetWidth]

        # tf.keras.preprocessing.image.save_img("/tmp/2.png", img)

        # cropped = img *255
        # cropped = tf.image.encode_png(cropped)
        # tf.io.write_file("/tmp/2.png",cropped)

        # img = tf.image.random_contrast(img, 0.7, 1.3)
        # img = tf.keras.preprocessing.image.random_rotation(img,10)
        # tf.keras.layers.experimental.preprocessing.
        # img = tfio.experimental.image.decode_tiff(img)
        # img = tf.image.rgb_to_grayscale(img)
        # img = tf.image.convert_image_dtype(img, tf.float32)
        # img = 1 - img * 2.
        # img = 1 - img
        # print (height)
        # print (width)
        # augment=True
        # print (img.shape)
        # image = np.expand_dims(image, -1)
        image = self.image_resize(image.astype('float32'), height=img_size[0])
        # img= tf.cast(img, dtype="int64")

        if False:
            if self.do_binarize_otsu:
                if image.shape[2] > 1:
                    image = tf.image.rgb_to_grayscale(image)
                image = image * 255
                image = self.otsu_thresholding(image)
                # img = self.sauvola(img.numpy())
                # print (img)
                # img = tf.cast(img, tf.int32)
                # img = img * 255
            if self.do_binarize_sauvola:
                if image.shape[2] > 1:
                    image = tf.image.rgb_to_grayscale(image)
                image = self.sauvola(image.numpy())
            image = tf.image.resize(image, [img_size[0], 65536], preserve_aspect_ratio=True)

        multiplier = 1.0
        if augment:
            rotation_range = 5
            randombrightness = 1 - random.uniform(0.85, 1.15)
            randomhue = 1 - random.uniform(0.98, 1.02)
            # randomSaturation = random.uniform(0.9, 1.1)
            randomRotate = random.uniform(-rotation_range * (2 * np.pi / 360), rotation_range * (2 * np.pi / 360))
            # randomShear = random.uniform(-20, 20)

            # img = tf.image.adjust_brightness(img, randombrightness)
            # img = tf.image.adjust_hue(img, randomhue)
            # img = tf.image.adjust_saturation(img, randomSaturation)
            if random.uniform(0, 1) < 0.25:
                image = tfa.image.rotate(image, randomRotate, fill_mode='constant', fill_value=0)

            # usage of the following random functions below causes a memory leak in TF:
            # https://www.gitmemory.com/issue/tensorflow/tensorflow/36164/594116192
            # the code above suffers less from leaks (but still does)
            # img = tf.image.random_brightness(img, 0.15)
            # img = tf.image.random_hue(img, 0.02)
            # img = tf.image.random_saturation(img, 0.9, 1.1)
            # if random.uniform(0, 1) < 0.25:
            #     img = tf.keras.preprocessing.image.random_rotation(img, 20)
            # img = tf.keras.preprocessing.image.random_shear(img) #Do NOT use for DIFOR
            # img = tf.keras.preprocessing.image.random_shift(img, 0.2, 0.2)
            # img = tf.keras.preprocessing.image.random_zoom(img, 0.9, row_axis=0, col_axis=1, channel_axis=2)
            multiplier = 0.5 + random.uniform(0, 1)

        if len(image.shape) == 2:
            image = np.expand_dims(image, -1)
        if self.random_crop:
            # print('cropping')
            random_crop = tf.random.uniform(shape=[1], minval=0.8, maxval=1.0)[0]
            original_width = tf.cast(tf.shape(image)[1], tf.float32)
            original_height = tf.cast(tf.shape(image)[0], tf.float32)

            crop_height = tf.cast(random_crop * original_height, tf.int32)
            randomseed = random.randint(0, 100000), random.randint(0, 1000000)
            random_crop = tf.random.uniform(shape=[1], minval=0.5, maxval=1.0)[0]
            crop_width = tf.cast(random_crop * original_width, tf.int32)
            crop_size = (crop_height, crop_width, channels)
            image = tf.image.stateless_random_crop(image, crop_size, randomseed)
            height = image.shape[0]
            width = image.shape[1]
            scale = img_size[0] / height
            newWidth = tf.cast(width * scale, tf.int32)
            # image = tf.image.resize(image, [imgSize[0], newWidth])
            image = self.image_resize(image.astype('float32'), height=img_size[0])
            # print (image)
            # image = image.astype('uint8')
        else:
            image = self.image_resize(image, height=img_size[0])
            if len(image.shape) == 2:
                image = np.expand_dims(image, -1)
            image = image * 255
            image = image.astype('uint8')
            # gtImageEncoded = tf.image.encode_png(image)
            # # gtImageEncoded = tf.image.encode_png(img)
            # tf.io.write_file("/tmp/test-" + str(id) + ".png", gtImageEncoded)

        # gtImageEncoded = tf.image.encode_png(tf.image.convert_image_dtype(image, dtype=tf.uint8))
        # # gtImageEncoded = tf.image.encode_png(img)
        # tf.io.write_file("/tmp/test-"+str(id)+".png", gtImageEncoded)
        # print(image)
        # gtImageEncoded = tf.image.encode_png(image)
        # # print(gtImageEncoded)
        # # gtImageEncoded = tf.image.encode_png(img)
        # tf.io.write_file("/tmp/test-"+str(id)+".png", gtImageEncoded)
        image = image/255
        # print(image.shape)

        if self.center_zero:
            image -= 0.5
        # print(image)
        return image

    def image_resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        # return the resized image
        return resized

    def read_single_lmdb(self, image_id):
        """ Reads a single image from LMDB.
            Parameters:
            ---------------
            image_id    integer unique ID for image

            Returns:
            ----------
            image       image array
        """

        # Start a new read transaction
        with self.env.begin() as txn:
            # Encode the key the same way as we stored it
            data = txn.get(f"{image_id:08}".encode("ascii"))
            # Remember it's a CIFAR_Image object that is loaded
            image = pickle.loads(data)
            # Retrieve the relevant bits
            # image = image.get_image()
            # label = cifar_image.label

        return image #, label

    def store_single_lmdb(self, image, image_id):
        """ Stores a single image to a LMDB.
            Parameters:
            ---------------
            image       image array, (32, 32, 3) to be stored
            image_id    integer unique ID for image
            label       image label
        """
        map_size = 1024 * 1024 * 1024 * 128

        # Create a new LMDB environment
        env = lmdb.open(str(self.lmdb_dir / f"single_lmdb"), map_size=map_size)

        # Start a new write transaction
        with env.begin(write=True) as txn:
            # All key-value pairs need to be strings
            value = image
            key = f"{image_id:08}"
            txn.put(key.encode("ascii"), pickle.dumps(value))
        env.close()

    def write_lmdb(self):
        self.lmdb_dir.mkdir(parents=True, exist_ok=True)



    def getImage(self, imgPath, imgSize, channels) -> np.ndarray:
        if imgPath is None:
            img = np.zeros([imgSize[0], imgSize[1]], dtype=float)
            print("Image broken, zeroing")
            return img

        if channels == 1:
            img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        elif channels == 3:
            img = cv2.imread(imgPath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        elif channels == 4:
            img = cv2.imread(imgPath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        return img

    def sauvola(self, image, k, as_tensor=True):
        thresh_sauvola = threshold_sauvola(image, window_size=self.sauvola_window, k=k)
        binary_sauvola = np.invert(image > thresh_sauvola)*255
        if as_tensor:
            return tf.convert_to_tensor(binary_sauvola)
        else:
            return binary_sauvola

    # https://colab.research.google.com/drive/1CdVfa2NlkQBga1E9dBwHved36Tk7Bg61#scrollTo=Jw-NU1wbHnWA
    def otsu_thresholding(self, image):
        image = tf.convert_to_tensor(image, name="image")
        rank = image.shape.rank
        if rank != 2 and rank != 3:
            raise ValueError("Image should be either 2 or 3-dimensional.")

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

        final = tf.where(image > threshold, 0, 1)
        return final

    # https://colab.research.google.com/drive/1CdVfa2NlkQBga1E9dBwHved36Tk7Bg61#scrollTo=Jw-NU1wbHnWA
    def adaptive_thresholding(self, image):
        image = tf.convert_to_tensor(image, name="image")
        window = 40
        rank = image.shape.rank
        if rank != 2 and rank != 3:
            raise ValueError("Image should be either 2 or 3-dimensional.")

        if not isinstance(window, int):
            raise ValueError("Window size value must be an integer.")
        r, c, channels = image.shape
        if window > min(r, c):
            raise ValueError("Window size should be lesser than the size of the image.")

        if rank == 3:
            image = tf.image.rgb_to_grayscale(image)
            image = tf.squeeze(image, 2)

        image = tf.image.convert_image_dtype(image, tf.dtypes.float32)

        i = 0
        final = tf.zeros((r, c))
        while i < r:
            j = 0
            r1 = min(i + window, r)
            while j < c:
                c1 = min(j + window, c)
                cur = image[i:r1, j:c1]
                thresh = tf.reduce_mean(cur)
                new = tf.where(cur > thresh, 255.0, 0.0)

                s1 = [x for x in range(i, r1)]
                s2 = [x for x in range(j, c1)]
                X, Y = tf.meshgrid(s2, s1)
                ind = tf.stack([tf.reshape(Y, [-1]), tf.reshape(X, [-1])], axis=1)

                final = tf.tensor_scatter_nd_update(final, ind, tf.reshape(new, [-1]))
                j += window
            i += window
        final = tf.expand_dims(final, -1)
        return final

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        (X, Y), y = self.__data_generation(indexes)

        return (X, Y), y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        """Generates data containing batch_size samples"""
        # Initialization
        X = []
        Y = []
        y = np.empty(self.batch_size, dtype=int)
        Z = []
        maxWidthFirst = 0
        maxWidthSecond = 0
        maxHeightFirst = 0
        maxHeightSecond = 0
        # Generate data
        for i, k in enumerate(indexes):
            # Store sample
            # X[i,] = np.load('data/' + ID + '.npy')
            imgPath = self.list_IDs[k]
            # X[i,] = self.preprocess(ID, self.imgSize)
            X.append(self.preprocess(k, imgPath, self.img_size, self.channels, self.augment))
            if X[i].shape[0] > maxHeightFirst:
                maxHeightFirst = X[i].shape[0]
            if X[i].shape[1] > maxWidthFirst:
                maxWidthFirst = X[i].shape[1]

            label = self.labels[imgPath]

            if random.randint(0, 1) == 1:
                index = self.labels.values()
                results = [i for i, search in enumerate(index) if search != str(label)]
                result = random.sample(results, 1)[0]
                imgPath = list(self.labels)[result]
                negImage = self.preprocess(result, imgPath, self.img_size, self.channels, self.augment)
                Y.append(negImage)
                y[i] = 1
            else:
                index = self.labels.values()
                results = [i for i, search in enumerate(index) if search == str(label)]
                result = random.sample(results, 1)[0]
                imgPath = list(self.labels)[result]
                posImage = self.preprocess(result, imgPath, self.img_size, self.channels, self.augment)
                Y.append(posImage)
                y[i] = 0
            if Y[i].shape[0] > maxHeightSecond:
                maxHeightSecond = Y[i].shape[0]
            if Y[i].shape[1] > maxWidthSecond:
                maxWidthSecond = Y[i].shape[1]
            Z.append(((X, Y), y))
        for i, ID in enumerate(indexes):
            X[i] = tf.image.resize_with_pad(X[i], maxHeightFirst, maxWidthFirst)
            Y[i] = tf.image.resize_with_pad(Y[i], maxHeightSecond, maxWidthSecond)

        X = tf.convert_to_tensor(X)
        Y = tf.convert_to_tensor(Y)
        y = tf.convert_to_tensor(y)
        return [X, Y], y
