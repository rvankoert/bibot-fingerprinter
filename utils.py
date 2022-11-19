import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.models import Model
import matplotlib
matplotlib.use('Agg')

class Utils:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    # def make_pairs(images, labels):
    #     # initialize two empty lists to hold the (image, image) pairs and
    #     # labels to indicate if a pair is positive or negative
    #     pairImages = []
    #     pairLabels = []
    #     # calculate the total number of classes present in the dataset
    #     # and then build a list of indexes for each class label that
    #     # provides the indexes for all examples with a given label
    #     uniqueLabels = np.unique(labels)
    #     numClasses = len(uniqueLabels)
    #     idx = [np.where(labels == uniqueLabels[i])[0] for i in range(0, numClasses)]
    #     # loop over all images
    #     for idxA in range(len(images)):
    #         # grab the current image and label belonging to the current
    #         # iteration
    #         currentImage = tf.convert_to_tensor(images[idxA])
    #         label = labels[idxA]
    #         # randomly pick an image that belongs to the *same* class
    #         # label
    #         labelIndex = np.where(uniqueLabels == label)[0][0]
    #         idxB = np.random.choice(idx[labelIndex])
    #         posImage = tf.convert_to_tensor(images[idxB])
    #         # prepare a positive pair and update the images and labels
    #         # lists, respectively
    #         # tf.keras.preprocessing.image.save_img('posImage1.png', tf.keras.preprocessing.image.array_to_img(currentImage))
    #         # tf.keras.preprocessing.image.save_img('posImage2.png', tf.keras.preprocessing.image.array_to_img(posImage))
    #         pairImages.append([currentImage, posImage])
    #         pairLabels.append([1])
    #         # grab the indices for each of the class labels *not* equal to
    #         # the current label and randomly pick an image corresponding
    #         # to a label *not* equal to the current label
    #         negIdx = np.where(labels != label)[0]
    #         negImage = tf.convert_to_tensor(images[np.random.choice(negIdx)])
    #         # prepare a negative pair of images and update our lists
    #         # tf.keras.preprocessing.image.save_img('negImage1.png', tf.keras.preprocessing.image.array_to_img(currentImage))
    #         # tf.keras.preprocessing.image.save_img('negImage2.png', tf.keras.preprocessing.image.array_to_img(negImage))
    #         pairImages.append([currentImage, negImage])
    #         pairLabels.append([0])
    #     # return a 2-tuple of our image pairs and labels
    #     return (np.array(pairImages), np.array(pairLabels))

    def plot_training(H, plotPath):
        # construct a plot that plots and saves the training history
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(H.history["loss"], label="train_loss")
        plt.plot(H.history["val_loss"], label="val_loss")
        plt.plot(H.history["accuracy"], label="train_acc")
        plt.plot(H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(plotPath)

    def get_feature_maps(self, model, layer_id, input_image):
        model_ = Model(inputs=[model.input], outputs=[model.layers[layer_id].output])
        print(model.layers[layer_id].name)
        return model_.predict(np.expand_dims(input_image, axis=0))[0, :, :, :].transpose((2, 0, 1))

    def initialize_image(self, channels):
        # We start from a gray image with some random noise
        img = tf.random.uniform((1, 180, 180, channels))
        # ResNet50V2 expects inputs in the range [-1, +1].
        # Here we scale our random inputs to [-0.125, +0.125]
        return (img - 0.5) * 0.25

    def deprocess_image(self, img):
        # Normalize array: center on 0., ensure variance is 0.15
        img -= img.mean()
        img /= img.std() + 1e-5
        img *= 0.15

        # Center crop
        img = img[25:-25, 25:-25, :]

        # Clip to [0, 1]
        img += 0.5
        img = np.clip(img, 0, 1)

        # Convert to RGB array
        img *= 255
        img = np.clip(img, 0, 255).astype("uint8")
        return img
