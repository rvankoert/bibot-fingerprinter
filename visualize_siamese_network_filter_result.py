import os
import metrics
from datagenerator import DataGenerator
from dataset_generic import DatasetGeneric
from config import *
from utils import *
import tensorflow.keras as keras
import tensorflow.keras.backend as K

import numpy as np
import tensorflow as tf
import random
import argparse
from matplotlib import pyplot as plt
import tensorflow_addons as tfa

# disable GPU for now, because it is already running on my dev machine
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_DETERMINISTIC_OPS'] = '1'

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--seed', metavar='seed', type=int, default=42,
                    help='random seed to be used')
parser.add_argument('--gpu', metavar='gpu', type=int, default=0,
                    help='gpu to be used')
parser.add_argument('--percent_validation', metavar='percent_validation', type=float, default=0.15,
                    help='percent_validation to be used')
parser.add_argument('--learning_rate', metavar='learning_rate', type=float, default=0.00001,
                    help='learning_rate to be used')
parser.add_argument('--epochs', metavar='epochs', type=int, default=40,
                    help='epochs to be used')
parser.add_argument('--batch_size', metavar='batch_size', type=int, default=1,
                    help='batch_size to be used, when using variable sized input this must be 1')

parser.add_argument('--height', metavar='height', type=int, default=51,
                    help='height to be used')
parser.add_argument('--width', metavar='width', type=int, default=751,
                    help='width to be used')
parser.add_argument('--channels', metavar='channels', type=int, default=3,
                    help='channels to be used')
parser.add_argument('--output', metavar='output', type=str, default='output',
                    help='base output to be used')
parser.add_argument('--trainset', metavar='trainset', type=str, default='/data/cvl-database-1-1/train.txt',
                    help='trainset to be used')
parser.add_argument('--testset', metavar='testset', type=str, default='/data/cvl-database-1-1/test.txt',
                    help='testset to be used')
parser.add_argument('--use_testset', metavar='use_testset', type=bool, default=False,
                    help='testset to be used')
parser.add_argument('--spec', metavar='spec ', type=str, default='Cl11,11,32 Mp3,3 Cl7,7,64 Gm',
                    help='spec')
parser.add_argument('--existing_model', metavar='existing_model ', type=str, default='',
                    help='existing_model')
parser.add_argument('--dataset', metavar='dataset ', type=str, default='ecodices',
                    help='dataset. ecodices or iisg')
parser.add_argument('--do_binarize_otsu', action='store_true',
                    help='prefix to use for testing')
parser.add_argument('--do_binarize_sauvola', action='store_true',
                    help='do_binarize_sauvola')
parser.add_argument('--train', metavar='train ', type=str,
                    help='file to use for training')
parser.add_argument('--val', metavar='val ', type=str,
                    help='file to use for validation')
parser.add_argument('--test', metavar='test ', type=str,
                    help='file to use for testing')
parser.add_argument('--sauvola_window', metavar='sauvola_window ', type=int, default=11,
                    help='sauvola_window')
parser.add_argument('--use_existing_lmdb', help='use_existing_lmdb', action='store_true')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

SEED = args.seed
GPU = args.gpu
PERCENT_VALIDATION = args.percent_validation
LEARNING_RATE = args.learning_rate
config.IMG_SHAPE = (args.height, args.width, args.channels)
config.BATCH_SIZE = args.batch_size
config.EPOCHS = args.epochs
config.BASE_OUTPUT = args.output

MODEL_PATH = os.path.sep.join([config.BASE_OUTPUT, "siamese_model"])
# MODEL_PATH = 'checkpoints/difornet14-saved-model-23-0.92.hdf5'
# MODEL_PATH = 'checkpoints/difornet15-saved-model-10-0.79.hdf5'
MODEL_PATH = 'checkpoints/difornet13-saved-model-89-0.93.hdf5'
MODEL_PATH = "checkpoints/difornet13-saved-model-68-0.94.hdf5"
MODEL_PATH = "checkpoints/difornet13-saved-model-49-0.94.hdf5"  # iisg
MODEL_PATH = "checkpoints/difornet14-saved-model-45-0.97.hdf5"
# MODEL_PATH = "checkpoints-iisg/difornet17-saved-model-44-0.92.hdf5"
MODEL_PATH = "checkpoints-iisg/difornet14-saved-model-19-0.94.hdf5"
MODEL_PATH = "checkpoints-iisg/difornet14-saved-model-98-0.97.hdf5"
MODEL_PATH = "checkpoints-iisg/difornet19-saved-model-19-0.94.hdf5"
MODEL_PATH = "checkpoints-iisg/difornet19-saved-model-128-0.95.hdf5"
MODEL_PATH = "checkpoints/difornet23-best_val_loss"
MODEL_PATH = "checkpoints/difornet24-best_val_loss"
MODEL_PATH = "checkpoints-iisg/difornetD-saved-model-07-0.95.hdf5"
MODEL_PATH = 'checkpoints/difornet48-best_val_loss/'
MODEL_PATH = 'checkpoints/difornet49-best_val_loss/'
MODEL_PATH = "checkpoints/difornet50-best_val_loss/"
MODEL_PATH = "checkpoints/difornet51-best_val_loss/"

if args.existing_model:
    MODEL_PATH = args.existing_model

PLOT_PATH = os.path.sep.join([config.BASE_OUTPUT, "plot.png"])

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# if GPU >= 0:
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     if (len(gpus) > 0):
#         tf.config.experimental.set_virtual_device_configuration(gpus[GPU], [
#             tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

imgSize = config.IMG_SHAPE
# print("[INFO] loading DiFor dataset...")

keras.losses.custom_loss = metrics.contrastive_loss
from keras.utils.generic_utils import get_custom_objects

get_custom_objects().update({"contrastive_loss": metrics.contrastive_loss})
get_custom_objects().update({"average": metrics.average})
get_custom_objects().update({"contrastive_loss_new": metrics.contrastive_loss_new})

model = keras.models.load_model(MODEL_PATH)

model.summary()

layer_name = "conv3_block4_out"

submodel = model.get_layer(index=2)
print(submodel.summary())

model.summary()

partition = {'train': [], 'validation': [], 'test': []}
trainLabels = {}
valLabels = {}
testLabels = {}

# find /home/rutger/ecodices-data/ -name '*.jpg' -exec echo {} 0 \; > /home/rutger/ecodices-data/all.txt
# split train/val:
# shuf all.txt | split -l $(( $(wc -l <all.txt) * 80 / 100 )); mv xab val.txt; mv xaa train.txt

if args.dataset == 'cvl':
    training_generator, validation_generator, test_generator = DatasetCVL().generators(args.channels,
                                                                                       args.do_binarize_otsu,
                                                                                       args.do_binarize_sauvola)
if args.dataset == 'iisg':
    training_generator, validation_generator, test_generator = DatasetIISG().generators(args.channels, args.do_binarize_otsu,
                                                                                                           args.do_binarize_sauvola)
if args.dataset == 'ecodices':
    training_generator, validation_generator, test_generator = DatasetEcodices().generators(args.channels, args.do_binarize_otsu,
                                                                                                           args.do_binarize_sauvola)
if args.dataset == 'medieval':
    training_generator, validation_generator, test_generator = DatasetMedieval().generators(args.channels,args.do_binarize_otsu, args.do_binarize_sauvola)
if args.dataset == 'medieval_small':
    training_generator, validation_generator, test_generator = DatasetMedieval30Percent().generators(args.channels,
                                                                                                     args.do_binarize_otsu,
                                                                                                     args.do_binarize_sauvola)
if args.dataset == 'medieval_small_sample':
    training_generator, validation_generator, test_generator = DatasetMedieval30PercentSample().generators(args.channels,
                                                                                                           args.do_binarize_otsu,
                                                                                                           args.do_binarize_sauvola)

if args.dataset == 'place_century_script':
    training_generator, validation_generator, test_generator = DatasetPlaceCenturyScript().generators(args.channels,
                                                                                                      args.do_binarize_otsu, args.do_binarize_sauvola)

if args.train:
    training_generator = DatasetGeneric().generator(args.train,
                                                    batch_size=args.batch_size,
                                                    channels=args.channels,
                                                    do_binarize_otsu=args.do_binarize_otsu,
                                                    do_binarize_sauvola=args.do_binarize_sauvola,
                                                    sauvola_window=args.sauvola_window,
                                                    cache_path='data/lmdb/train',
                                                    center_zero=args.center_zero,
                                                    use_existing_lmdb=args.use_existing_lmdb)

if args.val:
    validation_generator = DatasetGeneric().generator(args.val,
                                                      batch_size=args.batch_size,
                                                      channels=args.channels,
                                                      do_binarize_otsu=args.do_binarize_otsu,
                                                      do_binarize_sauvola=args.do_binarize_sauvola,
                                                      sauvola_window=args.sauvola_window,
                                                      cache_path='data/lmdb/train',
                                                      center_zero=args.center_zero,
                                                      use_existing_lmdb=args.use_existing_lmdb)

if args.test:
    test_generator = DatasetGeneric().generator(args.test,
                                                batch_size=args.batch_size,
                                                channels=args.channels,
                                                do_binarize_otsu=args.do_binarize_otsu,
                                                do_binarize_sauvola=args.do_binarize_sauvola,
                                                sauvola_window=args.sauvola_window,
                                                cache_path='data/lmdb/train',
                                                center_zero=args.center_zero,
                                                use_existing_lmdb=args.use_existing_lmdb)

def compute_loss(input_image, filter_index):
    activation = feature_extractor(input_image)
    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)


# @tf.function
def gradient_ascent_step(img, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return loss, img


def visualize_filter(filter_index, channels):
    # We run gradient ascent for 20 steps
    iterations = 30
    learning_rate = 10.0
    utils = Utils()
    img = utils.initialize_image(channels)
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate)

    # Decode the resulting input image
    img = utils.deprocess_image(img[0].numpy())
    return loss, img


# 1 3 5 6
for layerId in range(len(submodel.layers)):
    layer = submodel.layers[layerId]
    if not layer.name.startswith("conv") and not layer.name.startswith("add"):
        continue
    feature_extractor = keras.Model(inputs=submodel.inputs, outputs=layer.output)
    # feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)

    all_imgs = []
    numFilters = layer.output_shape[3]
    # numFilters = 6
    i = 0
    for filter_index in range(numFilters):
        print("Processing filter %d" % (filter_index,))
        loss, img = visualize_filter(filter_index, args.channels)

        all_imgs.append(img)

    utils = Utils()
    while i < 10:
        item = test_generator.__getitem__(i)

        i = i + 1

        X = item
        # Rendering
        img1 = tf.keras.preprocessing.image.array_to_img(K.squeeze(X[0][0], axis=-0))

        maps = utils.get_feature_maps(submodel, layerId, img1)

        fig = plt.figure(figsize=(40, numFilters * 2))
        columns = 2
        rows = numFilters

        # ax enables access to manipulate each of subplots
        ax = []

        for j in range(numFilters):
            img = all_imgs[j - 1]
            # create subplot and append to ax
            ax.append(fig.add_subplot(rows, columns, j * 2 + 1))
            ax[-1].set_title("ax:" + str(j))  # set title
            if args.channels == 1:
                img = tf.squeeze(img)
            plt.imshow(img, cmap='gray')
            ax.append(fig.add_subplot(rows, columns, j * 2 + 2))
            # ax[-1].set_title("ax:" + str(j))  # set title
            if args.channels == 1:
                maps[j - 1] = tf.squeeze(maps[j - 1])
            plt.imshow(maps[j - 1], cmap='gray')

        # plt.show()  # finally, render the plot

        # plt.show()
        plt.tight_layout()
        plt.savefig('results/{}-{}.png'.format(layer.name, i))
        plt.close()
