import metrics
from config import *
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
import random
import argparse
from matplotlib import pyplot as plt
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
from keras.utils.generic_utils import get_custom_objects

from dataset_generic import DatasetGeneric

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
parser.add_argument('--dataset', metavar='dataset ', type=str, default='',
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
parser.add_argument('--sauvola_k', metavar='sauvola_k ', type=float, default=0.2,
                    help='sauvola_k')
parser.add_argument('--minimum_width', metavar='minimum_width ', type=int, default=51,
                    help='minimum_width')
parser.add_argument('--use_existing_lmdb', help='use_existing_lmdb', action='store_true')
parser.add_argument('--center_zero', help='center_zero: beta, only implemented for cvl dataset', action='store_true')

args = parser.parse_args()

SEED = args.seed
GPU = args.gpu
PERCENT_VALIDATION = args.percent_validation
LEARNING_RATE = args.learning_rate
config.IMG_SHAPE = (args.height, args.width, args.channels)
config.BATCH_SIZE = args.batch_size
config.EPOCHS = args.epochs
config.BASE_OUTPUT = args.output

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

if not args.existing_model:
    print('use --existing_model')
    exit()

PLOT_PATH = os.path.sep.join([config.BASE_OUTPUT, "plot.png"])

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

imgSize = config.IMG_SHAPE

get_custom_objects().update({"contrastive_loss": metrics.contrastive_loss})
get_custom_objects().update({"average": metrics.average})
get_custom_objects().update({"contrastive_loss_new": metrics.contrastive_loss_new})

model = keras.models.load_model(args.existing_model)

model.summary()

submodel = model.get_layer(index=2)
submodel.summary()

if args.train:
    training_generator = DatasetGeneric().generator(imgSize,
                                                    args.train,
                                                    batch_size=args.batch_size,
                                                    channels=args.channels,
                                                    do_binarize_otsu=args.do_binarize_otsu,
                                                    do_binarize_sauvola=args.do_binarize_sauvola,
                                                    sauvola_window=args.sauvola_window,
                                                    sauvola_k=args.sauvola_k,
                                                    minimum_width=args.minimum_width,
                                                    cache_path='data/lmdb/train',
                                                    center_zero=args.center_zero,
                                                    use_existing_lmdb=args.use_existing_lmdb)
if args.val:
    validation_generator = DatasetGeneric().generator(imgSize,
                                                      args.val,
                                                      batch_size=args.batch_size,
                                                      channels=args.channels,
                                                      do_binarize_otsu=args.do_binarize_otsu,
                                                      do_binarize_sauvola=args.do_binarize_sauvola,
                                                      sauvola_window=args.sauvola_window,
                                                      sauvola_k=args.sauvola_k,
                                                      minimum_width=args.minimum_width,
                                                      cache_path='data/lmdb/val',
                                                      center_zero=args.center_zero,
                                                      use_existing_lmdb=args.use_existing_lmdb)

if args.test:
    test_generator = DatasetGeneric().generator(imgSize,
                                                args.test,
                                                batch_size=args.batch_size,
                                                channels=args.channels,
                                                do_binarize_otsu=args.do_binarize_otsu,
                                                do_binarize_sauvola=args.do_binarize_sauvola,
                                                sauvola_window=args.sauvola_window,
                                                sauvola_k=args.sauvola_k,
                                                minimum_width=args.minimum_width,
                                                cache_path='data/lmdb/test',
                                                center_zero=args.center_zero,
                                                use_existing_lmdb=args.use_existing_lmdb)
i = 0

if args.batch_size > 1:
    print('use "--batch_size 1", not working yet with larger batches')
    exit()

def loss(output):
    # 1 is the imagenet index corresponding to Goldfish, 294 to Bear and 413 to Assault Rifle.
    # return (output[0][1], output[1][1], output[2][1])
    return output[0][0]


def model_modifier(model):
    model.layers[-1].activation = tf.keras.activations.linear
    return model

errors=0
while i < 1000:
    item = test_generator.__getitem__(i)

    # put sample into list
    i = i + 1

    X = item
    # Rendering

    saliency = Saliency(model,
                        model_modifier=model_modifier,
                        clone=False)
    predicted = model.predict(X[0])
    # Generate saliency map with smoothing that reduce noise by adding noise
    saliency_map2 = saliency(loss,
                             X[0],
                             smooth_samples=20,  # The number of calculating gradients iterations.
                             smooth_noise=0.20)  # noise spread level.
    saliency_map1 = normalize(saliency_map2[0])
    saliency_map2 = normalize(saliency_map2[1])

    subplot_args = {'nrows': 4, 'ncols': 1, 'figsize': (18, 6),
                    'subplot_kw': {'xticks': [], 'yticks': []}}
    f, ax = plt.subplots(**subplot_args)
    title = "actual: different"
    if X[1] == 0:
        title = "actual: same"
    if predicted[0][0] < 0.5:
        title = title + ", predicted: same"
    else:
        title = title + ", predicted: different"
    if X[1] == 0 and predicted[0][0] >= 0.5:
        errors += 1
    if X[1] == 1 and predicted[0][0] < 0.5:
        errors += 1
    ax[0].set_title(title, fontsize=14)
    img1 = tf.keras.preprocessing.image.array_to_img(K.squeeze(X[0][0], axis=-0))
    if args.channels==1:
        cmap='gray'
    else:
        cmap='jet'
    ax[0].imshow(img1, cmap)
    ax[1].imshow(saliency_map1[0], cmap='jet')
    img2 = tf.keras.preprocessing.image.array_to_img(K.squeeze(X[0][1], axis=-0))
    ax[2].set_title(predicted[0][0], fontsize=14)
    ax[2].imshow(img2, cmap)
    ax[3].imshow(saliency_map2[0], cmap='jet')
    plt.tight_layout()
    plt.savefig('results-saliency/{}.png'.format(i))
    plt.close()
    print('errors ' + str(errors)+'/' + str(i))
print('errors ' + str(errors))