import os

import sys
from math import ceil

from directory_generator import DirectoryGenerator

os.environ['TF_DETERMINISTIC_OPS'] = '1'

from config import *
from tensorflow.keras.models import Model
import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
import random
import argparse
import metrics
from progress.bar import Bar

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--seed', metavar='seed', type=int, default=43,
                    help='random seed to be used')
parser.add_argument('--gpu', metavar='gpu', type=int, default=0,
                    help='gpu to be used')
parser.add_argument('--percent_validation', metavar='percent_validation', type=float, default=0.15,
                    help='percent_validation to be used')
parser.add_argument('--learning_rate', metavar='learning_rate', type=float, default=0.0001,
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
parser.add_argument('--existing_model', metavar='existing_model ', type=str,
                     default='checkpoints-iisg/existing_difornetG_e05-saved-model-93-0.97.hdf5',
                    help='existing_model')
parser.add_argument('--model_name', metavar='model_name ', type=str, default='difornet10',
                    help='model_name')
parser.add_argument('--loss', metavar='loss ', type=str, default="contrastive_loss",
                    help='contrastive_loss, binary_crossentropy, mse')
parser.add_argument('--optimizer', metavar='optimizer ', type=str, default='adam',
                    help='optimizer: adam, adadelta, rmsprop, sgd')
parser.add_argument('--memory_limit', metavar='memory_limit ', type=int, default=4096,
                    help='memory_limit for gpu. Default 4096')
parser.add_argument('--dataset', metavar='dataset ', type=str, default='ecodices',
                    help='dataset. ecodices or iisg')
parser.add_argument('--dir', metavar='dir ', type=str,
                    help='directory to used as input. It should contain snippets')
parser.add_argument('--do_binarize_otsu', action='store_true',
                    help='prefix to use for testing')
parser.add_argument('--do_binarize_sauvola', action='store_true',
                    help='do_binarize_sauvola')
parser.add_argument('--sauvola_window', metavar='sauvola_window ', type=int, default=11,
                    help='sauvola_window')

args = parser.parse_args()

imgSize = (args.height, args.width, args.channels)


MODEL_PATH = os.path.sep.join([args.output, args.model_name])
PLOT_PATH = os.path.sep.join([args.output, args.model_name, "plot.png"])

random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
if args.gpu >= 0:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if (len(gpus) > 0):
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=args.memory_limit)])

# do_binarize_otsu = False
# if args.do_binarize_otsu:
#     do_binarize_otsu = True
#
# do_binarize_sauvola = False
# if args.do_binarize_sauvola:
#     do_binarize_sauvola = True
if args.dir:
    test_generator = DirectoryGenerator().testGenerator(
        path=args.dir,
        do_binarize_otsu=args.do_binarize_otsu,
        do_binarize_sauvola=args.do_binarize_sauvola,
        height=args.height,
        width=args.width,
        channels=args.channels,
        sauvola_window=args.sauvola_window
    )
else:
    if args.dataset == 'iisg':
        training_generator, validation_generator, test_generator = DatasetIISG().generators()
    if args.dataset == 'ecodices':
        training_generator, validation_generator, test_generator = DatasetEcodices().generators()

print("loading existing model...")

from keras.utils.generic_utils import get_custom_objects

get_custom_objects().update({"contrastive_loss": metrics.contrastive_loss})
get_custom_objects().update({"accuracy": metrics.accuracy})
get_custom_objects().update({"average": metrics.average})
get_custom_objects().update({"contrastive_loss_new": metrics.contrastive_loss_new})

model = keras.models.load_model(args.existing_model, custom_objects=get_custom_objects())

# select one of the siblings of the siamese net
submodel = model.get_layer(index=2)
print(submodel.summary())

model.summary()
# compile the model
print("[INFO] compiling model...")

intermediate_layer_model = Model(inputs=submodel.input,
                                 outputs=submodel.layers[-1].output)

print("seed: {}".format(args.seed))
print("gpu: {}".format(args.gpu))
print("learning_rate: {}".format(args.learning_rate))
print("IMG_SHAPE: {}".format(imgSize))
print("batch_size: {}".format(args.batch_size))
print("epochs: {}".format(args.epochs))
print("BASE_OUTPUT: {}".format(args.output))
print("spec: {}".format(args.spec))
print("model_name: {}".format(args.model_name))
print("existing_model: {}".format(args.existing_model))
print("optimizer: {}".format(args.optimizer))
print("loss: {}".format(args.loss))

np.set_printoptions(threshold=sys.maxsize)

correctCounter = 0
if True:
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    ### as csv
    if not os.path.exist('/tmp/test/'):
        os.mkdir('/tmp/test/')
    results_file = open('/tmp/test/result.csv', 'w')
    total_elements = len(test_generator.labels)
    bar = Bar('Processing', max=len(test_generator.labels))

    for i in range(ceil(total_elements/args.batch_size)):
        batch = test_generator.__getitem__(i)
        predict = intermediate_layer_model.predict_on_batch(
            batch
        )
        for batch_item_no in range(len(predict)):
            label = test_generator.getLabel(i*args.batch_size+batch_item_no)
            result = [np.asarray(predict[batch_item_no])]
            bar.next()

            results_file.write(label + ",")
            np.savetxt(results_file, result, newline='\n', delimiter=',')
    bar.finish()


    results_file.close()
