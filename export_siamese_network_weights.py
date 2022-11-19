import os

import sys


from directory_generator import DirectoryGenerator

os.environ['TF_DETERMINISTIC_OPS'] = '1'

from config import *
import numpy as np
import random
import argparse
import metrics
from siamese_network import *

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
parser.add_argument('--results_file', default='/tmp/test/result.csv')
parser.add_argument("--weights_path", required=True)
args = parser.parse_args()

config.SEED = args.seed
config.GPU = args.gpu
# config.GPU = -1
config.LEARNING_RATE = args.learning_rate
config.IMG_SHAPE = (args.height, args.width, args.channels)
config.BATCH_SIZE = args.batch_size
config.EPOCHS = args.epochs
config.BASE_OUTPUT = args.output
config.SPEC = args.spec
config.MODEL_NAME = args.model_name
config.EXISTING_MODEL = args.existing_model
config.OPTIMIZER = args.optimizer
config.LOSS = args.loss

RESULTS_FILE = args.results_file
WEIGHTS_PATH = args.weights_path
MODEL_PATH = os.path.sep.join([config.BASE_OUTPUT, config.MODEL_NAME])
PLOT_PATH = os.path.sep.join([config.BASE_OUTPUT, config.MODEL_NAME, "plot.png"])

random.seed(config.SEED)
np.random.seed(config.SEED)
tf.random.set_seed(config.SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.GPU)
if config.GPU >= 0:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if (len(gpus) > 0):
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=args.memory_limit)])

imgSize = config.IMG_SHAPE
if args.dir:
    test_generator = DirectoryGenerator().testGenerator(args.dir, args.channels, args.do_binarize_otsu, args.do_binarize_sauvola)
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

# model = keras.models.load_model(args.existing_model, custom_objects=get_custom_objects())
model = siamese_network.build_difornet40(config.IMG_SHAPE)
# https://www.tensorflow.org/tutorials/keras/save_and_load#manually_save_weights
model.load_weights(WEIGHTS_PATH)

# select one of the siblings of the siamese net
# submodel = model.get_layer(index=2)
# print(submodel.summary())

model.summary()
# compile the model
print("[INFO] compiling model...")

intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.layers[-1].output)

print("seed: {}".format(config.SEED))
print("gpu: {}".format(config.GPU))
print("learning_rate: {}".format(config.LEARNING_RATE))
print("IMG_SHAPE: {}".format(config.IMG_SHAPE))
print("batch_size: {}".format(config.BATCH_SIZE))
print("epochs: {}".format(config.EPOCHS))
print("BASE_OUTPUT: {}".format(config.BASE_OUTPUT))
print("spec: {}".format(config.SPEC))
print("model_name: {}".format(config.MODEL_NAME))
print("existing_model: {}".format(config.EXISTING_MODEL))
print("optimizer: {}".format(config.OPTIMIZER))
print("loss: {}".format(config.LOSS))
print("results_file: ", RESULTS_FILE)

np.set_printoptions(threshold=sys.maxsize)

correctCounter = 0
if True:
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    tf.random.set_seed(config.SEED)
    predict = intermediate_layer_model.predict(
        test_generator
    )
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    tf.random.set_seed(config.SEED)
    ### as csv
    results_file = open(RESULTS_FILE, "w")

    for i in range(len(predict)):
        label = test_generator.getLabel(i)
        result = np.asarray([predict[i]])
        # print(result)
        # print (label)

        results_file.write(label + ",")
        np.savetxt(results_file, result, newline='\n', delimiter=',')
        # print(predict[i])

    results_file.close()
