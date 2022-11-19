import os

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from dataset_generic import DatasetGeneric

os.environ['TF_DETERMINISTIC_OPS'] = '1'

from siamese_network import *
from config import *
from utils import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
import random
import argparse
import metrics
from keras.utils.generic_utils import get_custom_objects

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--seed', metavar='seed', type=int, default=43,
                    help='random seed to be used')
parser.add_argument('--gpu', metavar='gpu', type=int, default=0,
                    help='GPU to be used. Use -1 for CPU')
parser.add_argument('--percent_validation', metavar='percent_validation', type=float, default=0.15,
                    help='percent_validation to be used')
parser.add_argument('--learning_rate', metavar='learning_rate', type=float, default=0.0001,
                    help='learning_rate to be used')
parser.add_argument('--epochs', metavar='epochs', type=int, default=40,
                    help='Epochs to be used. Default 40')
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
parser.add_argument('--spec', metavar='spec ', type=str, default='Cl11,11,32 Mp3,3 Cl7,7,64 Gm',
                    help='spec')
parser.add_argument('--existing_model', metavar='existing_model ', type=str, default=None,
                    help='existing_model')
parser.add_argument('--model_name', metavar='model_name ', type=str, default='difornet10',
                    help='model_name')
parser.add_argument('--loss', metavar='loss ', type=str, default="contrastive_loss",
                    help='contrastive_loss, binary_crossentropy, mse')
parser.add_argument('--optimizer', metavar='optimizer ', type=str, default='adam',
                    help='optimizer: adam, adadelta, rmsprop, sgd')
parser.add_argument('--memory_limit', metavar='memory_limit ', type=int, default=0,
                    help='memory_limit for gpu. Default 0 meaning use any available')
parser.add_argument('--dataset', metavar='dataset ', type=str, default='custom',
                    help='dataset. ecodices or iisg or cvl')
parser.add_argument('--train', metavar='train ', type=str,
                    help='file to use for training')
parser.add_argument('--val', metavar='val ', type=str,
                    help='file to use for validation')
parser.add_argument('--test', metavar='test ', type=str,
                    help='file to use for testing')

parser.add_argument('--do_binarize_otsu', action='store_true',
                    help='prefix to use for testing')
parser.add_argument('--do_binarize_sauvola', action='store_true',
                    help='do_binarize_sauvola')
parser.add_argument('--weights_path', metavar='weights_path', type=str,
                    help='path to store the weights of the trained model')
parser.add_argument('--num_workers', metavar='num_workers ', type=int, default=16,
                    help='num_workers')
parser.add_argument('--max_queue_size', metavar='max_queue_size ', type=int, default=64,
                    help='max_queue_size')
parser.add_argument('--center_zero', help='center_zero: beta, only implemented for cvl dataset', action='store_true')
parser.add_argument('--early_stopping', metavar='early_stopping ', type=int, default=3,
                    help='early_stopping')
parser.add_argument('--sauvola_window', metavar='sauvola_window ', type=int, default=11,
                    help='sauvola_window')
parser.add_argument('--sauvola_k', metavar='sauvola_k ', type=float, default=0.2,
                    help='sauvola_k')

parser.add_argument('--minimum_width', metavar='minimum_width ', type=int, default=51,
                    help='minimum_width')
parser.add_argument('--dropout', metavar='dropout', type=float, default=0.0,
                    help='dropout rate to be used')
parser.add_argument('--dropout_dense', metavar='dropout_dense', type=float, default=0.0,
                    help='dropout_dense rate to be used')
parser.add_argument('--batch_normalization', action='store_true',
                    help='batch_normalization')
parser.add_argument('--dense_layers', metavar='dense_layers ', type=int, default=2,
                    help='dense_layers')
parser.add_argument('--dense_units', metavar='dense_units ', type=int, default=256,
                    help='dense_units')
parser.add_argument('--random_crop', help='random crop', action='store_true')
parser.add_argument('--use_existing_lmdb', help='use_existing_lmdb', action='store_true')
parser.add_argument('--use_float32', help='use_float32', action='store_true')
parser.add_argument('--decay_steps', metavar='decay_steps ', type=int, default=25000,
                    help='decay_steps')


args = parser.parse_args()

imgSize = (args.height, args.width, args.channels)

MODEL_PATH = os.path.sep.join([args.output, args.model_name])
WEIGHTS_PATH = args.weights_path if args.weights_path is not None else os.path.sep.join(
    [args.output, args.model_name + "_weights"])
PLOT_PATH = os.path.sep.join([args.output, args.model_name, "plot.png"])

random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

if not args.use_float32:
    print("using mixed_float16")
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
else:
    print("using float32")
    policy = tf.keras.mixed_precision.Policy('float32')
    tf.keras.mixed_precision.set_global_policy(policy)
    tf.keras.backend.set_floatx('float32')

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
                                                    random_crop=args.random_crop,
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

# TODO check if data generators are set

get_custom_objects().update({"contrastive_loss": metrics.contrastive_loss})
get_custom_objects().update({"accuracy": metrics.accuracy})
get_custom_objects().update({"average": metrics.average})

# configure the siamese network
print("[INFO] building siamese network...")
imgA = Input(shape=(None, None, imgSize[2]))
imgB = Input(shape=(None, None, imgSize[2]))
featureExtractor=None
if (args.model_name == "siamese_model"):
    featureExtractor = siamese_network.build_siamese_model(imgSize, args.spec)
if (args.model_name == "signet"):
    featureExtractor = siamese_network.build_signet(imgSize)
if (args.model_name == "difornetA"):
    featureExtractor = siamese_network.build_difornetA(imgSize)
if (args.model_name == "difornetB"):
    featureExtractor = siamese_network.build_difornetB(imgSize)
if (args.model_name == "difornetC"):
    featureExtractor = siamese_network.build_difornetC(imgSize)
if (args.model_name == "difornetD"):
    featureExtractor = siamese_network.build_difornetD(imgSize)
if (args.model_name == "difornetE"):
    featureExtractor = siamese_network.build_difornetE(imgSize)
if (args.model_name == "difornetF"):
    featureExtractor = siamese_network.build_difornetF(imgSize)
if (args.model_name == "difornetG"):
    featureExtractor = siamese_network.build_difornetG(imgSize)

if (args.model_name == "difornet"):
    featureExtractor = siamese_network.build_difornet(imgSize)
if (args.model_name == "difornet2"):
    featureExtractor = siamese_network.build_difornet2(imgSize)
if (args.model_name == "difornet3"):
    featureExtractor = siamese_network.build_difornet3(imgSize)
if (args.model_name == "difornet3_dropout"):
    featureExtractor = siamese_network.build_difornet3_dropout(imgSize)
if (args.model_name == "difornet3_autoencoded"):
    featureExtractor = siamese_network.build_difornet3_autoencoded(imgSize)
if (args.model_name == "difornet4"):
    featureExtractor = siamese_network.build_difornet4(imgSize)
if (args.model_name == "difornet5"):
    featureExtractor = siamese_network.build_difornet5(imgSize)
if (args.model_name == "difornet6"):
    featureExtractor = siamese_network.build_difornet6(imgSize)  # runs OOM on 2070 Super
if (args.model_name == "difornet7"):
    featureExtractor = siamese_network.build_difornet7(imgSize)
if (args.model_name == "difornet8"):
    featureExtractor = siamese_network.build_difornet8(imgSize)
if (args.model_name == "difornet9"):
    featureExtractor = siamese_network.build_difornet9(imgSize)
if (args.model_name == "difornet10"):
    featureExtractor = siamese_network.build_difornet10(imgSize)
if (args.model_name == "difornet11"):
    featureExtractor = siamese_network.build_difornet11(imgSize)
if (args.model_name == "difornet12"):
    featureExtractor = siamese_network.build_difornet12(imgSize)
if (args.model_name == "difornet13"):
    featureExtractor = siamese_network.build_difornet13(imgSize)
if (args.model_name == "difornet14"):
    featureExtractor = siamese_network.build_difornet14(imgSize)
if (args.model_name == "difornet15"):
    featureExtractor = siamese_network.build_difornet15(imgSize)
if (args.model_name == "difornet16"):
    featureExtractor = siamese_network.build_difornet16(imgSize)
if (args.model_name == "difornet17"):
    featureExtractor = siamese_network.build_difornet17(imgSize)
if (args.model_name == "difornet18"):
    featureExtractor = siamese_network.build_difornet18(imgSize)
if (args.model_name == "difornet19"):
    featureExtractor = siamese_network.build_difornet19(imgSize)
if (args.model_name == "difornet20"):
    featureExtractor = siamese_network.build_difornet20(imgSize)
if (args.model_name == "difornet21"):
    featureExtractor = siamese_network.build_difornet21(imgSize)
if (args.model_name == "difornet22"):
    featureExtractor = siamese_network.build_difornet22(imgSize)
if (args.model_name == "difornet23"):
    featureExtractor = siamese_network.build_difornet23(imgSize)
if (args.model_name == "difornet24RK"):
    featureExtractor = siamese_network.build_difornet24RK(imgSize)
if (args.model_name == "difornet24"):
    featureExtractor = siamese_network.build_difornet24(imgSize)
if (args.model_name == "difornet25"):
    featureExtractor = siamese_network.build_difornet25(imgSize)
if (args.model_name == "difornet26"):
    featureExtractor = siamese_network.build_difornet26(imgSize)
if (args.model_name == "difornet27"):
    featureExtractor = siamese_network.build_difornet27(imgSize)
if (args.model_name == "difornet28"):
    featureExtractor = siamese_network.build_difornet28(imgSize)
if (args.model_name == "difornet29"):
    featureExtractor = siamese_network.build_difornet29(imgSize)
if (args.model_name == "difornet30"):
    featureExtractor = siamese_network.build_difornet30(imgSize)
if (args.model_name == "difornet31"):
    featureExtractor = siamese_network.build_difornet31(imgSize)
if (args.model_name == "difornet32"):
    featureExtractor = siamese_network.build_difornet32(imgSize)
if (args.model_name == "difornet33"):
    featureExtractor = siamese_network.build_difornet33(imgSize)
if (args.model_name == "difornet34"):
    featureExtractor = siamese_network.build_difornet34(imgSize)
if (args.model_name == "difornet35"):
    featureExtractor = siamese_network.build_difornet35(imgSize)
if (args.model_name == "difornet36"):
    featureExtractor = siamese_network.build_difornet36(imgSize)
if (args.model_name == "difornet37"):
    featureExtractor = siamese_network.build_difornet37(imgSize)
if (args.model_name == "difornet38"):
    featureExtractor = siamese_network.build_difornet38(imgSize)
if (args.model_name == "difornet39"):
    featureExtractor = siamese_network.build_difornet39(imgSize)
if (args.model_name == "difornet40"):
    featureExtractor = siamese_network.build_difornet40(imgSize)
if (args.model_name == "difornet41"):
    featureExtractor = siamese_network.build_difornet41(imgSize)
if (args.model_name == "difornet42"):
    featureExtractor = siamese_network.build_difornet42(imgSize)
if (args.model_name == "difornet42a"):
    featureExtractor = siamese_network.build_difornet42a(imgSize)
if (args.model_name == "difornet43"):
    featureExtractor = siamese_network.build_difornet43(imgSize)
if (args.model_name == "difornet43a"):
    featureExtractor = siamese_network.build_difornet43a(imgSize)
if (args.model_name == "difornet44"):
    featureExtractor = siamese_network.build_difornet44(imgSize)
if (args.model_name == "difornet44a"):
    featureExtractor = siamese_network.build_difornet44a(imgSize)
if (args.model_name == "difornet44b"):
    featureExtractor = siamese_network.build_difornet44b(imgSize)
if (args.model_name == "difornet44c"):
    featureExtractor = siamese_network.build_difornet44c(imgSize)
if (args.model_name == "difornet45"):
    featureExtractor = siamese_network.build_difornet45(imgSize)
if (args.model_name == "difornet45a"):
    featureExtractor = siamese_network.build_difornet45a(imgSize)
if (args.model_name == "difornet45b"):
    featureExtractor = siamese_network.build_difornet45b(imgSize)
if (args.model_name == "difornet45d"):
    featureExtractor = siamese_network.build_difornet45d(imgSize)
if (args.model_name == "difornet45e"):
    featureExtractor = siamese_network.build_difornet45e(imgSize)
if (args.model_name == "difornet45f"):
    featureExtractor = siamese_network.build_difornet45f(imgSize)
if (args.model_name == "difornet46"):
    featureExtractor = siamese_network.build_difornet46(imgSize)
if (args.model_name == "difornet47"):
    featureExtractor = siamese_network.build_difornet47(imgSize)
if (args.model_name == "difornet48"):
    featureExtractor = siamese_network.build_difornet48(imgSize)
if (args.model_name == "difornet49"):
    featureExtractor = siamese_network.build_difornet49(imgSize)
if (args.model_name == "difornet50"):
    featureExtractor = siamese_network.build_difornet50(imgSize)
if (args.model_name == "difornet51"):
    featureExtractor = siamese_network.build_difornet51(imgSize)
if (args.model_name == "difornet52a"):
    featureExtractor = siamese_network.build_difornet52a(imgSize)
if (args.model_name == "difornet52b"):
    featureExtractor = siamese_network.build_difornet52b(imgSize)
if (args.model_name == "difornet53"):
    featureExtractor = siamese_network.build_difornet53(imgSize)
if (args.model_name == "difornet54"):
    featureExtractor = siamese_network.build_difornet54(imgSize)
if (args.model_name == "difornet55"):
    featureExtractor = siamese_network.build_difornet55(imgSize, args.dropout, args.dropout_dense,
                                                        args.batch_normalization, args.dense_layers, args.dense_units)

if not args.existing_model:
    print("creating new model...")
    if featureExtractor is None:
        print("did you enter a non-existing modelname?")
        exit()
    featsA = featureExtractor(imgA)
    featsB = featureExtractor(imgB)

    # finally, construct the siamese network
    distance = Lambda(metrics.euclidean_distance)([featsA, featsB])
    model = Model(inputs=[imgA, imgB], outputs=distance)
else:
    print("loading existing model...")
    from keras.utils.generic_utils import get_custom_objects

    get_custom_objects().update({"contrastive_loss": metrics.contrastive_loss})
    get_custom_objects().update({"accuracy": metrics.accuracy})
    get_custom_objects().update({"average": metrics.average})
    model = keras.models.load_model(args.existing_model)

    test_single_network_arm = False
    if test_single_network_arm:
        submodel = model.get_layer(index=2)
        for layer in submodel.layers:
            if layer.name.startswith("conv") and layer.name.endswith("s"):
                print(layer.name)
                layer.trainable = True
        submodel = model.get_layer(index=2)
        for layer in submodel.layers:
            if layer.name.startswith("conv") and layer.name.endswith("s"):
                print(layer.name)
                layer.trainable = True
        submodel = model.get_layer(index=2)
        for layer in submodel.layers:
            if layer.name.startswith("conv") and not layer.name.endswith("s"):
                print(layer.name)
                layer.trainable = False
        submodel = model.get_layer(index=2)
        for layer in submodel.layers:
            if layer.name.startswith("conv") and not layer.name.endswith("s"):
                print(layer.name)
                layer.trainable = False

# select one of the siblings of the siamese net
submodel = model.get_layer(index=2)
print(submodel.summary())

model.summary()
# compile the model
print("[INFO] compiling model...")

accuracy = "accuracy"
loss = "mse"
# contrastive_loss: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
if args.loss == "contrastive_loss":
    loss = metrics.contrastive_loss_new
    accuracy = metrics.accuracy
if args.loss == "binary_crossentropy":
    loss = "binary_crossentropy"

optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate, nesterov=False)
if args.optimizer == "sgd":
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate, nesterov=False)
if args.optimizer == "adam":
    optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
if args.optimizer == "adadelta":
    optimizer = keras.optimizers.Adadelta(learning_rate=args.learning_rate, rho=0.95, epsilon=1e-07, name="Adadelta")
if args.optimizer == "rmsprop":
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=args.learning_rate)

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=args.learning_rate,
    decay_steps=args.decay_steps,
    decay_rate=0.99)

model.compile(loss=loss,
              optimizer=optimizer,
              metrics=[accuracy, metrics.average])

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

if False:
    correctCounter = 0
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    predict = model.predict(
        test_generator
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    sum = 0
    for i in range(len(predict)):
        # try:
        cutoff = 0.15998286
        sum = sum + predict[i]
        item = test_generator.__getitem__(i)
        firstImage = item[0][0]
        secondImage = item[0][1]
        y = item[1][0]
        correct = ((predict[i] > cutoff and y == 1) or (predict[i] <= cutoff and y == 0))
        if correct:
            correctCounter = correctCounter + 1
        print("%s\t%s\t%s\t%s" % (y, predict[i], correct, predict[i]))

        subplot_args = {'nrows': 2, 'ncols': 1, 'figsize': (9, 3),
                        'subplot_kw': {'xticks': [], 'yticks': []}}
        f, ax = plt.subplots(**subplot_args)
        ax[0].set_title(y, fontsize=14)
        ax[0].imshow(firstImage[0])
        ax[1].set_title(predict[i], fontsize=14)
        ax[1].imshow(secondImage[0])
        plt.tight_layout()
        plt.savefig('results/{}.png'.format(i))
        plt.close()
    print("correct {}".format(correctCounter))
    print("avg {}".format(sum / i))

    # except:
    #         print("An exception occurred")

if args.train:
    # train the model
    print("[INFO] training model...")
    earlyStopping = EarlyStopping(monitor='val_loss', patience=args.early_stopping, verbose=0, mode='min')
    mcp_save_best_val_loss = ModelCheckpoint('checkpoints/{}-best_val_loss/'.format(args.model_name),
                                             save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=5, verbose=1, min_delta=1e-4,
                                       mode='auto', cooldown=1)
    checkpoint_folder = "checkpoints-" + args.dataset

    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    filepath = checkpoint_folder + "/" + args.model_name + "-saved-model-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=False, mode='max')

    history = model.fit(
        training_generator,
        validation_data=validation_generator,
        callbacks=[earlyStopping, mcp_save_best_val_loss, reduce_lr_loss],
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_queue_size=args.max_queue_size,
        workers=args.num_workers
    )

    # serialize the model to disk
    print("[INFO] saving siamese model...")
    model.save(MODEL_PATH)
    # https://www.tensorflow.org/tutorials/keras/save_and_load#manually_save_weights
    model.save_weights(WEIGHTS_PATH)
    # plot the training history
    print("[INFO] plotting training history...")
    Utils.plot_training(history, PLOT_PATH)
