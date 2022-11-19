from config import *
import metrics
from matplotlib import pyplot as plt
from tf_keras_vis.utils import num_of_gpus
import tensorflow.keras as keras
import tensorflow as tf
import argparse
from keras.utils.generic_utils import get_custom_objects

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_DETERMINISTIC_OPS'] = '1'

_, gpus = num_of_gpus()
print('Tensorflow recognized {} GPUs'.format(gpus))

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--seed', metavar='seed', type=int, default=42,
                    help='random seed to be used')
parser.add_argument('--gpu', metavar='gpu', type=int, default=0,
                    help='gpu to be used')
parser.add_argument('--height', metavar='height', type=int, default=51,
                    help='height to be used')
parser.add_argument('--width', metavar='width', type=int, default=751,
                    help='width to be used')
parser.add_argument('--channels', metavar='channels', type=int, default=3,
                    help='channels to be used')
parser.add_argument('--output', metavar='output', type=str, default='output',
                    help='base output to be used')
parser.add_argument('--existing_model', metavar='existing_model ', type=str, default='',
                    help='existing_model')
parser.add_argument('--use_float32', help='use_float32 crop', action='store_true')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

get_custom_objects().update({"contrastive_loss": metrics.contrastive_loss})
get_custom_objects().update({"accuracy": metrics.accuracy})
get_custom_objects().update({"average": metrics.average})
get_custom_objects().update({"contrastive_loss_new": metrics.contrastive_loss_new})

if not args.use_float32:
    print("using mixed_float16")
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
else:
    print("using float32")
    policy = tf.keras.mixed_precision.Policy('float32')
    tf.keras.mixed_precision.set_global_policy(policy)


if not args.existing_model:
    print("use --existing_model")
    exit()
tf.random.set_seed(args.seed)
model = keras.models.load_model(args.existing_model)
model.summary()
model = model.get_layer(index=2)
model.summary()
# input_layer = InputLayer(input_shape=(51, 751, 3), name="input_1")
# # model.input = input
# model.layers[0]= input_layer
#
# model = Model(inputs=model.input,
#                                  outputs=model.output)
# model.summary()

def model_modifier(cloned_model):
    cloned_model.layers[-2].activation = tf.keras.activations.linear
    return cloned_model

from tf_keras_vis.activation_maximization import ActivationMaximization

activation_maximization = ActivationMaximization(model,
                                                 model_modifier,
                                                 clone=False,
                                                 )

from tf_keras_vis.utils.scores import CategoricalScore


# Instead of CategoricalScore object, you can define the scratch function such as below:
def score_function(output):
    # The `output` variable refer to the output of the model,
    # so, in this case, `output` shape is `(3, 1000)` i.e., (samples, classes).
    return output[:, 20]

from tf_keras_vis.activation_maximization.callbacks import PrintLogger, Progress

for i in range(96):
    score = CategoricalScore(i)
    maxcolor = 255
    if args.channels == 1:
        maxcolor = 1
    seed_input = tf.random.uniform((51, 251, args.channels), 0, 1, dtype=tf.dtypes.float32)
    # print(seed_input)
    activations = activation_maximization(score,
                                          steps=1024,
                                          callbacks=[Progress()],
                                          seed_input=seed_input,
                                          input_range=(0.0, 1.0)
                                          )

    # Render
    f, ax = plt.subplots(figsize=(4, 4))
    if args.channels == 1:
        activation = tf.squeeze(activations[0])
        activation = activation * 255
        ax.imshow(activation, cmap='gray', vmin=0, vmax=255)
        # print(maxcolor)
    else:
        activation = activations[0]
        # if args.channels==4:
        #     activation = activation[:,:,:3]
        ax.imshow(activation)
    ax.set_title(i, fontsize=16)
    ax.axis('off')
    plt.tight_layout()
    # plt.show()
    plt.savefig('dense/dense-{}.png'.format(i))
