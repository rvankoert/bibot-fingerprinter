import tensorflow as tf
from keras.layers import BatchNormalization
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Concatenate


class siamese_network:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def build_siamese_model(inputShape, spec):
        # specify the inputs for the feature extractor network
        # inputs = Input(inputShape)
        inputs = Input((None, None, inputShape[2]))
        # define the first set of CONV => RELU => POOL => DROPOUT layers
        splitted = spec.split();
        print(splitted);
        first = True
        for layer in splitted:
            if layer.startswith('C'):
                neuronType = layer[1:2]
                layer = layer[2:]
                numbers = layer.split(',')
                outputs = int(numbers[2])
                height = int(numbers[0])
                width = int(numbers[1])
                if neuronType == 'l':
                    activation = tf.keras.layers.LeakyReLU(alpha=0.3)
                if neuronType == 't':
                    activation = tf.keras.activations.tanh()
                if neuronType == 'r':
                    activation = tf.keras.activations.relu()
                if neuronType == 's':
                    activation = tf.keras.activations.sigmoid()
                if neuronType == 'e':
                    activation = "elu"
                if first:
                    x = Conv2D(outputs, (height, width), strides=(1, 1), padding="same",
                               activation=activation)(inputs)
                    first = False
                else:
                    x = Conv2D(outputs, (height, width), strides=(1, 1), padding="same",
                               activation=activation)(x)
            if layer.startswith('F'):
                outputs = int(layer[2:])
                x = Dense(outputs)(x)
            if layer.startswith('Mp'):
                layer = layer[2:]
                numbers = layer.split(',')
                height = int(numbers[0])
                width = int(numbers[1])
                x = MaxPooling2D(pool_size=(height, width))(x)

            if layer.startswith('Gm'):
                x = GlobalMaxPooling2D()(x)
            if layer.startswith('Ga'):
                x = GlobalAveragePooling2D()(x)
            if layer.startswith('D'):
                layer = layer[1:]
                dropout = float(layer)
                x = Dropout(dropout)(x)

        # x = Conv2D(16, (5, 5), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)
        # # x = Dropout(0.1)(x)
        # # second set of CONV => RELU => POOL => DROPOUT layers
        # x = Conv2D(32, (3, 3), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        # # x = Dropout(0.1)(x)
        #
        # x = Conv2D(64, (3, 3), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)
        # # x = MaxPooling2D(pool_size=(2, 2))(x)
        # # x = Conv2D(32, (3, 3), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)
        # # x = Conv2D(128, (3, 3), padding="same")(x)
        # # x = MaxPooling2D(pool_size=(2, 2))(x)
        # # x = Dropout(0.1)(x)
        #
        # # x = Conv2D(128, (3, 3), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x)
        # # x = MaxPooling2D(pool_size=(2, 2))(x)
        # # # x = Dropout(0.1)(x)
        #
        # x = Conv2D(256, (3, 3), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x)
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        # x = Dropout(0.1)(x)

        # x = BatchNormalization(epsilon=1e-06, momentum=0.9)(x)
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        # x = Dropout(0.2)(x)
        # x = Conv2D(64, (3, 3), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x)
        # x = BatchNormalization(epsilon=1e-06, momentum=0.9)(x)
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        # x = Conv2D(256, (3, 3), padding="valid", activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x)
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        # x = Dropout(0.2)(x)
        # x = Conv2D(256, (3, 3), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x)
        # x = Dropout(0.2)(x)
        # x = Conv2D(128, (3, 5), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x)
        # x = Dropout(0.2)(x)

        # prepare the final outputs
        # x = GlobalMaxPooling2D()(x)
        # x = Flatten()(x)
        # x = Dropout(0.5)(x)
        # outputs = Dense(embeddingDim)(pooledOutput)
        outputs = x
        # x = Dense(1024)(x)
        # outputs = Dense(128)(x)
        # build the modelZ
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_signet(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input(inputShape)
        x = Conv2D(16, (11, 11), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3))(
            inputs)
        # x = BatchNormalization(epsilon=1e-06, momentum=0.9)(x)
        # x = MaxPooling2D(pool_size=(3, 3),strides=(2,2))(x)
        # x = Conv2D(256, (5, 5), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)
        # x = Dropout(0.3)(x)
        # x = Conv2D(384, (3, 3), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)
        # x = MaxPooling2D(pool_size=(3, 3),strides=(2,2))(x)
        # x = Conv2D(256, (3, 3), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)
        # x = MaxPooling2D(pool_size=(3, 3),strides=(2,2))(x)
        # x = Dropout(0.3)(x)

        x = GlobalMaxPooling2D()(x)

        # x = Dense(1024)(x)
        # x = Dense(128)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornetA(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        conv1 = Conv2D(96, (11, 11), strides=(1, 1), padding="same", activation="elu",
                       name="conv1")(inputs)
        globalMaxPooling = GlobalMaxPooling2D(name="globalMP1")(conv1)

        x = Dense(96)(globalMaxPooling)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornetB(inputShape):
        # specify the inputs for the feature extractor network
        model = keras.models.load_model(
            '/home/rutger/src/siamesenew/checkpoints-iisg/difornetA-saved-model-04-0.77.hdf5')
        model.summary()
        model = model.get_layer(index=2)
        model.summary()

        i = -len(model.layers)
        x = model.layers[i]
        i = i + 1
        x = model.layers[i](x.output)
        for layer in model.layers:
            print(layer.name)
            i = i + 1
            if i > 1:
                x = model.layers[i](x)

            layer.trainable = False
            if layer.name == "conv1":
                conv1 = layer
                # conv1.trainable = False
                break
        maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), name="maxpool1")(conv1.output)
        conv2 = Conv2D(32, (3, 3), padding="same", activation="elu", name="conv2")(maxpool1)

        globalmaxpool = GlobalMaxPooling2D(name="globalMP2")(conv2)

        x = Dense(96)(globalmaxpool)
        x = Dense(96)(x)
        outputs = x
        model = Model(inputs=model.inputs, outputs=outputs)
        # return the model to the calling function
        return model

    def build_difornetC(inputShape):
        # specify the inputs for the feature extractor network
        model = keras.models.load_model(
            '/home/rutger/src/siamesenew/checkpoints-iisg/difornet20-saved-model-149-0.97.hdf5')
        model.summary()
        model = model.get_layer(index=2)
        model.summary()

        i = -len(model.layers)
        x = model.layers[i]
        i = i + 1
        x = model.layers[i](x.output)
        for layer in model.layers:
            print(layer.name)
            i = i + 1
            if i > 1:
                x = model.layers[i](x)

            layer.trainable = False
            if layer.name == "conv1":
                conv1 = layer
                conv1.trainable = False
            if layer.name == "conv2":
                conv2 = layer
                conv2.trainable = False
            if layer.name == "conv3":
                conv3 = layer
                conv3.trainable = False
            if layer.name == "conv4":
                conv4 = layer
                conv4.trainable = False
                break
        # maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), name="maxpool2")(conv2.output)
        conv5 = Conv2D(256, (3, 3), padding="same", activation="elu", name="conv5")(conv4.output)

        globalmaxpool = GlobalMaxPooling2D(name="globalMP2")(conv5)

        # x = Dense(256)(globalmaxpool)
        x = Dense(96)(globalmaxpool)
        outputs = x
        model = Model(inputs=model.inputs, outputs=outputs)
        # return the model to the calling function
        return model

    def build_difornetD(inputShape):
        # specify the inputs for the feature extractor network
        model = keras.models.load_model('/home/rutger/src/siamesenew/checkpoints/difornetC-best_val_loss')
        model.summary()
        model = model.get_layer(index=2)
        model.summary()

        i = -len(model.layers)
        x = model.layers[i]
        i = i + 1
        x = model.layers[i](x.output)
        for layer in model.layers:
            print(layer.name)
            layer.trainable = True

        # return the model to the calling function
        return model

    def build_difornetE(inputShape):
        # specify the inputs for the feature extractor network
        model = keras.models.load_model('/home/rutger/src/siamesenew/checkpoints/difornetC-best_val_loss')
        model.summary()
        model = model.get_layer(index=2)
        model.summary()

        i = -len(model.layers)
        x = model.layers[i]
        i = i + 1
        x = model.layers[i](x.output)
        for layer in model.layers:
            print(layer.name)
            i = i + 1
            if i > 1:
                x = model.layers[i](x)

            layer.trainable = False
            if layer.name == "conv1":
                conv1 = layer
                conv1.trainable = False
            if layer.name == "conv2":
                conv2 = layer
                conv2.trainable = False
            if layer.name == "conv3":
                conv3 = layer
                conv3.trainable = False
            if layer.name == "conv4":
                conv4 = layer
                conv4.trainable = False
            if layer.name == "conv5":
                conv5 = layer
                conv5.trainable = False
                break
        # maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), name="maxpool2")(conv2.output)
        conv6 = Conv2D(384, (3, 3), padding="same", activation="elu", name="conv6")(conv5.output)

        globalmaxpool = GlobalMaxPooling2D(name="globalMP2")(conv6)

        x = Dense(256)(globalmaxpool)
        x = Dense(96)(x)
        outputs = x
        model = Model(inputs=model.inputs, outputs=outputs)
        # return the model to the calling function
        return model

    def build_difornetF(inputShape):
        # specify the inputs for the feature extractor network
        model = keras.models.load_model('/home/rutger/src/siamesenew/checkpoints/difornetE-best_val_loss')
        model.summary()
        model = model.get_layer(index=2)
        model.summary()

        i = -len(model.layers)
        x = model.layers[i]
        i = i + 1
        x = model.layers[i](x.output)
        for layer in model.layers:
            print(layer.name)
            i = i + 1
            if i > 1:
                x = model.layers[i](x)

            layer.trainable = False
            if layer.name == "conv1":
                conv1 = layer
                conv1.trainable = False
            if layer.name == "conv2":
                conv2 = layer
                conv2.trainable = False
            if layer.name == "conv3":
                conv3 = layer
                conv3.trainable = False
            if layer.name == "conv4":
                conv4 = layer
                conv4.trainable = False
            if layer.name == "conv5":
                conv5 = layer
                conv5.trainable = False
            if layer.name == "conv6":
                conv6 = layer
                conv6.trainable = False
                break
        # maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), name="maxpool2")(conv2.output)
        conv7 = Conv2D(384, (3, 3), padding="same", activation="elu", name="conv7")(conv6.output)

        globalmaxpool = GlobalMaxPooling2D(name="globalMP2")(conv7)

        x = Dense(256)(globalmaxpool)
        x = Dense(96)(x)
        outputs = x
        model = Model(inputs=model.inputs, outputs=outputs)
        # return the model to the calling function
        return model

    def build_difornetG(inputShape):
        # specify the inputs for the feature extractor network
        model = keras.models.load_model('/home/rutger/src/siamesenew/checkpoints/difornetF-best_val_loss')
        model.summary()
        model = model.get_layer(index=2)
        model.summary()

        i = -len(model.layers)
        x = model.layers[i]
        i = i + 1
        x = model.layers[i](x.output)
        for layer in model.layers:
            print(layer.name)
            i = i + 1
            if i > 1:
                x = model.layers[i](x)

            layer.trainable = False
            if layer.name == "conv1":
                conv1 = layer
                conv1.trainable = False
            if layer.name == "conv2":
                conv2 = layer
                conv2.trainable = False
            if layer.name == "conv3":
                conv3 = layer
                conv3.trainable = False
            if layer.name == "conv4":
                conv4 = layer
                conv4.trainable = False
            if layer.name == "conv5":
                conv5 = layer
                # conv5.trainable = False
            if layer.name == "conv6":
                conv6 = layer
                # conv6.trainable = False
            if layer.name == "conv7":
                conv7 = layer
                # conv7.trainable = False
                break
        # maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), name="maxpool2")(conv2.output)
        conv8 = Conv2D(384, (3, 3), padding="same", activation="elu", name="conv8")(conv7.output)

        globalmaxpool = GlobalMaxPooling2D(name="globalMP2")(conv8)

        x = Dense(1024)(globalmaxpool)
        x = Dense(96)(x)
        outputs = x
        model = Model(inputs=model.inputs, outputs=outputs)
        # return the model to the calling function
        return model

    def build_difornet(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(64, (11, 11), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    name="conv1")(inputs)
        # x = BatchNormalization(epsilon=1e-06, momentum=0.9)(x)
        x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool1")(x1)
        x3 = Conv2D(64, (7, 7), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3), name="conv2")(x2)
        # x = Dropout(0.3)(x)
        x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x3)
        x5 = Conv2D(64, (3, 3), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3), name="conv3")(x4)
        # x5 = MaxPooling2D(pool_size=(3, 3),strides=(2,2))(x4)
        # x = Conv2D(256, (3, 3), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)
        # x = MaxPooling2D(pool_size=(3, 3),strides=(2,2))(x)
        # x = Dropout(0.3)(x)

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP2")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP3")(x5)
        # z1 = GlobalAveragePooling2D()(x1)
        # z3 = GlobalAveragePooling2D()(x3)
        # z5 = GlobalAveragePooling2D()(x5)
        # y2 = GlobalAveragePooling2D()(x)
        #
        x = Concatenate()([y1, y3, y5])
        # x = Concatenate()([y1,y3,y5,z1,z3,z5])

        x = Dense(1024)(x)
        x = Dense(128)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet2(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(8, (11, 11), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    name="conv1")(inputs)
        # x2 = x1
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), name="maxpool1")(x1)
        x3 = Conv2D(16, (3, 3), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3), name="conv2")(x2)
        # x = Dropout(0.3)(x)
        # x4 = x3
        x4 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x3)
        x5 = Conv2D(32, (3, 3), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3), name="conv3")(x4)
        # x6=x5
        # # x6 = MaxPooling2D(pool_size=(2, 2),strides=(1,1))(x5)
        # x7 = Conv2D(64, (3, 3), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),name="conv4")(x6)
        # x8=x7
        # # x8 = MaxPooling2D(pool_size=(2, 2),strides=(1,1))(x7)
        # x9 = Conv2D(128, (3, 3), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),name="conv5")(x8)
        # x10=x9
        # # x10 = MaxPooling2D(pool_size=(2, 2),strides=(1,1))(x9)
        # x11 = Conv2D(256, (3, 3), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),name="conv6")(x10)

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP2")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP3")(x5)
        # y7 = GlobalMaxPooling2D(name="globalMP4")(x7)
        # y9 = GlobalMaxPooling2D(name="globalMP5")(x9)
        # y11 = GlobalMaxPooling2D(name="globalMP6")(x11)
        # z1 = GlobalAveragePooling2D()(x1)
        # z3 = GlobalAveragePooling2D()(x3)
        # z5 = GlobalAveragePooling2D()(x5)
        # y2 = GlobalAveragePooling2D()(x)
        #
        x = Concatenate()([y1, y3, y5])
        # x = Concatenate()([y1,y3,y5,y7,y9,y11])
        # x = Concatenate()([y1,y3,y5,z1,z3,z5])

        x = Dense(1024)(x)
        x = Dense(128)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet3_dropout(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1a = Conv2D(96, (11, 11), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                     name="conv1")(inputs)
        x1b = Dropout(0.2)(x1a)
        # x = BatchNormalization(epsilon=1e-06, momentum=0.9)(x)
        x2 = MaxPooling2D(pool_size=(2, 2), name="maxpool1")(x1b)
        x3 = Conv2D(256, (7, 7), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3), name="conv2")(x2)
        x4 = Dropout(0.2)(x3)
        x5 = MaxPooling2D(pool_size=(2, 2))(x4)
        x6 = Conv2D(384, (3, 3), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3), name="conv3")(x5)
        x7 = Dropout(0.2)(x6)
        # x5 = MaxPooling2D(pool_size=(3, 3),strides=(2,2))(x4)
        x8 = Conv2D(512, (3, 3), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3), name="conv4")(x7)
        # x = MaxPooling2D(pool_size=(3, 3),strides=(2,2))(x)
        x9 = Dropout(0.2)(x8)

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1a)
        y3 = GlobalMaxPooling2D(name="globalMP2")(x3)
        y6 = GlobalMaxPooling2D(name="globalMP3")(x6)
        y9 = GlobalMaxPooling2D(name="globalMP4")(x9)
        z1a = GlobalAveragePooling2D()(x1a)
        z3 = GlobalAveragePooling2D()(x3)
        z5 = GlobalAveragePooling2D()(x5)
        z9 = GlobalAveragePooling2D()(x9)
        #
        # x = Concatenate()([y1, y3, y6, y8])
        x = Concatenate()([y1, y3, y6, y9, z1a, z3, z5, z9])
        x = Dropout(0.5)(x)

        x = Dense(1024)(x)
        x = Dropout(0.5)(x)
        x = Dense(1024)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet3(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1a = Conv2D(96, (11, 11), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                     name="conv1")(inputs)
        x1b = Dropout(0.2)(x1a)
        # x = BatchNormalization(epsilon=1e-06, momentum=0.9)(x)
        x2 = MaxPooling2D(pool_size=(2, 2), name="maxpool1")(x1b)
        x3 = Conv2D(256, (7, 7), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3), name="conv2")(x2)
        x4 = Dropout(0.2)(x3)
        x5 = MaxPooling2D(pool_size=(2, 2))(x4)
        x6 = Conv2D(384, (3, 3), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3), name="conv3")(x5)
        x7 = Dropout(0.2)(x6)
        # x5 = MaxPooling2D(pool_size=(3, 3),strides=(2,2))(x4)
        x8 = Conv2D(512, (3, 3), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3), name="conv4")(x7)
        # x = MaxPooling2D(pool_size=(3, 3),strides=(2,2))(x)
        # x9 = Dropout(0.2)(x8)

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1a)
        y3 = GlobalMaxPooling2D(name="globalMP2")(x3)
        y6 = GlobalMaxPooling2D(name="globalMP3")(x6)
        y9 = GlobalMaxPooling2D(name="globalMP4")(x8)
        z1a = GlobalAveragePooling2D()(x1a)
        z3 = GlobalAveragePooling2D()(x3)
        z5 = GlobalAveragePooling2D()(x5)
        z9 = GlobalAveragePooling2D()(x8)
        #
        # x = Concatenate()([y1, y3, y6, y9])
        x = Concatenate()([y1, y3, y6, y9, z1a, z3, z5, z9])
        x = Dropout(0.5)(x)

        x = Dense(1024)(x)
        x = Dropout(0.5)(x)
        x = Dense(1024)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet3_autoencoded(inputShape):
        # newModel = tf.keras.Sequential()
        # model.add(VGG16(weights='imagenet'))
        # VGG = VGG16(include_top=False, weights='imagenet', input_shape=inputShape)
        # VGG = VGG16(weights='imagenet')
        model = keras.models.load_model('/home/rutger/src/diabola2/checkpoints/best_val')
        i = -len(model.layers)
        x = model.layers[i]
        i = i + 1
        x = model.layers[i](x.output)
        for layer in model.layers:
            i = i + 1
            if i > 1:
                x = model.layers[i](x)

            layer.trainable = False
            if layer.name == "conv1":
                conv1 = layer
            if layer.name == "conv2":
                conv2 = layer
            if layer.name == "conv3":
                conv3 = layer
            if layer.name == "conv4":
                conv4 = layer
                break
        y1 = GlobalMaxPooling2D(name="globalMP1")(conv1.output)
        y3 = GlobalMaxPooling2D(name="globalMP2")(conv2.output)
        y6 = GlobalMaxPooling2D(name="globalMP3")(conv3.output)
        y9 = GlobalMaxPooling2D(name="globalMP4")(conv4.output)
        z1a = GlobalAveragePooling2D()(conv1.output)
        z3 = GlobalAveragePooling2D()(conv2.output)
        z5 = GlobalAveragePooling2D()(conv3.output)
        z9 = GlobalAveragePooling2D()(conv4.output)

        x = Concatenate()([y1, y3, y6, y9, z1a, z3, z5, z9])
        x = Dropout(0.5, name="fc_dropout1")(x)

        x = Dense(1024)(x)
        x = Dropout(0.5, name="fc_dropout2")(x)
        x = Dense(1024)(x)
        outputs = x
        model = Model(inputs=model.inputs, outputs=outputs)
        # return the model to the calling function
        return model

    def build_difornet4(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1a = Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                     name="conv1a")(inputs)
        x1b = Conv2D(8, (5, 5), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                     name="conv1b")(inputs)
        x1c = Conv2D(8, (7, 7), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                     name="conv1c")(inputs)
        x1d = Conv2D(8, (11, 11), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                     name="conv1d")(inputs)
        x1e = Conv2D(8, (13, 13), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                     name="conv1e")(inputs)
        x1 = tf.keras.layers.Add()([x1a, x1b, x1c, x1d, x1e])

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)

        x = Dense(1024)(y1)
        x = Dropout(0.5)(x)
        x = Dense(1024)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet5(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    name="conv1")(inputs)
        x2 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    name="conv2")(x1)
        x3 = Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    name="conv3")(x2)
        x4 = Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    name="conv4")(x3)
        x5a = Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                     name="conv5")(x4)
        x5b = tf.keras.layers.Add()([x3, x5a])
        x6a = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                     name="conv6")(x5b)
        x6b = tf.keras.layers.Add()([x2, x6a])
        x7a = Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                     name="conv7")(x6b)
        x7b = tf.keras.layers.Add()([x1, x7a])

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y2 = GlobalMaxPooling2D(name="globalMP2")(x2)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y4 = GlobalMaxPooling2D(name="globalMP4")(x4)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5b)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6b)
        y7 = GlobalMaxPooling2D(name="globalMP7")(x7b)

        x = Concatenate()([y1, y2, y3, y4, y5, y6, y7])

        x = Dense(1024)(x)
        x = Dropout(0.5)(x)
        x = Dense(1024)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet6(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    name="conv1")(inputs)
        x2 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    name="conv2")(x1)
        x3 = Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    name="conv3")(x2)
        x4 = Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    name="conv4")(x3)
        x5 = Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    name="conv5")(x4)
        x6 = Conv2D(256, (3, 3), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    name="conv6")(x5)
        x7 = Conv2D(512, (3, 3), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    name="conv7")(x6)

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y2 = GlobalMaxPooling2D(name="globalMP2")(x2)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y4 = GlobalMaxPooling2D(name="globalMP4")(x4)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)
        y7 = GlobalMaxPooling2D(name="globalMP7")(x7)

        x = Concatenate()([y1, y2, y3, y4, y5, y6, y7])

        x = Dense(1024)(x)
        x = Dense(1024)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet7(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(32, (7, 7), strides=(2, 2), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    name="conv1")(inputs)
        x2 = Conv2D(64, (5, 5), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    name="conv2")(x1)
        x3 = Conv2D(128, (5, 5), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    name="conv3")(x2)
        x4 = Conv2D(256, (3, 3), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    name="conv4")(x3)
        # x5 = Conv2D(512, (3, 3), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
        #             name="conv5")(x4)
        # x6 = Conv2D(1024, (3, 3), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
        #             name="conv6")(x5)

        # y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        # y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        # y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x4)

        # x = Concatenate()([y1,y3,y5,y7])

        x = Dense(1024)(y6)
        x = Dense(1024)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet8(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(96, (11, 11), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(128, (5, 5), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool2")(x3)
        x5 = Conv2D(192, (5, 5), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    name="conv3")(x4)
        x6 = Conv2D(256, (3, 3), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    name="conv4")(x5)
        x7 = Conv2D(384, (3, 3), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    name="conv5")(x6)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x7)

        x = Dense(1024)(y6)
        x = Dense(1024)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet9(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(96, (11, 11), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(128, (5, 5), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool2")(x3)
        x5 = Conv2D(192, (5, 5), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    name="conv3")(x4)
        x6 = Conv2D(256, (3, 3), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    name="conv4")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)

        x = Dense(1024)(y6)
        x = Dense(128)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet10(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(96, (11, 11), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(128, (5, 5), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool2")(x3)
        x5 = Conv2D(192, (5, 5), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    name="conv3")(x4)
        x6 = Conv2D(256, (3, 3), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    name="conv4")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)

        x = Dense(64)(y6)
        # x = Dense(128)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet11(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x = Conv2D(8, (5, 5), strides=(2, 2), padding="same", activation="relu",
                   name="conv1")(inputs)
        x = Conv2D(16, (5, 5), strides=(2, 2), padding="same", activation="relu",
                   name="conv2")(x)
        x = Conv2D(32, (5, 5), strides=(2, 2), padding="same", activation="relu",
                   name="conv3")(x)
        x = Conv2D(96, (5, 5), strides=(2, 2), padding="same", activation="relu",
                   name="conv4")(x)
        x = GlobalMaxPooling2D(name="globalMP6")(x)
        x = Dense(128)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet12(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(96, (11, 11), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(128, (5, 5), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool2")(x3)
        x5 = Conv2D(192, (5, 5), strides=(1, 1), padding="same", activation="elu",
                    name="conv3")(x4)
        x6 = Conv2D(256, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv4")(x5)
        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)
        z1 = GlobalAveragePooling2D(name="globalAP1")(x1)
        z3 = GlobalAveragePooling2D(name="globalAP3")(x3)
        z5 = GlobalAveragePooling2D(name="globalAP5")(x5)
        z6 = GlobalAveragePooling2D(name="globalAP6")(x6)

        x = Concatenate()([y1, y3, y5, y6, z1, z3, z5, z6])

        x = Dense(1024)(x)
        x = Dense(128)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet13(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(96, (11, 11), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(128, (5, 5), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool2")(x3)
        x5 = Conv2D(192, (5, 5), strides=(1, 1), padding="valid", activation="elu",
                    name="conv3")(x4)
        x6 = Conv2D(256, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv4")(x5)
        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)
        z1 = GlobalAveragePooling2D(name="globalAP1")(x1)
        z3 = GlobalAveragePooling2D(name="globalAP3")(x3)
        z5 = GlobalAveragePooling2D(name="globalAP5")(x5)
        z6 = GlobalAveragePooling2D(name="globalAP6")(x6)

        x = Concatenate()([y1, y3, y5, y6, z1, z3, z5, z6])
        x = Dropout(0.3)(x)
        x = Dense(1024)(x)
        x = Dropout(0.5)(x)
        x = Dense(128)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet14(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(96, (11, 11), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(128, (5, 5), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool2")(x3)
        x5 = Conv2D(192, (5, 5), strides=(1, 1), padding="valid", activation="elu",
                    name="conv3")(x4)
        x6 = Conv2D(256, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv4")(x5)
        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)

        x = Concatenate()([y1, y3, y5, y6])

        x = Dense(1024)(x)
        x = Dense(128)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet15(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(96, (11, 11), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(128, (5, 5), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool2")(x3)
        x5 = Conv2D(192, (5, 5), strides=(1, 1), padding="valid", activation="elu",
                    name="conv3")(x4)
        x6 = Conv2D(256, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv4")(x5)

        z1 = GlobalAveragePooling2D(name="globalAP1")(x1)
        z3 = GlobalAveragePooling2D(name="globalAP3")(x3)
        z5 = GlobalAveragePooling2D(name="globalAP5")(x5)
        z6 = GlobalAveragePooling2D(name="globalAP6")(x6)

        x = Concatenate()([z1, z3, z5, z6])

        x = Dense(1024)(x)
        x = Dense(128)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet16(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(96, (11, 11), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    name="conv1")(inputs)
        # x3 = Conv2D(128, (5, 5), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
        #             name="conv2")(x1)
        # x5 = Conv2D(192, (3, 3), strides=(1, 1), padding="valid", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
        #             name="conv3")(x3)
        # x6 = Conv2D(256, (3, 3), strides=(1, 1), padding="valid", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
        #             name="conv4")(x5)
        # x7 = Conv2D(512, (3, 3), strides=(1, 1), padding="valid", activation=tf.keras.layers.LeakyReLU(alpha=0.3),
        #             name="conv5")(x6)
        y6 = GlobalMaxPooling2D(name="globalMP7")(x1)

        x = Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(y6)
        x = Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet17(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(16, (11, 11), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(32, (5, 5), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool2")(x3)
        x5 = Conv2D(64, (5, 5), strides=(1, 1), padding="valid", activation="elu",
                    name="conv3")(x4)
        x6 = Conv2D(128, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv4")(x5)
        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)

        x = Concatenate()([y1, y3, y5, y6])

        x = Dense(1024)(x)
        x = Dense(96)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet18(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(32, (11, 11), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(32, (5, 5), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool2")(x3)
        x5 = Conv2D(64, (5, 5), strides=(1, 1), padding="valid", activation="elu",
                    name="conv3")(x4)
        x6 = Conv2D(128, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv4")(x5)
        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)

        x = Concatenate()([y1, y3, y5, y6])

        x = Dense(1024)(x)
        x = Dense(96)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet19(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(64, (11, 11), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(32, (5, 5), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool2")(x3)
        x5 = Conv2D(64, (5, 5), strides=(1, 1), padding="valid", activation="elu",
                    name="conv3")(x4)
        x6 = Conv2D(128, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv4")(x5)
        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)

        x = Concatenate()([y1, y3, y5, y6])

        x = Dense(1024)(x)
        x = Dense(96)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet20(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(96, (11, 11), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(32, (5, 5), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool2")(x3)
        x5 = Conv2D(64, (5, 5), strides=(1, 1), padding="valid", activation="elu",
                    name="conv3")(x4)
        x6 = Conv2D(128, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv4")(x5)
        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)

        x = Concatenate()([y1, y3, y5, y6])

        x = Dense(1024)(x)
        x = Dense(96)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet21(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(16, (11, 11), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(32, (5, 5), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool2")(x3)
        x5 = Conv2D(32, (5, 5), strides=(1, 1), padding="valid", activation="elu",
                    name="conv3")(x4)
        x6 = Conv2D(64, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv4")(x5)
        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)

        x = Concatenate()([y1, y3, y5, y6])

        x = Dense(1024)(x)
        x = Dense(96)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet22(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(16, (11, 11), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(32, (5, 5), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool2")(x3)
        x5 = Conv2D(32, (5, 5), strides=(1, 1), padding="valid", activation="elu",
                    name="conv3")(x4)
        x6 = Conv2D(64, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv4")(x5)
        x7 = Conv2D(128, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv5")(x6)
        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)
        y7 = GlobalMaxPooling2D(name="globalMP7")(x7)

        x = Concatenate()([y1, y3, y5, y6, y7])

        x = Dense(1024)(x)
        x = Dense(96)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet23(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool2")(x3)
        x5 = Conv2D(16, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv3")(x4)
        x6 = Conv2D(16, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv4")(x5)
        x7 = Conv2D(16, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv5")(x6)
        x8 = Conv2D(16, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv6")(x7)
        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)
        y7 = GlobalMaxPooling2D(name="globalMP7")(x7)
        y8 = GlobalMaxPooling2D(name="globalMP8")(x8)

        x = Concatenate()([y1, y3, y5, y6, y7, y8])

        x = Dense(128)(x)
        x = Dense(16)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet24RK(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(4, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name="maxpool1")(x1)

        x3 = Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name="maxpool2")(x3)
        x5 = Conv2D(16, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv3")(x4)
        # y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        # y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        # y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x5)

        # x = Concatenate()([y1, y3, y5, y6])

        x = Dense(128)(y6)
        x = Dense(96)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet24(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(4, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(x1)
        x3 = Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool2")(x3)
        x5 = Conv2D(8, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv3")(x4)
        x6 = Conv2D(8, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv4")(x5)
        x7 = Conv2D(8, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv5")(x6)
        x8 = Conv2D(8, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv6")(x7)
        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)
        y7 = GlobalMaxPooling2D(name="globalMP7")(x7)
        y8 = GlobalMaxPooling2D(name="globalMP8")(x8)

        x = Concatenate()([y1, y3, y5, y6, y7, y8])

        x = Dense(128)(x)
        x = Dense(16)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet25(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool2")(x3)
        x5 = Conv2D(16, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv3")(x4)
        x6 = Conv2D(16, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv4")(x5)
        x7 = Conv2D(16, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv5")(x6)
        x8 = Conv2D(16, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv6")(x7)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)
        y7 = GlobalMaxPooling2D(name="globalMP7")(x7)
        y8 = GlobalMaxPooling2D(name="globalMP8")(x8)

        x = Concatenate()([y6, y7, y8])

        x = Dense(128)(x)
        x = Dense(16)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet26(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(8, (6, 6), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(16, (6, 6), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool2")(x3)
        x5 = Conv2D(16, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv3")(x4)
        x6 = Conv2D(16, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv4")(x5)
        x7 = Conv2D(16, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv5")(x6)
        x8 = Conv2D(16, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv6")(x7)
        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)
        y7 = GlobalMaxPooling2D(name="globalMP7")(x7)
        y8 = GlobalMaxPooling2D(name="globalMP8")(x8)

        x = Concatenate()([y1, y3, y5, y6, y7, y8])

        x = Dense(128)(x)
        x = Dense(16)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet27(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(8, (3, 3), strides=(2, 2), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool2")(x3)
        x5 = Conv2D(16, (3, 3), strides=(2, 2), padding="valid", activation="elu",
                    name="conv3")(x4)
        x6 = Conv2D(16, (2, 2), strides=(1, 1), padding="valid", activation="elu",
                    name="conv4")(x5)
        x7 = Conv2D(16, (1, 1), strides=(1, 1), padding="valid", activation="elu",
                    name="conv5")(x6)
        x8 = Conv2D(16, (1, 1), strides=(1, 1), padding="valid", activation="elu",
                    name="conv6")(x7)
        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)
        y7 = GlobalMaxPooling2D(name="globalMP7")(x7)
        y8 = GlobalMaxPooling2D(name="globalMP8")(x8)

        x = Concatenate()([y1, y3, y5, y6, y7, y8])

        x = Dense(128)(x)
        x = Dense(16)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet28(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool2")(x3)
        x5 = Conv2D(16, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv3")(x4)
        x6 = Conv2D(16, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv4")(x5)
        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)

        x = Concatenate()([y1, y3, y5, y6])

        x = Dense(128)(x)
        x = Dense(16)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet29(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool2")(x3)
        x5 = Conv2D(16, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv3")(x4)
        x6 = Conv2D(16, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv4")(x5)
        x7 = Conv2D(16, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv5")(x6)
        x8 = Conv2D(16, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv6")(x7)
        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)
        y7 = GlobalMaxPooling2D(name="globalMP7")(x7)
        y8 = GlobalMaxPooling2D(name="globalMP8")(x8)

        x = Concatenate()([y1, y3, y5, y6, y7, y8])

        x = Dense(128)(x)
        x = Dense(32)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet30(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool2")(x3)
        x5 = Conv2D(16, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv3")(x4)
        x6 = Conv2D(16, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv4")(x5)
        x7 = Conv2D(16, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv5")(x6)
        x8 = Conv2D(16, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv6")(x7)
        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)
        y7 = GlobalMaxPooling2D(name="globalMP7")(x7)
        y8 = GlobalMaxPooling2D(name="globalMP8")(x8)

        x = Concatenate()([y1, y3, y5, y6, y7, y8])

        x = Dense(64)(x)
        x = Dense(16)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet31(inputShape):
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)

        x = GlobalMaxPooling2D(name="globalMP1")(x1)

        x = Dense(8)(x)
        x = Dense(8)(x)
        outputs = x
        model = Model(inputs, outputs)

        return model

    def build_difornet32(inputShape):
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP2")(x3)

        x = Concatenate()([y1, y3])

        x = Dense(8)(x)
        x = Dense(8)(x)
        outputs = x
        model = Model(inputs, outputs)

        return model

    def build_difornet33(inputShape):
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool2")(x3)

        x5 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv3")(x4)

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP2")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP3")(x5)

        x = Concatenate()([y1, y3, y5])

        x = Dense(8)(x)
        x = Dense(8)(x)
        outputs = x
        model = Model(inputs, outputs)

        return model

    def build_difornet34(inputShape):
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool2")(x3)

        x5 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv3")(x4)

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP2")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP3")(x5)

        x = Concatenate()([y1, y3, y5])

        x = Dense(16)(x)
        x = Dense(16)(x)
        outputs = x
        model = Model(inputs, outputs)

        return model

    def build_difornet35(inputShape):
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP2")(x3)

        x = Concatenate()([y1, y3])

        x = Dense(16)(x)
        x = Dense(16)(x)
        outputs = x
        model = Model(inputs, outputs)

        return model

    def build_difornet36(inputShape):
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool2")(x3)

        x5 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv3")(x4)

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP2")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP3")(x5)

        x = Concatenate()([y1, y3, y5])

        x = Dense(16)(x)
        x = Dense(64)(x)
        outputs = x
        model = Model(inputs, outputs)

        return model

    def build_difornet37(inputShape):
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool2")(x3)

        x5 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv3")(x4)

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP2")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP3")(x5)

        x = Concatenate()([y1, y3, y5])

        x = Dense(64)(x)
        x = Dense(16)(x)
        outputs = x
        model = Model(inputs, outputs)

        return model

    def build_difornet38(inputShape):
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool2")(x3)

        x5 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv3")(x4)

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP2")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP3")(x5)

        x = Concatenate()([y1, y3, y5])

        x = Dense(8)(x)
        x = Dense(64)(x)
        outputs = x
        model = Model(inputs, outputs)

        return model

    def build_difornet39(inputShape):
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool2")(x3)

        x5 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv3")(x4)

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP2")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP3")(x5)

        x = Concatenate()([y1, y3, y5])

        x = Dense(64)(x)
        x = Dense(16)(x)
        outputs = x
        model = Model(inputs, outputs)

        return model

    def build_difornet40(inputShape):
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(16, (2, 2), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(16, (2, 2), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool2")(x3)

        x5 = Conv2D(16, (2, 2), strides=(1, 1), padding="same", activation="elu",
                    name="conv3")(x4)

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP2")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP3")(x5)

        x = Concatenate()([y1, y3, y5])

        x = Dense(64)(x)
        x = Dense(16)(x)
        outputs = x
        model = Model(inputs, outputs)

        return model

    def build_difornet41(inputShape):
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool2")(x3)

        x5 = Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv3")(x4)

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP2")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP3")(x5)

        x = Concatenate()([y1, y3, y5])

        x = Dense(64)(x)
        x = Dense(16)(x)
        outputs = x
        model = Model(inputs, outputs)

        return model

    def build_difornet42(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(4, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(4, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool2")(x3)
        x5 = Conv2D(4, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv3")(x4)
        x6 = Conv2D(4, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv4")(x5)
        x7 = Conv2D(4, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv5")(x6)
        x8 = Conv2D(4, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv6")(x7)
        x9 = Conv2D(4, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv7")(x8)
        x10 = Conv2D(4, (1, 1), strides=(1, 1), padding="valid", activation="elu",
                     name="conv8")(x9)
        x11 = Conv2D(4, (1, 1), strides=(1, 1), padding="valid", activation="elu",
                     name="conv9")(x10)
        x12 = Conv2D(4, (1, 1), strides=(1, 1), padding="valid", activation="elu",
                     name="conv10")(x11)
        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)
        y7 = GlobalMaxPooling2D(name="globalMP7")(x7)
        y8 = GlobalMaxPooling2D(name="globalMP8")(x8)
        y9 = GlobalMaxPooling2D(name="globalMP9")(x9)
        y10 = GlobalMaxPooling2D(name="globalMP10")(x10)
        y11 = GlobalMaxPooling2D(name="globalMP11")(x11)
        y12 = GlobalMaxPooling2D(name="globalMP12")(x12)

        x = Concatenate()([y1, y3, y5, y6, y7, y8, y9, y10, y11, y12])

        x = Dense(128)(x)
        x = Dense(96)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet42a(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(4, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(4, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool2")(x3)
        x5 = Conv2D(4, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv3")(x4)
        x6 = Conv2D(4, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv4")(x5)
        x7 = Conv2D(4, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv5")(x6)
        x8 = Conv2D(4, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv6")(x7)
        x9 = Conv2D(4, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv7")(x8)
        x10 = Conv2D(4, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                     name="conv8")(x9)
        x11 = Conv2D(4, (1, 1), strides=(1, 1), padding="valid", activation="elu",
                     name="conv9")(x10)
        x12 = Conv2D(4, (1, 1), strides=(1, 1), padding="valid", activation="elu",
                     name="conv10")(x11)
        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)
        y7 = GlobalMaxPooling2D(name="globalMP7")(x7)
        y8 = GlobalMaxPooling2D(name="globalMP8")(x8)
        y9 = GlobalMaxPooling2D(name="globalMP9")(x9)
        y10 = GlobalMaxPooling2D(name="globalMP10")(x10)
        y11 = GlobalMaxPooling2D(name="globalMP11")(x11)
        y12 = GlobalMaxPooling2D(name="globalMP12")(x12)

        x = Concatenate()([y1, y3, y5, y6, y7, y8, y9, y10, y11, y12])

        x = Dense(128)(x)
        x = Dense(96)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet43(inputShape):
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation="elu", name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(x1)
        x3 = Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool2")(x3)
        x5 = Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv3")(x4)

        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)

        x = y5

        x = Dense(128)(x)
        x = Dense(96)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet43a(inputShape):
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation="elu", name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(x1)
        x3 = Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool2")(x3)
        x5 = Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv3")(x4)

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)

        x = x = Concatenate()([y1, y3, y5])

        x = Dense(128)(x)
        x = Dense(96)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet44(inputShape):
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation="elu", name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(x1)
        x3 = Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv2")(x2)
        x4 = Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv3")(x3)
        x5 = Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv4")(x4)

        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)

        x = y5

        x = Dense(128)(x)
        x = Dense(96)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet44a(inputShape):
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation="elu", name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(x1)
        x3 = Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv2")(x2)
        x4 = Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv3")(x3)
        x5 = Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv4")(x4)

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y4 = GlobalMaxPooling2D(name="globalMP4")(x4)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)

        x = Concatenate()([y1, y3, y5, y5])

        x = Dense(128)(x)
        x = Dense(96)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet44b(inputShape):
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation="elu", name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(x1)
        x3 = Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv2")(x2)
        x4 = Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv3")(x3)
        x5 = Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv4")(x4)

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y4 = GlobalMaxPooling2D(name="globalMP4")(x4)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)

        x = Concatenate()([y1, y3, y4, y5])

        x = Dense(128)(x)
        x = Dense(96)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet44c(inputShape):
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation="elu", name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(x1)
        x3 = Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv2")(x2)
        x4 = Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv3")(x3)
        x5 = Conv2D(32, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv4")(x4)

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y4 = GlobalMaxPooling2D(name="globalMP4")(x4)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)

        x = Concatenate()([y1, y3, y4, y5])

        x = Dense(128)(x)
        x = Dense(96)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet45(inputShape):
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation="elu", name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(x1)
        x3 = Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv2")(x2)
        x4 = Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv3")(x3)
        x5 = Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv4")(x4)
        x6 = Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv5")(x5)

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y4 = GlobalMaxPooling2D(name="globalMP4")(x4)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)

        x = Concatenate()([y1, y3, y4, y5, y6])

        x = Dense(128)(x)
        x = Dense(96)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet45a(inputShape):
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation="elu", name="conv1")(inputs)
        x2 = Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv2")(x1)
        x3 = Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv3")(x2)
        x4 = Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv4")(x3)
        x5 = Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv5")(x4)

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y2 = GlobalMaxPooling2D(name="globalMP2")(x2)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y4 = GlobalMaxPooling2D(name="globalMP4")(x4)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)

        x = Concatenate()([y1, y2, y3, y4, y5])

        x = Dense(128)(x)
        x = Dense(96)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet45b(inputShape):
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation="elu", name="conv1")(inputs)
        x2 = Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv2")(x1)
        x3 = Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv3")(x2)
        x4 = Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv4")(x3)
        x5 = Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv5")(x4)
        x6 = Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv6")(x5)

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y2 = GlobalMaxPooling2D(name="globalMP2")(x2)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y4 = GlobalMaxPooling2D(name="globalMP4")(x4)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)

        x = Concatenate()([y1, y2, y3, y4, y5, y6])

        x = Dense(128)(x)
        x = Dense(96)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet45c(inputShape):
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation="elu", name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(x1)
        x3 = Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv2")(x2)
        x4 = Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv3")(x3)
        x5 = Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv4")(x4)
        x6 = Conv2D(32, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv5")(x5)

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y4 = GlobalMaxPooling2D(name="globalMP4")(x4)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)

        x = Concatenate()([y1, y3, y4, y5, y6])

        x = Dense(128)(x)
        x = Dense(96)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet45d(inputShape):
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation="elu", name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(x1)
        x3 = Conv2D(32, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv2")(x2)
        x4 = Conv2D(32, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv3")(x3)
        x5 = Conv2D(32, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv4")(x4)
        x6 = Conv2D(64, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv5")(x5)

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y4 = GlobalMaxPooling2D(name="globalMP4")(x4)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)

        x = Concatenate()([y1, y3, y4, y5, y6])

        x = Dense(128)(x)
        x = Dense(96)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet45e(inputShape):
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation="elu", name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(x1)
        x3 = Conv2D(64, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv2")(x2)
        x4 = Conv2D(64, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv3")(x3)
        x5 = Conv2D(64, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv4")(x4)
        x6 = Conv2D(128, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv5")(x5)

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y4 = GlobalMaxPooling2D(name="globalMP4")(x4)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)

        x = Concatenate()([y1, y3, y4, y5, y6])

        x = Dense(128)(x)
        x = Dense(96)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet45f(inputShape):
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation="elu", name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(x1)
        x3 = Conv2D(128, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv2")(x2)
        x4 = Conv2D(128, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv3")(x3)
        x5 = Conv2D(128, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv4")(x4)
        x6 = Conv2D(256, (5, 5), strides=(1, 1), padding="valid", activation="elu", name="conv5")(x5)

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y4 = GlobalMaxPooling2D(name="globalMP4")(x4)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)

        x = Concatenate()([y1, y3, y4, y5, y6])

        x = Dense(128)(x)
        x = Dense(96)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet46(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(32, (11, 11), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), name="maxpool1")(x1)

        x3 = Conv2D(32, (5, 5), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), name="maxpool2")(x3)
        x5 = Conv2D(64, (5, 5), strides=(1, 1), padding="valid", activation="elu",
                    name="conv3")(x4)
        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)

        x = Concatenate()([y1, y3, y5])

        x = Dense(1024)(x)
        x = Dense(96)(x)

        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet47(inputShape):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x1 = Conv2D(96, (11, 11), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(inputs)
        x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(32, (5, 5), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool2")(x3)
        x5 = Conv2D(64, (5, 5), strides=(1, 1), padding="same", activation="elu",
                    name="conv3")(x4)
        # x5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool3")(x5)
        x6 = Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv4")(x5)
        x7 = Conv2D(192, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv5")(x6)
        x8 = Conv2D(256, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv6")(x7)
        x9 = Conv2D(384, (3, 3), strides=(1, 1), padding="valid", activation="elu",
                    name="conv7")(x8)
        # x10 = Conv2D(512, (3, 3), strides=(1, 1), padding="valid", activation="elu",
        #              name="conv8")(x9)
        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)
        y7 = GlobalMaxPooling2D(name="globalMP7")(x7)
        y8 = GlobalMaxPooling2D(name="globalMP8")(x8)
        y9 = GlobalMaxPooling2D(name="globalMP9")(x9)
        z1 = GlobalAveragePooling2D(name="globalAP1")(x1)
        z3 = GlobalAveragePooling2D(name="globalAP3")(x3)
        z5 = GlobalAveragePooling2D(name="globalAP5")(x5)
        z6 = GlobalAveragePooling2D(name="globalAP6")(x6)
        z7 = GlobalAveragePooling2D(name="globalAP7")(x7)
        z8 = GlobalAveragePooling2D(name="globalAP8")(x8)
        z9 = GlobalAveragePooling2D(name="globalAP9")(x9)
        # y10 = GlobalMaxPooling2D(name="globalMP10")(x10)

        x = Concatenate()([y1, y3, y5, y6, y7, y8, y9,z1, z3, z5, z6, z7, z8, z9])

        x = Dense(1024)(x)
        x = Dense(96)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model


    def build_difornet48(inputShape):
        # specify the inputs for the feature extractor network
        dropout = 0 #.1
        dropoutdense = 0 #0.5
        inputs = Input((None, None, inputShape[2]))
        x = tf.keras.layers.Masking()(inputs)
        x1 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(x)
        # x1 = BatchNormalization(axis=-1)(x1)
        x = Dropout(dropout)(x1)
        x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool1")(x)

        x3 = Conv2D(32, (2, 2), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        # x3 = BatchNormalization(axis=-1)(x3)
        x = Dropout(dropout)(x3)
        x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool2")(x)
        x5 = Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv3")(x4)
        # x5 = BatchNormalization(axis=-1)(x5)
        x = Dropout(dropout)(x5)
        x6 = Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv4")(x)
        # x6 = BatchNormalization(axis=-1)(x6)
        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)
        z1 = GlobalAveragePooling2D(name="globalAP1")(x1)
        z3 = GlobalAveragePooling2D(name="globalAP3")(x3)
        z5 = GlobalAveragePooling2D(name="globalAP5")(x5)
        z6 = GlobalAveragePooling2D(name="globalAP6")(x6)
        # y10 = GlobalMaxPooling2D(name="globalMP10")(x10)

        x = Concatenate()([y1, y3, y5, y6, z1, z3, z5, z6])

        x = Dense(1024)(x)
        x = Dropout(dropoutdense)(x)
        x = Dense(1024)(x)
        x = Dropout(dropoutdense)(x)

        x = Dense(96)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet49(inputShape):
        # specify the inputs for the feature extractor network
        dropout = 0 #.1
        dropoutdense = 0#.5
        inputs = Input((None, None, inputShape[2]))
        x = tf.keras.layers.Masking()(inputs)
        x1 = Conv2D(96, (11, 11), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(x)
        x1 = BatchNormalization(axis=-1)(x1)
        x = Dropout(dropout)(x1)
        x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool1")(x)

        x3 = Conv2D(32, (5, 5), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x3 = BatchNormalization(axis=-1)(x3)
        x = Dropout(dropout)(x3)
        x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool2")(x)
        x5 = Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv3")(x4)
        x5 = BatchNormalization(axis=-1)(x5)
        x = Dropout(dropout)(x5)
        x6 = Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv4")(x)
        # x6 = BatchNormalization(axis=-1)(x6)
        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)
        z1 = GlobalAveragePooling2D(name="globalAP1")(x1)
        z3 = GlobalAveragePooling2D(name="globalAP3")(x3)
        z5 = GlobalAveragePooling2D(name="globalAP5")(x5)
        z6 = GlobalAveragePooling2D(name="globalAP6")(x6)
        # y10 = GlobalMaxPooling2D(name="globalMP10")(x10)

        x = Concatenate()([y1, y3, y5, y6, z1, z3, z5, z6])

        # x = Dense(1024)(x)
        # x = Dropout(dropoutdense)(x)
        # x = Dense(1024)(x)
        # x = Dropout(dropoutdense)(x)

        x = Dense(96)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model



    def build_difornet50(inputShape):
        # specify the inputs for the feature extractor network
        dropout = 0 #.1
        dropoutdense = 0 #0.5
        inputs = Input((None, None, inputShape[2]))
        x = tf.keras.layers.Masking()(inputs)
        s = x
        x1 = Conv2D(96, (11, 11), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(x)
        x1 = BatchNormalization(axis=-1)(x1)
        x = Dropout(dropout)(x1)
        x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool1")(x)

        x3 = Conv2D(24, (2, 2), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x3 = BatchNormalization(axis=-1)(x3)
        x = Dropout(dropout)(x3)
        x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool2")(x)
        x5 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv3")(x4)
        x5 = BatchNormalization(axis=-1)(x5)
        x = Dropout(dropout)(x5)
        x6 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv4")(x)

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)


        s1 = Conv2D(96, (11, 11), strides=(1, 1), padding="same", activation="elu",
                    name="conv1s")(s)
        # x1 = BatchNormalization(axis=-1)(x1)
        s = Dropout(dropout)(s1)
        s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool1s")(s)

        s3 = Conv2D(24, (2, 2), strides=(1, 1), padding="same", activation="elu",
                    name="conv2s")(s2)
        s3 = BatchNormalization(axis=-1)(s3)
        s = Dropout(dropout)(s3)
        s4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool2s")(s)
        s5 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv3s")(s4)
        s5 = BatchNormalization(axis=-1)(s5)
        s = Dropout(dropout)(s5)
        s6 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv4s")(s)

        s6 = BatchNormalization(axis=-1)(s6)
        z1 = GlobalAveragePooling2D(name="globalAP1")(s1)
        z3 = GlobalAveragePooling2D(name="globalAP3")(s3)
        z5 = GlobalAveragePooling2D(name="globalAP5")(s5)
        z6 = GlobalAveragePooling2D(name="globalAP6")(s6)
        # y10 = GlobalMaxPooling2D(name="globalMP10")(x10)

        # x = Concatenate()([y1, y3, y5, y6, z1, z3, z5, z6])
        # y = Concatenate()([y1, y3, y5, y6])
        # z = Concatenate()([z1, z3, z5, z6])
        # y = Concatenate()([y1, y3, y5, y6])
        # z = Concatenate()([z1, z3, z5, z6])
        # x = Dense(1024)(x)
        # x = Dropout(dropoutdense)(x)
        y = Dense(256)(y6)
        z = Dense(256)(z6)
        x = Concatenate()([y, z])
        # x = Dropout(dropoutdense)(x)

        x = Dense(96)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet51(inputShape):
        # specify the inputs for the feature extractor network
        dropout = 0 #.1
        dropoutdense = 0 #0.5
        inputs = Input((None, None, inputShape[2]))
        x = tf.keras.layers.Masking()(inputs)
        s = x
        x1 = Conv2D(96, (11, 11), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(x)
        x1 = BatchNormalization(axis=-1)(x1)
        x = Dropout(dropout)(x1)
        x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool1")(x)

        x3 = Conv2D(24, (2, 2), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x3 = BatchNormalization(axis=-1)(x3)
        x = Dropout(dropout)(x3)
        x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool2")(x)
        x5 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv3")(x4)
        x5 = BatchNormalization(axis=-1)(x5)
        x = Dropout(dropout)(x5)
        x6 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv4")(x)

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)


        s1 = Conv2D(96, (11, 11), strides=(1, 1), padding="same", activation="elu",
                    name="conv1s")(s)
        # x1 = BatchNormalization(axis=-1)(x1)
        s = Dropout(dropout)(s1)
        s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool1s")(s)

        s3 = Conv2D(24, (2, 2), strides=(1, 1), padding="same", activation="elu",
                    name="conv2s")(s2)
        s3 = BatchNormalization(axis=-1)(s3)
        s = Dropout(dropout)(s3)
        s4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool2s")(s)
        s5 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv3s")(s4)
        s5 = BatchNormalization(axis=-1)(s5)
        s = Dropout(dropout)(s5)
        s6 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv4s")(s)

        s6 = BatchNormalization(axis=-1)(s6)
        z1 = GlobalAveragePooling2D(name="globalAP1")(s1)
        z3 = GlobalAveragePooling2D(name="globalAP3")(s3)
        z5 = GlobalAveragePooling2D(name="globalAP5")(s5)
        z6 = GlobalAveragePooling2D(name="globalAP6")(s6)
        # y10 = GlobalMaxPooling2D(name="globalMP10")(x10)

        # x = Concatenate()([y1, y3, y5, y6, z1, z3, z5, z6])
        # y = Concatenate()([y1, y3, y5, y6])
        # z = Concatenate()([z1, z3, z5, z6])
        # y = Concatenate()([y1, y3, y5, y6])
        # z = Concatenate()([z1, z3, z5, z6])
        # x = Dense(1024)(x)
        # x = Dropout(dropoutdense)(x)
        y = Dense(256)(y6)
        z = Dense(256)(z6)
        y = Dense(48)(y)
        z = Dense(48)(z)
        x = Concatenate()([y, z])
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model


    def build_difornet52a(inputShape):
        # specify the inputs for the feature extractor network
        dropout = 0 #.1
        dropoutdense = 0 #0.5
        inputs = Input((None, None, inputShape[2]))
        x = tf.keras.layers.Masking()(inputs)
        x = Conv2D(96, (11, 11), strides=(1, 1), padding="same", activation="elu",
                   name="conv1")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Dropout(dropout)(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool1")(x)

        x = Conv2D(24, (2, 2), strides=(1, 1), padding="same", activation="elu",
                   name="conv2")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Dropout(dropout)(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool2")(x)
        x = Conv2D(48, (3, 3), strides=(1, 1), padding="same", activation="elu",
                   name="conv3")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Dropout(dropout)(x)
        x = Conv2D(48, (3, 3), strides=(1, 1), padding="same", activation="elu",
                   name="conv4")(x)

        x = GlobalMaxPooling2D(name="globalMP6")(x)

        x = Dense(256)(x)
        x = Dense(48)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def build_difornet52b(inputShape):
        # specify the inputs for the feature extractor network
        dropout = 0 #.1
        dropoutdense = 0 #0.5
        inputs = Input((None, None, inputShape[2]))
        x = tf.keras.layers.Masking()(inputs)

        x = Conv2D(96, (11, 11), strides=(1, 1), padding="same", activation="elu",
                    name="conv1s")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Dropout(dropout)(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool1s")(x)

        x = Conv2D(24, (2, 2), strides=(1, 1), padding="same", activation="elu",
                    name="conv2s")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Dropout(dropout)(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool2s")(x)
        x = Conv2D(48, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv3s")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Dropout(dropout)(x)
        x = Conv2D(48, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv4s")(x)

        x = BatchNormalization(axis=-1)(x)
        x = GlobalAveragePooling2D(name="globalAP6")(x)
        x = Dense(256)(x)
        x = Dense(48)(x)
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    # @tf.function(jit_compile=True)
    def build_difornet53(inputShape):
        # specify the inputs for the feature extractor network
        dropout = 0 #.1
        dropoutdense = 0 #0.5
        inputs = Input((None, None, inputShape[2]))
        x = tf.keras.layers.Masking()(inputs)
        s = x
        x1 = Conv2D(96, (11, 11), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(x)
        x1 = BatchNormalization(axis=-1)(x1)
        x = Dropout(dropout)(x1)
        x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool1")(x)

        x3 = Conv2D(24, (2, 2), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x3 = BatchNormalization(axis=-1)(x3)
        x = Dropout(dropout)(x3)
        x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool2")(x)
        x5 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv3")(x4)
        x5 = BatchNormalization(axis=-1)(x5)
        x = Dropout(dropout)(x5)
        x6 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv4")(x)

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)


        s1 = Conv2D(96, (11, 11), strides=(1, 1), padding="same", activation="elu",
                    name="conv1s")(s)
        # x1 = BatchNormalization(axis=-1)(x1)
        s = Dropout(dropout)(s1)
        s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool1s")(s)

        s3 = Conv2D(24, (2, 2), strides=(1, 1), padding="same", activation="elu",
                    name="conv2s")(s2)
        s3 = BatchNormalization(axis=-1)(s3)
        s = Dropout(dropout)(s3)
        s4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool2s")(s)
        s5 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv3s")(s4)
        s5 = BatchNormalization(axis=-1)(s5)
        s = Dropout(dropout)(s5)
        s6 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv4s")(s)

        s6 = BatchNormalization(axis=-1)(s6)
        z1 = GlobalAveragePooling2D(name="globalAP1")(s1)
        z3 = GlobalAveragePooling2D(name="globalAP3")(s3)
        z5 = GlobalAveragePooling2D(name="globalAP5")(s5)
        z6 = GlobalAveragePooling2D(name="globalAP6")(s6)
        # y10 = GlobalMaxPooling2D(name="globalMP10")(x10)

        # x = Concatenate()([y1, y3, y5, y6, z1, z3, z5, z6])
        # y = Concatenate()([y1, y3, y5, y6])
        # z = Concatenate()([z1, z3, z5, z6])
        # y = Concatenate()([y1, y3, y5, y6])
        # z = Concatenate()([z1, z3, z5, z6])
        # x = Dense(1024)(x)
        # x = Dropout(dropoutdense)(x)
        y = Dense(256)(y6)
        z = Dense(256)(z6)
        y = Dense(48, activation='sigmoid')(y)
        z = Dense(48, activation='sigmoid')(z)
        x = Concatenate()([y, z])
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model


    def build_difornet54(inputShape):
        # specify the inputs for the feature extractor network
        dropout = 0 #.1
        dropoutdense = 0 #0.5
        inputs = Input((None, None, inputShape[2]))
        # x = tf.keras.layers.Masking()(inputs)
        x = inputs
        s = x
        x1 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(x)
        x1 = BatchNormalization(axis=-1)(x1)
        if dropout>0:
            x1 = Dropout(dropout)(x1)
        x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool1")(x1)

        x3 = Conv2D(24, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        x3 = BatchNormalization(axis=-1)(x3)
        if dropout>0:
            x3 = Dropout(dropout)(x3)
        x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool2")(x3)
        x5 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv3")(x4)
        x5 = BatchNormalization(axis=-1)(x5)
        if dropout > 0:
            x5 = Dropout(dropout)(x5)
        x6 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv4")(x5)

        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)


        s1 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv1s")(s)
        s1 = BatchNormalization(axis=-1)(s1)
        if dropout > 0:
            s1 = Dropout(dropout)(s1)
        s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool1s")(s1)

        s3 = Conv2D(24, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv2s")(s2)
        s3 = BatchNormalization(axis=-1)(s3)
        if dropout>0:
            s3 = Dropout(dropout)(s3)
        s4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool2s")(s3)
        s5 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv3s")(s4)
        s5 = BatchNormalization(axis=-1)(s5)
        if dropout>0:
            s5 = Dropout(dropout)(s5)
        s6 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv4s")(s5)

        s6 = BatchNormalization(axis=-1)(s6)
        z6 = GlobalAveragePooling2D(name="globalAP6")(s6)
        # y10 = GlobalMaxPooling2D(name="globalMP10")(x10)

        # x = Concatenate()([y1, y3, y5, y6, z1, z3, z5, z6])
        # y = Concatenate()([y1, y3, y5, y6])
        # z = Concatenate()([z1, z3, z5, z6])
        # y = Concatenate()([y1, y3, y5, y6])
        # z = Concatenate()([z1, z3, z5, z6])
        # x = Dense(1024)(x)
        # x = Dropout(dropoutdense)(x)
        y = Dense(256)(y6)
        z = Dense(256)(z6)
        y = Dense(48, activation='sigmoid')(y)
        z = Dense(48, activation='sigmoid')(z)
        x = Concatenate()([y, z])
        # x= y
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model


    def build_difornet55(inputShape, dropout=0, dropoutdense=0, batch_normalization=False,
                         dense_layers=1, dense_units=256):
        # specify the inputs for the feature extractor network
        inputs = Input((None, None, inputShape[2]))
        x = tf.keras.layers.Masking()(inputs)
        s = x
        x1 = Conv2D(96, (11, 11), strides=(1, 1), padding="same", activation="elu",
                    name="conv1")(x)
        if batch_normalization:
            x1 = BatchNormalization(axis=-1)(x1)
        if dropout > 0:
            x = Dropout(dropout)(x1)
        x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool1")(x)

        x3 = Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv2")(x2)
        if batch_normalization:
            x3 = BatchNormalization(axis=-1)(x3)
        if dropout > 0:
            x = Dropout(dropout)(x3)
        x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool2")(x)
        x5 = Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv3")(x4)
        if batch_normalization:
            x5 = BatchNormalization(axis=-1)(x5)
        if dropout > 0:
            x = Dropout(dropout)(x5)
        x6 = Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv4")(x)

        y1 = GlobalMaxPooling2D(name="globalMP1")(x1)
        y3 = GlobalMaxPooling2D(name="globalMP3")(x3)
        y5 = GlobalMaxPooling2D(name="globalMP5")(x5)
        y6 = GlobalMaxPooling2D(name="globalMP6")(x6)


        s1 = Conv2D(96, (11, 11), strides=(1, 1), padding="same", activation="elu",
                    name="conv1s")(s)
        # x1 = BatchNormalization(axis=-1)(x1)
        if dropout > 0:
            s = Dropout(dropout)(s1)
        s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool1s")(s)

        s3 = Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv2s")(s2)
        if batch_normalization:
            s3 = BatchNormalization(axis=-1)(s3)
        if dropout > 0:
            s = Dropout(dropout)(s3)
        s4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool2s")(s)
        s5 = Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv3s")(s4)
        s5 = BatchNormalization(axis=-1)(s5)
        if dropout > 0:
            s = Dropout(dropout)(s5)
        s6 = Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation="elu",
                    name="conv4s")(s)

        if batch_normalization:
            s6 = BatchNormalization(axis=-1)(s6)
        z1 = GlobalAveragePooling2D(name="globalAP1")(s1)
        z3 = GlobalAveragePooling2D(name="globalAP3")(s3)
        z5 = GlobalAveragePooling2D(name="globalAP5")(s5)
        z6 = GlobalAveragePooling2D(name="globalAP6")(s6)
        # y10 = GlobalMaxPooling2D(name="globalMP10")(x10)

        # x = Concatenate()([y1, y3, y5, y6, z1, z3, z5, z6])
        # y = Concatenate()([y1, y3, y5, y6])
        # z = Concatenate()([z1, z3, z5, z6])
        # y = Concatenate()([y1, y3, y5, y6])
        # z = Concatenate()([z1, z3, z5, z6])
        # x = Dense(1024)(x)
        # x = Dropout(dropoutdense)(x)

        y=y6
        z=z6
        for i in range(dense_layers):
            y = Dense(dense_units)(y)
            z = Dense(dense_units)(z)
            if dropoutdense>0:
                y = Dropout(dropout)(y)
                z = Dropout(dropout)(z)
        y = Dense(48)(y)
        z = Dense(48)(z)
        x = Concatenate()([y, z])
        outputs = x
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model
