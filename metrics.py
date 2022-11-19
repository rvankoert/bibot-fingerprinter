import tensorflow.keras.backend as K
import tensorflow as tf


def cosine_distance(vectors):
    x, y = vectors
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    print(x)
    result = K.mean(x * y, axis=-1, keepdims=True)
    print(result)
    return result


def cos_dist_output_shape(shapes):
    return shapes[0]


def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1,
                       keepdims=True)
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))


def contrastive_loss(y, predictions, margin=1.25):
    # explicitly cast the true class label data type to the predicted
    # class label data type (otherwise we run the risk of having two
    # separate data types, causing TensorFlow to error out)
    y = tf.cast(y, float)
    # calculate the contrastive loss between the true labels and
    # the predicted labels
    squared_predictions = K.square(predictions)
    squared_margin = K.square(K.maximum(margin - predictions, K.epsilon()))
    loss = K.mean(y * squared_predictions + (1 - y) * squared_margin)
    # return the computed contrastive loss to the calling function
    return loss


def contrastive_loss_new(y, preds):
    # explicitly cast the true class label data type to the predicted
    # class label data type (otherwise we run the risk of having two
    # separate data types, causing TensorFlow to error out)
    y = tf.cast(y, tf.float16)
    margin = 0.4
    positive_loss = K.maximum((y - 1) * (margin - preds), K.epsilon())
    negative_loss = K.maximum(-y * (preds - (1 - margin)), K.epsilon())
    loss = K.mean(positive_loss + negative_loss)
    # return the computed contrastive loss to the calling function
    return loss


def accuracy(y_true, y_pred):
    """Compute classification accuracy with a fixed threshold on distances."""
    return K.mean(K.equal(y_true, K.cast(y_pred > 0.5, y_true.dtype)))


def average(y_true, y_pred):
    """Compute classification accuracy with a fixed threshold on distances."""
    return K.mean(y_pred)
