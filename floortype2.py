

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import cv2

tf.logging.set_verbosity(tf.logging.INFO)
""" Matching layer shapes method (https://stackoverflow.com/questions/45903774/convolutional-network-error-tensorflow-value-error)
Conv_1 28, 28, 1 -> 28, 28, 10
Pool_1 28, 28, 10 -> 14, 14, 10
Conv_2 14, 14, 10 -> 14, 14, 10
Pool_2 14, 14, 10 -> 7, 7, 10
Flatten-> FC Layer 7, 7, 10 = 7 * 7 * 10 = 490 -> 1024



Conv_1 480, 640, 3 --> 480, 640, 48
Pool_1 480, 640, 48 -> 240, 320, 48

"""

def getImageShape():
    return 480, 640, 3
    #TODO: could image size be different from actual image size?

def floortype_classifier_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    h, w, d = getImageShape()
    input_layer = tf.reshape(features["x"], [-1, h, w, d], name="input")
    input_layer = tf.cast(input_layer, tf.float32)
    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 480, 640, 3]
    # Output Tensor Shape: [batch_size,  480, 640, 16]
    print("IN floortype_classifier_model_fn  AFTER input_layer")
    print(input_layer.dtype)

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=16,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    print("IN floortype_classifier_model_fn  AFTER conv1")

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size,  480, 640, 16]
    # Output Tensor Shape: [batch_size, 240, 320, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 240, 320, 32]
    # Output Tensor Shape: [batch_size, 240 * 320 * 32]
    pool1_flat = tf.reshape(pool1, [-1, 240 * 320 * 48])

    # Dense Layer
    # Densely connected layer with 514 neurons
    # Input Tensor Shape: [batch_size, 240 * 320 * 32]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool1_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.5 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer (unit==2 since there are two classes - TILE, CARPET)
    # Input Tensor Shape: [batch_size, 512]
    # Output Tensor Shape: [batch_size, 2]
    logits = tf.layers.dense(inputs=dropout, units=2)


    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


CARPET_LABEL = 0
TILE_LABEL = 1

def get_filenames_and_labels():
    carpetPath = os.path.join(
        '/Users/jlim2/PycharmProjects/tensorflow/markedFrames/',
        "carpet/")
    tilePath = os.path.join(
        '/Users/jlim2/PycharmProjects/tensorflow/markedFrames/',
        "tile/")

    # labelDict = dict()
    labels = []
    filenames = []

    carpetFramenames = os.listdir(carpetPath)
    # Clean up non-jpg files and put into label dict
    for frame in carpetFramenames:
        filename = os.path.join(carpetPath, frame)
        if (frame.endswith("jpg")):
            filenames.append(filename)
            labels.append(CARPET_LABEL)
        else:
            print("Non-jpg file", filename, "while reading in frame data, skipping...")
    tileFramenames = os.listdir(tilePath)
    for frame in tileFramenames:
        filename = os.path.join(tilePath, frame)
        if (frame.endswith("jpg")):
            filenames.append(filename)
            labels.append(TILE_LABEL)
        else:
            print("Non-jpg file", filename, "while reading in frame data, skipping...")

    return filenames, labels


def filenames_to_tensor_numpy(filenames):
    images_numpy = []
    for i in range(len(filenames)):
        # image = cv2.imread(filenames[i])
        # images_numpy.append(image)
        image = tf.image.decode_jpeg(filenames[i])
        image = tf.cast(image, tf.float32)
        # image_np = cv2.imread(filenames[i])
        # cv2.imshow("oriImg",image_np)
        # cv2.waitKey(10)
        images_numpy.append(image)
        # print(np.asarray(images_numpy))
    return np.asarray(images_numpy)



def filenames_to_filenamequeue(filenames):
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError("Failed to find file: " + f)
    with tf.name_scope("train_images"):
        # Create a queue that produces the filenames to read
        filename_queue = tf.train.string_input_producer(filenames)
    return filename_queue


def main(unused_argv):
    # Load training and eval data
    filenames, labels = get_filenames_and_labels()
    print("AFTER get_filenames+and+labels")

    floor_images_numpy = filenames_to_tensor_numpy(filenames)
    print("AFTER floor_images_numpy")

    floor_labels_numpy = np.asarray(labels, dtype=np.int32)
    print("AFTER floor_labels_numpy")
    # Create the Estimator
    floor_classifier = tf.estimator.Estimator(
        model_fn=floortype_classifier_model_fn,
        model_dir="/Users/jlim2/PycharmProjects/tensorflow/"
    )
    print("AFTER floor_classifier")


    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    print("AFTER tensors_to_log")

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1)

    print("AFTER logging_hook")


    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": floor_images_numpy},
        y=floor_labels_numpy,
        batch_size=int(10),
        num_epochs=None,
        shuffle=True
    )
    print("AFTER train_input_fn")


    floor_classifier.train(
        input_fn=train_input_fn,
        steps=1000,
        hooks=[logging_hook]
    )
    print("AFTER floor_classifier.train")


    # # Evaluate the model and print results
    # eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": eval_data},
    #     y=eval_labels,
    #     num_epochs=1,
    #     shuffle=False)
    # eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    # print(eval_results)


if __name__ == "__main__":
    tf.app.run()
