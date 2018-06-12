"""
floortype_cnn.py
Author: Jinyoung Lim
Date: June 2018

A convolutional neural network to classify floor types of Olin Rice. Based on  MNIST tensorflow tutorial (layer
architecture) and cat vs dog kaggle (preprocessing) as guides

Acknowledgements:
    Tensorflow MNIST cnn tutorial: https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/examples/tutorials/layers/cnn_mnist.py
    Cat vs Dog: https://pythonprogramming.net/convolutional-neural-network-kats-vs-dogs-machine-learning-tutorial/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import cv2
import random


BASE_PATH = '/Users/jlim2/PycharmProjects/tensorflow/markedFrames/'

TRAIN_DIR_PATH = os.path.join(
    BASE_PATH,
    "floorframes/")

CARPET_DIR_PATH = os.path.join(
    BASE_PATH,
    "carpet/")
TILE_DIR_PATH = os.path.join(
    BASE_PATH,
    "tile/")

TEST_DIR_PATH = os.path.join(
    BASE_PATH,
    "testframes/"
)

# [CARPET_YES, TILE_YES]
CARPET_LABEL = 0
TILE_LABEL = 1

IMAGE_SIZE = 50
LEARNING_RATE = 1e-3


MODEL_NAME = 'carpetvstile-{}-{}.model'.format(LEARNING_RATE, 'tfcnn')


tf.logging.set_verbosity(tf.logging.INFO)

def floortype_classifier_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    input_layer = tf.reshape(features["input"], [-1, IMAGE_SIZE, IMAGE_SIZE, 3], name="input")
    input_layer = tf.cast(input_layer, tf.float32)
    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 50, 50, 3]
    # Output Tensor Shape: [batch_size,  50, 50, 16]

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=16,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # print("IN floortype_classifier_model_fn  AFTER conv1")

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size,  50, 50, 16]
    # Output Tensor Shape: [batch_size, 25, 25, 16]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 25, 25, 16]
    # Output Tensor Shape: [batch_size, 25 * 25 * 16]
    pool1_flat = tf.reshape(pool1, [-1, 25 * 25 * 16])

    # Dense Layer
    # Densely connected layer with 514 neurons
    # Input Tensor Shape: [batch_size, 50 * 50 * 16]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool1_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.5 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer (unit==2 since there are two classes - TILE, CARPET)
    # Input Tensor Shape: [batch_size, 1024]
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
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
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


def create_train_data():
    """
    https://pythonprogramming.net/convolutional-neural-network-kats-vs-dogs-machine-learning-tutorial/
    :return:
    """
    training_data = []
    for filename in os.listdir(TRAIN_DIR_PATH):
        if filename in os.listdir(CARPET_DIR_PATH):
            label = CARPET_LABEL
        elif filename in os.listdir(TILE_DIR_PATH):
            label = TILE_LABEL

        path = os.path.join(TRAIN_DIR_PATH, filename)
        img = cv2.imread(filename=path)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        training_data.append([np.array(img), np.array(label)])
    random.shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def process_test_data():
    """
    https://pythonprogramming.net/convolutional-neural-network-kats-vs-dogs-machine-learning-tutorial/
    :return:
    """
    testing_data = []
    for filename in os.listdir(TEST_DIR_PATH):
        path = os.path.join(TEST_DIR_PATH, filename)
        image = cv2.imread(filename=path)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        testing_data.append([np.array(image), filename])

    random.shuffle(testing_data)
    np.save("test_data.npy", testing_data)
    return testing_data


def train(train_data, model):
    tf.reset_default_graph()    #TODO: IS THIS CRITICAL?

    # Load training and eval data
    train = train_data[:-40]
    eval = train_data[-40:]

    train_images = np.array([i[0] for i in train]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
    train_labels = np.array([i[1] for i in train])

    test_images = np.array([i[0] for i in eval]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
    test_labels = np.array([i[1] for i in eval])


    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    # print("AFTER tensors_to_log")

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)



    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"input": train_images},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True
    )


    model.train(
        input_fn=train_input_fn,
        steps=1000,
        hooks=[logging_hook]
    )


    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"input": test_images},
        y=test_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = model.evaluate(input_fn=eval_input_fn)
    print(eval_results)

def test(test_data, model):
    test_images = np.array([i[0] for i in test_data]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)

    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1)
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"input": test_images},
        y=None,
        batch_size=128,
        num_epochs=1,
        shuffle=False,
        queue_capacity=1000,
        num_threads=1
    )
    pred_results = model.predict(
        input_fn=pred_input_fn
    )



    pred_list = list(pred_results)
    num_correct_preds = 0
    num_test = 0
    for i, data in enumerate(test_data):
        # image = data[0]
        # image = cv2.resize(image, (IMAGE_SIZE*10, IMAGE_SIZE*10))
        filename = data[1]
        imagefile = os.path.join(
            TEST_DIR_PATH,
            filename
        )
        image = cv2.imread(imagefile)


        # Center Text on Image: https://gist.github.com/xcsrz/8938a5d4a47976c745407fe2788c813a
        font = cv2.FONT_HERSHEY_SIMPLEX
        pred_text = "carpet" if pred_list[i]["classes"] == CARPET_LABEL else "tile"


        if "tile" in filename:
            answer_text = "tile"
        else:
            answer_text = "carpet"

        if pred_text == answer_text:
            num_correct_preds += 1

        pred_str = pred_text
        print("Answer: " + answer_text + " vs. Prediction: " + pred_str)


        text_size = cv2.getTextSize(pred_str, font, 1, 2)[0]
        text_x = int((image.shape[1] - text_size[0]) / 2)
        text_y = int((image.shape[0] + text_size[1]) / 2)

        cv2.putText(
            img=image,
            text=pred_str,
            org=(text_x, text_y),
            fontFace=font,
            fontScale=0.8,
            color=(0, 255, 0),
            thickness=2)

        cv2.imshow("Test Image {}".format(num_test), image)

        key = cv2.waitKey(0)

        num_test += 1
    cv2.destroyAllWindows()
    print("Accuracy: {}% (N={})".format(int(num_correct_preds/num_test * 100), num_test))


def main(unused_argv):
    # Get data
    # train_data = create_train_data()
    test_data = process_test_data()
    # If you have already created the dataset:
    train_data = np.load('train_data.npy')
    # test_data = np.load('test_data.npy')

    # Make estimator
    FLOOR_CLASSIFIER = tf.estimator.Estimator(
        model_fn=floortype_classifier_model_fn,
        model_dir="/Users/jlim2/PycharmProjects/tensorflow/floortype_model_moredata_061218"
    )

    # Train or Test
    # train(train_data, model=FLOOR_CLASSIFIER)
    test(test_data, model=FLOOR_CLASSIFIER)



if __name__ == "__main__":
    tf.app.run()
