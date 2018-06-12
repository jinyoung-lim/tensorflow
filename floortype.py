"""---------------------------------------------------------------------------------------------------------------------
floortype.py
Use CNN with TensorFlow to classify carpet and tile.
Author: Jinyoung JJ Lim
Date: Jun 2018
Reference: tensorflow models

---------------------------------------------------------------------------------------------------------------------"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import cv2
import os
import time
from datetime import datetime



FLAGS = tf.app.flags.FLAGS
# Basic model parameters
tf.app.flags.DEFINE_boolean(
    flag_name='batch_size',
    default_value=2,
    # default_value=64
    docstring="""Number of images to process in a batch"""
)
tf.app.flags.DEFINE_string(
    flag_name='data_dir',
    default_value='/Users/jlim2/PycharmProjects/tensorflow/markedFrames/',
    docstring="""Path to the marked frames directory"""
)

tf.app.flags.DEFINE_string(
    flag_name='carpet_dir',
    # default_value='carpet/',
    default_value='little_carpet/',
    docstring="""Path to the carpet frames directory"""
)

tf.app.flags.DEFINE_string(
    flag_name='tile_dir',
    # default_value='tile/',
    default_value='little_tile/',
    docstring="""Path to the tile frames directory"""
)

tf.app.flags.DEFINE_string(
    flag_name='floorframes_dir',
    # default_value='floorframes/',
    default_value='littlist_data',
    docstring="""Path to the all floor frames directory"""
)


tf.app.flags.DEFINE_integer(
    'log_frequency',
    1,
    """How often to log results to the console."""
)

NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 100
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10

NUM_EPOCHS_PER_DECAY = 100.0  # Epochs after which learning rate decays.
INITIAL_LEARNING_RATE = 0.1
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.



IMAGE_SIZE = 24 # following cifar10_input.py from tensorflow models



def getImageShape():
    if (not FLAGS.data_dir or not FLAGS.carpet_dir or not FLAGS.tile_dir):
        raise ValueError("Please supply a data_dir")
    # if (not FLAGS.data_dir or not FLAGS.carpet_dir or not FLAGS.tile_dir):
    #     raise ValueError("Please supply a data_dir")
    # filenames = os.listdir(FLAGS.data_dir + FLAGS.carpet_dir)
    # # Clean up non-jpg files
    # for file in filenames:
    #     if (not file.endswith("jpg")):
    #         filenames.remove(file)
    # im = cv2.imread(FLAGS.data_dir + FLAGS.carpet_dir + filenames[0])
    # (h, w, d) = im.shape
    #TODO: Undo hard-coding
    return 480, 640, 3

def convertToExample():
    """http://yeephycho.github.io/2016/08/15/image-data-in-tensorflow/"""

def getFilenameQueueAndLabelDict():
    carpetPath = os.path.join(FLAGS.data_dir, FLAGS.carpet_dir)
    tilePath = os.path.join(FLAGS.data_dir, FLAGS.tile_dir)

    labelDict = dict()
    filenames = []

    carpetFramenames = os.listdir(carpetPath)
    # Clean up non-jpg files and put into label dict
    for framename in carpetFramenames:
        filename = os.path.join(carpetPath, framename)
        if (framename.endswith("jpg")):
            filenames.append(filename)
            labelDict[filename] = 0    # carpet label == 0


    tileFramenames = os.listdir(tilePath)
    for framename in tileFramenames:
        filename = os.path.join(tilePath, framename)
        if (framename.endswith("jpg")):
            filenames.append(filename)
            labelDict[filename] = 1    # tile label == 1

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError("Failed to find file: " + f)

    with tf.name_scope("input"):
        # Create a queue that produces the filenames to read
        filename_queue = tf.train.string_input_producer(filenames)
    print(filename_queue)
    print(labelDict)
    return filename_queue, labelDict



def read_markedFrames(filename_queue):
    """Returns an object representing a single example"""
    h, w, d = getImageShape()

    class FloorTypeRecord(object):
        pass
    record = FloorTypeRecord()

    label_bytes = 1
    record.height = h
    record.width = w
    record.depth = d
    image_bytes = record.height * record.width * record.depth
    record_bytes = label_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the  format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    record.key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    # Label bytes
    record.label = tf.cast(
        tf.strided_slice(record_bytes,
                         [0], [label_bytes]),
        tf.int32
    )

    # Remaining bytes (Image bytes)
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes,
                         [label_bytes], [label_bytes + image_bytes]),
        [record.depth, record.height, record.width] # reshape [d * h * w] to [d, h, w]
    )

    record.uint8image = tf.transpose(depth_major, [1, 2, 0]) # [d, h, w] to [h, w, d]
    return record




def training_inputs():
    """Construct distorted imput for training"""
    batch_size = FLAGS.batch_size

    filename_queue, label_dict = getFilenameQueueAndLabelDict()
    record = read_markedFrames(filename_queue=filename_queue)
    print(record)

    reshaped_image = tf.cast(record.uint8image, tf.float32)
    # DEBUG
    print("DEBUG")
    image_tensor = tf.image.convert_image_dtype(record.uint8image, dtype=tf.uint8)
    print("sess begin")
    sess = tf.Session()
    with sess.as_default():
        image_numpy = image_tensor.eval(session=sess)
        print(sess.run(image_numpy))
    # with sess.as_default():
    #     image_numpy = image_tensor.eval()
    #     print("image numpy: ", image_numpy)
    #     sess.close()
    sess.close()
    print("session closed")
    # cv2.imshow("tf", image_numpy)
    # cv2.waitKey(10)

    # with tf.name_scope('data_augmentation'):
    #     h = IMAGE_SIZE
    #     w = IMAGE_SIZE
    #
    #     # Randomly crop a [h, w] section of the image
    #     distorted_image = tf.random_crop(reshaped_image, [h, w, 3])
    #
    #     # Randomly flip the image horizontally
    #     distorted_image = tf.image.random_flip_left_right(distorted_image)
    #
    #     # Randomly change brightness
    #     distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    #
    #     # Randomly change contrast
    #     distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    #
    #     # Standardize the image
    #     float_image = tf.image.per_image_standardization(distorted_image)
    #
    #     # Set the shapes of tensors.
    #     float_image.set_shape([h, w, 3])
    #     record.label.set_shape([1])
    #
    #
    #     # Ensure that the random shuffling has good mixing properties.
    #     min_fraction_of_examples_in_queue = 0.4
    #     min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
    #                              min_fraction_of_examples_in_queue)
    #     print ('Filling queue with %d floor images before starting to train. '
    #            'This will take some time.' % min_queue_examples)

    with tf.name_scope('data'):
        # Set the shapes of tensors.
        reshaped_image.set_shape([480, 640, 3])
        record.label.set_shape([1])

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        # min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
        #                          min_fraction_of_examples_in_queue)
        min_queue_examples = 2
        print('Filling queue with %d floor images before starting to train. '
              'This will take some time.' % min_queue_examples)

    # # Generate a batch of images and labels by building up a queue of examples.
    # images, labels =  _generate_image_and_label_batch(float_image, record.label,
    #                                        min_queue_examples, batch_size,
    #                                        shuffle=True)
    # data_dir = os.path.join(FLAGS.data_dir, FLAGS.floorframes_dir)

    # Generate a batch of images and labels by building up a queue of examples.
    images, labels = _generate_image_and_label_batch(reshaped_image, record.label,
                                                     min_queue_examples, batch_size,
                                                     shuffle=True)
    print(images)
    print(labels)
    return images, labels

def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle=True):
    num_preprocess_threads = 1
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples
    )

    # Display the training images in the visualizer
    tf.summary.image('image', images)

    return images, tf.reshape(label_batch, [batch_size])


def train():
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        images, labels = training_inputs()

        ############################################################################################
        # conv1
        with tf.variable_scope('conv1') as scope:
            kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, 64],
                                         stddev=5e-2,
                                         wd=None)
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv1)
        # pool1
        pool1 = tf.nn.max_pool(
            conv1,
            ksize=[1, 3, 3, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pool1')

        # norm1
        norm1 = tf.nn.lrn(
            pool1,
            4,
            bias=1.0,
            alpha=0.001 / 9.0,
            beta=0.75,
            name='norm1')

        # # conv2
        # with tf.variable_scope('conv2') as scope:
        #     kernel = _variable_with_weight_decay('weights',
        #                                          shape=[5, 5, 64, 64],
        #                                          stddev=5e-2,
        #                                          wd=None)
        #     conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        #     biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        #     pre_activation = tf.nn.bias_add(conv, biases)
        #     conv2 = tf.nn.relu(pre_activation, name=scope.name)
        #     _activation_summary(conv2)
        #
        # # norm2
        # norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
        #                   name='norm2')
        # # pool2
        # pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
        #                        strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # local3
        with tf.variable_scope('local3') as scope:
            # Move everything into depth so we can perform a single matrix multiply.
            # reshape = tf.reshape(pool2, [images.get_shape().as_list()[0], -1])
            reshape = tf.reshape(pool1, [images.get_shape().as_list()[0], -1])

            dim = reshape.get_shape()[1].value
            weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                                  stddev=0.04, wd=0.004)
            biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
            local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
            _activation_summary(local3)

        # local4
        with tf.variable_scope('local4') as scope:
            weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                                  stddev=0.04, wd=0.004)
            biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
            local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
            _activation_summary(local4)

        # linear layer(WX + b),
        # We don't apply softmax here because
        # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
        # and performs the softmax internally for efficiency.
        with tf.variable_scope('softmax_linear') as scope:
            weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                                  stddev=1 / 192.0, wd=None)
            biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                      tf.constant_initializer(0.0))
            softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
            _activation_summary(softmax_linear)

        ############################################################################################
        logits = softmax_linear
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits,
            name='cross_entropy_per_example'
        )
        cross_entropy_mean = tf.reduce_mean(
            cross_entropy, name='cross_entropy'
        )
        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')


        ############################################################################################
        num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        # NUM_EPOCHS_PER_DECAY = 100.0  # Epochs after which learning rate decays.
        # INITIAL_LEARNING_RATE = 0.1
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

        lr = tf.train.exponential_decay(
            INITIAL_LEARNING_RATE,
            global_step,
            decay_steps,
            LEARNING_RATE_DECAY_FACTOR,
            staircase=True
        )

        tf.summary.scalar('learning_rate', lr)

        loss_averages_op = _add_loss_summaries(loss)

        # Compute gradients.
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.GradientDescentOptimizer(lr)
            grads = opt.compute_gradients(loss)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        with tf.control_dependencies([apply_gradient_op]):
            variables_averages_op = variable_averages.apply(tf.trainable_variables())
        train_op = variables_averages_op

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1
                self._start_time = time.time()
                format_str = ('%s: step %d')
                print(format_str % (datetime.now(), self._step))

            def before_run(self, run_context):
                self._step += 1
                format_str = ('%s: step %d')
                print(format_str % (datetime.now(), self._step))
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.data_dir,
            hooks=[tf.train.StopAtStepHook(last_step=1000),
                   tf.train.NanTensorHook(loss),
                   _LoggerHook()],
            config=tf.ConfigProto(
                log_device_placement=False
            )
        ) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def _variable_with_weight_decay(name, shape, stddev, wd):
    dtype = tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(
      x.op.name + '/activations',
      x
  )
  tf.summary.scalar(
      x.op.name + '/sparsity',
      tf.nn.zero_fraction(x)
  )


def print_activations(t):
    """from alexnet_benchmark.py"""
    print(t.op.name, ' ', t.get_shape().as_list())


def main(argv=None):
    # if tf.gfile.Exists(FLAGS.floorframes_dir):
    #     tf.gfile.DeleteRecursively(FLAGS.floorframes_dir)
    # tf.gfile.MakeDirs(FLAGS.floorframes_dir)





    with tf.Session() as sess:
        tf.global_variables_initializer().run()




if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.app.run()