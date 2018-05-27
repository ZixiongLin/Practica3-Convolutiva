# -*- coding: utf-8 -*-

# Sample code to use string producer.

"""Librerias a usar tensorflow y numpy"""
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plot

"""clase one_hot, es una lista de muchos 0. y un 1."""


def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    o_h = np.zeros(n)
    o_h[x] = 1
    return o_h


num_classes = 10
batch_size = 4

# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------
"""Clase dataSource"""

"""Le pasas la fuente de datos y el batch_size"""


def dataSource(paths, batch_size):
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = []
    label_batch_list = []
    """enumerate()-> crea 2 valores i:iteracion y p: path del fichero"""
    for i, p in enumerate(paths):
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        """ _ -> indica que es una variable que no se va a emplear"""
        _, file_image = reader.read(filename_queue)
        """
            image-> imagen decodificada
            label-> coge la iteracion y la usa para crear el one_hot
        """
        image, label = tf.image.decode_jpeg(file_image), one_hot(int(i), num_classes)
        """¿?"""
        """Crop -> recorta
        pad-> rellena"""
        image = tf.image.resize_image_with_crop_or_pad(image, 80, 140)
        image = tf.image.rgb_to_grayscale(image)
        """La imagen viene en un formato 80*140 pero necesitamos 80*140*1"""
        image = tf.reshape(image, [80, 140, 1])
        image = tf.to_float(image) / 255. - 0.5
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch


# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------
"""Funcion modelo"""


def myModel(X, reuse=False):
    """Estructuras with es una estructura de contexto"""
    """Todo lo que hay dentro con el scope las variables de """
    with tf.variable_scope('ConvNet', reuse=reuse):
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)
        o5 = tf.layers.conv2d(inputs=o4, filters=128, kernel_size=3, activation=tf.nn.relu)
        o6 = tf.layers.max_pooling2d(inputs=o5, pool_size=2, strides=2)
        hf = tf.layers.flatten(o6)
        h = tf.layers.dense(inputs=hf, units=50, activation=tf.nn.relu)
        h2 = tf.layers.dense(inputs=h, units=30, activation=tf.nn.relu)

        y = tf.layers.dense(inputs=h2, units=num_classes, activation=tf.nn.softmax)
        """Pero resultado elu"""
        """mejor estabilidad -> tanh"""
        """softmax¿?"""
        # o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.selu)
        # o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        # o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.selu)
        # o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)
        # o5 = tf.layers.conv2d(inputs=o4, filters=128, kernel_size=3, activation=tf.nn.selu)
        # o6 = tf.layers.max_pooling2d(inputs=o5, pool_size=2, strides=2)
        #
        # h = tf.layers.dense(inputs=tf.reshape(o6, [batch_size * num_classes, 8 * 15 * 128]), units=10,
        #                     activation=tf.nn.selu)
        # h2 = tf.layers.dense(inputs=h, units=25, activation=tf.nn.tanh)
        #
        # h3 = tf.layers.dense(inputs=h2, units=25, activation=tf.nn.tanh)
        #
        # y = tf.layers.dense(inputs=h3, units=num_classes, activation=tf.nn.softmax)
    return y

#
example_batch_train, label_batch_train = dataSource(["train/0/*.jpg", "train/1/*.jpg",
                                                        "train/2/*.jpg", "train/3/*.jpg",
                                                        "train/4/*.jpg", "train/5/*.jpg",
                                                        "train/6/*.jpg", "train/7/*.jpg",
                                                        "train/8/*.jpg", "train/9/*.jpg"
                                                        ], batch_size=batch_size)

example_batch_valid, label_batch_valid = dataSource(["valid/0/*.jpg", "valid/1/*.jpg",
                                                        "valid/2/*.jpg", "valid/3/*.jpg",
                                                        "valid/4/*.jpg", "valid/5/*.jpg",
                                                        "valid/6/*.jpg", "valid/7/*.jpg",
                                                        "valid/8/*.jpg", "valid/9/*.jpg"], batch_size=batch_size)

example_batch_test, label_batch_test = dataSource(["test/0/*.jpg", "test/1/*.jpg",
                                                        "test/2/*.jpg", "test/3/*.jpg",
                                                        "test/4/*.jpg", "test/5/*.jpg",
                                                        "test/6/*.jpg", "test/7/*.jpg",
                                                       "test/8/*.jpg", "test/9/*.jpg"], batch_size=batch_size)
#
# example_batch_train, label_batch_train = dataSource(["train/0/*.jpg", "train/1/*.jpg",
#                                                         ], batch_size=batch_size)
#
# example_batch_valid, label_batch_valid = dataSource(["valid/0/*.jpg", "valid/1/*.jpg",
#                                                         ], batch_size=batch_size)
#
# example_batch_test, label_batch_test = dataSource(["test/0/*.jpg", "test/1/*.jpg",
#                                                       ], batch_size=batch_size)


example_batch_train_predicted = myModel(example_batch_train, reuse=False)
example_batch_valid_predicted = myModel(example_batch_valid, reuse=True)
example_batch_test_predicted = myModel(example_batch_test, reuse=True)

cost = tf.reduce_sum(tf.square(example_batch_train_predicted - tf.cast(label_batch_train,tf.float32)))
cost_valid = tf.reduce_sum(tf.square(example_batch_valid_predicted - tf.cast(label_batch_valid,tf.float32)))
cost_test = tf.reduce_sum(tf.square(example_batch_valid_predicted - tf.cast(label_batch_test,tf.float32)))
#cost = tf.reduce_mean(-tf.reduce_sum(tf.cast(label_batch_train,tf.float32) * tf.log(example_batch_train_predicted), reduction_indices=[1]))
#cost_valid = tf.reduce_mean(-tf.reduce_sum(tf.cast(label_batch_valid,tf.float32) * tf.log(example_batch_valid_predicted), reduction_indices=[1]))
#cost_test = tf.reduce_mean(-tf.reduce_sum(tf.cast(label_batch_test,tf.float32) * tf.log(example_batch_test_predicted), reduction_indices=[1]))

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
#optimizer = tf.train.RMSPropOptimizer(learning_rate=0.1).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
"""Nota: el learning rate si aumenta el error aumenta"""
#optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate=0.05).minimize(cost)
# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

saver = tf.train.Saver()

with tf.Session() as sess:
    file_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    error_train = []
    error_valid = []
    diferencia_error = 100.0

    """Primer entrenamiento"""
    error_train.append(sess.run(cost))
    """Validacion"""
    print("Iter:", 0, "---------------------------------------------")
    print(sess.run(label_batch_valid))
    print(sess.run(example_batch_valid_predicted))
    error_valid.append(sess.run(cost_valid))
    print("Error:", error_valid[-1])
    iteration = 1

    """Rutina de entrenamiento"""
    while diferencia_error > (1/(10**5)):
        sess.run(optimizer)
        """Validacion"""
        print("Iter:", iteration, "---------------------------------------------")
        label_valid = sess.run(label_batch_valid)
        predicted_valid = sess.run(example_batch_valid_predicted)
        # for b, r in zip(label_valid, predicted_valid):
        #     print(b, "--->", r)
        error_valid.append(sess.run(cost_valid))
        error_train.append(sess.run(cost))
        print("Error entrenamiento:", error_train[-1])
        print("Error:", error_valid[-1])
        diferencia_error = abs(error_valid[-2] - error_valid[-1])
        print("Diferencia de error: ", diferencia_error)
        iteration += 1
    total = 0.0
    error = 0.0

    test_data = sess.run(label_batch_test)
    test_hoped = sess.run(example_batch_test_predicted)
    for real_data, hoped in zip(test_data, test_hoped):
        if np.argmax(real_data) != np.argmax(hoped):
            error += 1
        total += 1
    fallo = error / total * 100
    print("El porcentaje de error es: ", fallo, "% y el de exito ", (100 - fallo), "%")

    tr_handle, = plot.plot(error_train)
    vl_handle, = plot.plot(error_valid)
    plot.legend(handles=[tr_handle, vl_handle],
                labels=['Error entrenamiento', 'Error validacion'])
    plot.title("Learning rate = 0.001")
    plot.show()
    plot.savefig('Grafica_entrenamiento.png')

    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)

    coord.request_stop()
    coord.join(threads)
