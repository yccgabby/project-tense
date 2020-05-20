from __future__ import absolute_import, division, print_function

# import tf and database
import tensorflow as tf 
import tensorflow_datasets as tfds 

# helper libraries
import math
import numpy as np 
import matplotlib.pyplot as plt 

# improve progress bar displays
import tqdm 
import tqdm.auto 
tqdm.tqdm = tqdm.auto.tqdm 

# download and reuse images of clothes 
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

# determine output value names and training/testing sets
class_names = ['Tshirts/tops', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_sets = metadata.splits['train'].num_examples
test_sets = metadata.splits['test'].num_examples

def normalize(images, labels): 
    images = tf.cast(images, tf.float32)
    images /= 255 # normalize rgb values to 0-1 
    return images, labels

# the map function applies the normalize function to each element in the train 
# and test datasets 
train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

m = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu, input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

m.compile(optimizer='adam', 
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])

BATCH_SIZE = 32
train_dataset = train_dataset.repeat().shuffle(train_sets).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

m.fit(train_dataset, epochs=10, steps_per_epoch=math.ceil(train_sets/BATCH_SIZE))

test_loss, test_accuracy = m.evaluate(test_dataset, steps=math.ceil(test_sets/32))
print('accuracy on test dataset: ', test_accuracy)

