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
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)), #transform 2d array of 28x28 to 1d array of 784 
    tf.keras.layers.Dense(126, activation=tf.nn.relu), 
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

m.compile(optimizer='adam', loss="sparse_categorical_crossentropy",metrics=['accuracy'])

BATCH_SIZE = 32
train_dataset = train_dataset.repeat().shuffle(train_sets).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

m.fit(train_dataset, epochs=10, steps_per_epoch=math.ceil(train_sets/BATCH_SIZE))

test_loss, test_accuracy = m.evaluate(test_dataset, steps=math.ceil(test_sets/32))
print('Accuracy on test dataset: ', test_accuracy)

for test_images, test_labels in test_dataset.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = m.predict(test_images)

print(predictions.shape) 

print(np.argmax(predictions[0]))

print(test_labels[0])









