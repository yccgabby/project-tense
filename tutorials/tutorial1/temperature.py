import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt

# declare the two sets of data, one as input and one as output
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

for f, c in enumerate(celsius):
    print("{} degrees Celsius is equal to {} degrees Fahrenheit".format(c, fahrenheit[f]))

# features are the inputs to our mdoel 
# labels are the outputs our model predicts 
# example is a pair of inputs/outputs 

# Build a dense layer 
# units specifies the number of neurons, which represents how many internal variables the layer has to learn to solve the problem
# the input to this layer is a single value (input_shape)
dl = tf.keras.layers.Dense(units=1, input_shape=[1])

# assemble layers into a model
m = tf.keras.Sequential([dl])

# an alternative declaration method could've been:
# m = tf.keras.Sequential([
#   tf.keras.layers.Dense(units=1, input_shape=[1])
# ])

# compile the model 
# loss function: way of measuring how far off predictions are from desired outcomes
# optimizer function: way of adjusting internal values in order to reduce the loss
m.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

# train the model 
# epochs mean how many full iterations of all examples will be completed
# fit method returns a history object that can be plotted
stats = m.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("finished training")

plt.xlabel('Epoch Number')
plt.ylabel('Loss Magnitude')
plt.plot(stats.history['loss'])

print("let's see how effective the model is!")
print("100 degrees celsius is:", m.predict([100.0]))
print()

# for a single neuron with single input and output, the math looks for y = mx + b
print("these are layer variables: {}".format(dl.get_weights()))






