import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os.path
from datetime import date

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU

def preparedata(data, stpback, stpfor):
	'''
	This function prepares the data in the format it needs to be in to be fed to the keras layers
	'''
	state = []
	objective = []

	for i in range(len(data) - stpback - stpfor - 1):
		state.append(data[i: i + stpback])
		objective.append(data[i + stpback : i + stpback + stpfor])

	state = np.asarray(state)
	objective = np.asarray(objective)

	instances = state.shape[0]
	state = state.reshape(instances, stpback, 1)

	return state, objective

start_time = time.time()


batch = 10
epochs = 50
stepsforward = 50
stepsback = 50
'''
This variable (stepsback) represents how many months before the current the network will look back to make the prediction for the next.
In principle, the larger this value, the more accurate the prediction will be. However, the program gets slower to run. 
Furthermore, for too large values, one might encounter the problem of vanishing or exploding gradients when backpropagating,
which actually reduces the accuracy of the network's predictions.
'''

np.random.seed(42)

spotsdata = np.loadtxt('sunspot_monthly.csv', delimiter=';') # Loading the sunspots data
spotmean = spotsdata[:,3]

maxval = max(spotmean)
spotmean_norm = spotmean / maxval # Normalise the values to lie between 0 and 1

# I've chosen to use 70% of the data for training and 30% for testing
spots_train_x, spots_train_y = preparedata(spotmean_norm[:int(0.7 * len(spotmean_norm))], stepsback, stepsforward)
spots_test_x, spots_test_y = preparedata(spotmean_norm[int(0.7 * len(spotmean_norm)):], stepsback, stepsforward)

# LSTM network
model = Sequential()
model.add(LSTM(20, input_shape = spots_train_x.shape[1:]))
model.add(Dense(stepsforward))
model.compile(loss = 'mean_squared_error', optimizer = 'Nadam')
model.fit(spots_train_x, spots_train_y, epochs = epochs, batch_size = batch, verbose = 1)

scores = model.evaluate(spots_test_x, spots_test_y, verbose = 1, batch_size = batch) * maxval

print("\n\nFinal score = ", scores)

train_prediction = model.predict(spots_train_x) * maxval
test_prediction = model.predict(spots_test_x) * maxval

# Plotting to be done if stepsforward is set to 1

"""
months = np.arange(len(spotmean))
Data, = plt.plot(months, spotmean, 'r-', label = 'Data')
Train, = plt.plot(months[stepsback + 1 : int(0.7 * len(spotmean))], train_prediction, 'b-', label = 'Train prediction')
Test, = plt.plot(months[int(0.7 * len(spotmean)) + stepsback + 1:], test_prediction, 'g-', label = 'Test prediction')
plt.legend(handles = [Data, Train, Test])
#plt.xlim(3000, 4000)
plt.xlabel('Months')
plt.ylabel('Average sunspot number')
plt.savefig('Predictions2.png')
plt.clf()
"""


lastmonths = spotmean_norm[-stepsback:]
lastmonths = lastmonths.reshape(1, stepsback, 1)

future = model.predict(lastmonths)
future = (future * maxval).flatten()

nextpeak_month = np.argmax(future)
nextpeak_value = future[nextpeak_month]

print("Next peak month = {}\nNext peak value = {}".format(nextpeak_month, nextpeak_value))

# Plot of future data

plt.plot(np.arange(stepsforward), future)
plt.xlabel('Months from October 2016')
plt.ylabel('Average sunspot number')
plt.savefig('Future2.png')
plt.clf()

end_time = time.time()
elapsed = end_time - start_time

print("Total time elapsed: {} minutes".format(round(elapsed / 60, 2)))