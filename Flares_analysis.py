import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os.path
from datetime import date
from dateutil.relativedelta import *

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU
from keras.layers.wrappers import TimeDistributed

def nbr_to_date(nbr):
	''' 
	This function receives a number of the form yyMMddhhmm (year, month, day, hour, minute) and converts it into 
	a date object so that arithmetic can be done with it. The hour/minute part of the number is ignored.
	'''
	nbr = int(nbr)
	nstring = str(nbr).zfill(10)
	year = int(str('20') + nstring[:2])
	month = int(nstring[2:4])
	day = int(nstring[4:6])

	return date(year, month, day)

def date_to_relative(day, dzero):
	'''
	Returns the number of days elapsed since dzero.
	'''
	relativeday = (day - dzero).days

	return relativeday

def preparedata(data_flares, data_spots, stpback, stpfor):
	'''
	This function prepares the data in the format it needs to be in to be fed to the keras layers
	'''
	state = []
	objective = []

	for i in range(len(data_flares[:, 0]) - stpback - stpfor - 1):
		statepartial = []
		for j in range(stpback):
			statepartial.append(np.append(data_flares[j + i, :], data_spots[j + i]))
		state.append(np.asarray(statepartial))

		objpartial = []
		for j in range(stpfor):
			objpartial.append(data_flares[j + i + stpback, :])
		objective.append(np.ndarray.flatten(np.asarray(objpartial)))

	state = np.asarray(state)
	objective = np.asarray(objective)

	instances = state.shape[0]
	state = state.reshape(instances, stpback, 3)

	return state, objective

def NRMSE(arraypred, arraydata, nbmax, intmax):
	'''
	Calculates the Normalised Root Mean Square Error of predicted values of solar flare intensity
	and solar flare numbers
	'''
	Error_int = 0
	Error_nb = 0
	max_int = 0
	max_nb = 0
	min_int = 99999999999
	min_nb = 99999999999

	for i in range(arraypred.shape[0]):
		for j in range(int(arraypred.shape[1] / 2)):
			Error_nb += (arraypred[i, 2 * j] * nbmax - arraydata[i, 2 * j] * nbmax) ** 2
			Error_int += (arraypred[i, 2 * j + 1] * intmax - arraydata[i, 2 * j + 1] * intmax) ** 2

			if arraydata[i, 2 * j] < min_nb:
				min_nb = arraydata[i, 2 * j]

			if arraydata[i, 2 * j] > max_nb:
				max_nb = arraydata[i, 2 * j]

			if arraydata[i, 2 * j + 1] < min_int:
				min_int = arraydata[i, 2 * j + 1]

			if arraydata[i, 2 * j + 1] > max_int:
				max_int = arraydata[i, 2 * j + 1]

	Error_int = np.sqrt(Error_int / (len(arraypred) / 2))
	Error_nb = np.sqrt(Error_nb / (len(arraypred) / 2))

	NErr_int = 100 * Error_int / ((max_int - min_int) * intmax)
	NErr_nb = 100 * Error_nb / ((max_nb - min_nb) * nbmax)

	return NErr_nb, NErr_int

start_time = time.time()

np.random.seed(42)

batch = 10
epochs = 30
stepsforward = 1
stepsback = 50

'''
Since the data for sunspots and flares come from different sources, they come in different presentation formats.
This first part of the program only deals with adjusting the data so that the flares and sunspots run through the same interval 
and they are shaped appropriately to feed into the network.
Let us start with the flare data.
'''
print('\n\nPreparing data...\n\n')

flares = np.loadtxt('flares.txt', skiprows = 3, usecols = (0, 7)) # Loading the flares data
spots = np.loadtxt('sunspot.txt', skiprows = 0, usecols = (0, 1, 2, 4)) # Loading the sunspots data
datezero = nbr_to_date(flares[0, 0])
intensities = flares[:, 1]

flaredays = np.zeros(len(flares[:, 0]))

for i in range(len(flaredays)):
	flaredays[i] = date_to_relative(nbr_to_date(flares[i, 0]), datezero)


# This is the array we will use for counting the time in days from datezero
dayaxis = np.arange(flaredays[-1] + 1) 

# This array will contain the total number of flares and their average intensity on each day from datezero, even if there were no flares.
flarenumbers = np.zeros((len(dayaxis), 2))

for i in range(len(dayaxis)):
	totalflares = 0
	avgintensity = 0

	for j in range(len(flaredays)):
		if flaredays[j] == i:
			totalflares += 1
			avgintensity += intensities[j]

	if totalflares != 0:
		flarenumbers[i, 0] = totalflares
		flarenumbers[i, 1] = avgintensity / totalflares


# Now, we move onto the sunspot data.
for i in range(len(spots[:, 0])):
	n = date_to_relative(date(int(spots[i, 0]), int(spots[i, 1]),int(spots[i, 2])), datezero)
	if n == 0:
		argzero = i

spotnumbers = spots[argzero:, 3]
lastday = len(spotnumbers)

flarenumbers = flarenumbers[:lastday, :]
dayaxis = dayaxis[:lastday]

# Normalising the data
flare_nb_max = max(flarenumbers[:, 0])
flare_intens_max = max(flarenumbers[:, 1])
spot_max = max(spotnumbers)

flarenumbersNorm = np.zeros(flarenumbers.shape)
flarenumbersNorm[:, 0] = flarenumbers[:, 0] / flare_nb_max
flarenumbersNorm[:, 1] = flarenumbers[:, 1] / flare_intens_max
spotnumbersNorm = spotnumbers / spot_max

# Now, the last steps of data preparation are separating state and objective vectors and dividing the data between train and test

# I've chosen to use 70% of the data for training and 30% for testing
train_x, train_y = preparedata(data_flares = flarenumbersNorm[:int(0.7 * len(flarenumbersNorm[:, 0])), :],\
	data_spots = spotnumbersNorm[:int(0.7 * len(spotnumbersNorm))], stpback = stepsback, stpfor = stepsforward)

test_x, test_y = preparedata(data_flares = flarenumbersNorm[int(0.7 * len(flarenumbersNorm[:, 0])):, :],\
	data_spots = spotnumbersNorm[int(0.7 * len(spotnumbersNorm)):], stpback = stepsback, stpfor = stepsforward)

print('Ready! Training network...\n\n')

# LSTM network
model = Sequential()
model.add(LSTM(20, input_shape = train_x.shape[1:], return_sequences = True, dropout = 0.2))
model.add(GRU(6))
model.add(Dense(2 * stepsforward))
model.compile(loss = 'mean_squared_error', optimizer = 'Nadam')
model.fit(train_x, train_y, epochs = epochs, batch_size = batch, verbose = 1)

scores = model.evaluate(test_x, test_y, verbose = 1, batch_size = batch)

print("\n\nFinal score = ", scores)


# Now, let us make predictions
train_prediction = model.predict(train_x)
test_prediction = model.predict(test_x)

# Arrays we'll need for plotting
flarenb_predict_train = np.zeros(int(len(train_prediction) / 2))
flareint_predict_train = np.zeros(int(len(train_prediction) / 2))

flarenb_predict_train = np.round(train_prediction[:, 0] * flare_nb_max)
flareint_predict_train = train_prediction[:, 1] * flare_intens_max

flarenb_predict_test = np.zeros(int(len(test_prediction) / 2))
flareint_predict_test = np.zeros(int(len(test_prediction) / 2))

flarenb_predict_test = np.round(test_prediction[:, 0] * flare_nb_max)
flareint_predict_test = test_prediction[:, 1] * flare_intens_max

lengtrain = len(flarenb_predict_train)
lengtest = len(flarenb_predict_test)

# The following plots will only work if stepsforward is set to 1. If not, comment the plots out and the program will give the NRMSE results

# Plots for the solar flare numbers
TP, = plt.plot(np.arange(lengtrain), train_prediction[:, 0] * flare_nb_max, label = 'Train prediction')
TD, = plt.plot(np.arange(lengtrain), train_y[:, 0] * flare_nb_max, label = 'Train data')
TEP, = plt.plot(np.arange(lengtest) + lengtrain, test_prediction[:, 0] * flare_nb_max, label = 'Test prediction')
TED, = plt.plot(np.arange(lengtest) + lengtrain, test_y[:, 0] * flare_nb_max, label = 'Test data')
plt.xlabel('Days from 2nd November 2008')
plt.ylabel('Number of daily flares')
plt.legend(handles = [TP, TD, TEP, TED], loc = 1)
plt.savefig('FlareNumberCompare1.png')
plt.clf()

TP, = plt.plot(np.arange(lengtrain), train_prediction[:, 0] * flare_nb_max, label = 'Train prediction')
TD, = plt.plot(np.arange(lengtrain), train_y[:, 0] * flare_nb_max, label = 'Train data')
plt.xlim(1000, 1100)
plt.xlabel('Days from 2nd November 2008')
plt.ylabel('Number of daily flares')
plt.legend(handles = [TP, TD], loc = 1)
plt.savefig('FlareNumberCompare2.png')
plt.clf()

# Plots for the solar flare intensities
TP, = plt.plot(np.arange(lengtrain), train_prediction[:, 1] * flare_intens_max, label = 'Train prediction')
TD, = plt.plot(np.arange(lengtrain), train_y[:, 1] * flare_intens_max, label = 'Train data')
TEP, = plt.plot(np.arange(lengtest) + lengtrain, test_prediction[:, 1] * flare_intens_max, label = 'Test prediction')
TED, = plt.plot(np.arange(lengtest) + lengtrain, test_y[:, 1] * flare_intens_max, label = 'Test data')
plt.xlabel('Days from 2nd November 2008')
plt.ylabel('Average flare intensity (total photon counts)')
plt.legend(handles = [TP, TD, TEP, TED], loc = 1)
plt.savefig('FlareIntensityCompare1.png')
plt.clf()

TP, = plt.plot(np.arange(lengtrain), train_prediction[:, 0] * flare_intens_max, label = 'Train prediction')
TD, = plt.plot(np.arange(lengtrain), train_y[:, 0] * flare_intens_max, label = 'Train data')
plt.xlim(1000, 1100)
plt.xlabel('Days from 2nd November 2008')
plt.ylabel('Average flare intensity (total photon counts)')
plt.legend(handles = [TP, TD], loc = 1)
plt.savefig('FlareIntensityCompare2.png')
plt.clf()

NRMSE_nb_train, NRMSE_int_train = NRMSE(train_prediction, train_y, flare_nb_max, flare_intens_max)
NRMSE_nb_test, NRMSE_int_test = NRMSE(test_prediction, test_y, flare_nb_max, flare_intens_max)

print('NRMSE number train, test = {}%, {}%'.format(np.round(NRMSE_nb_train, 2), np.round(NRMSE_nb_test, 2)))
print('NRMSE intensity train, test = {}%, {}%'.format(np.round(NRMSE_int_train, 2), np.round(NRMSE_int_test, 2)))

end_time = time.time()
elapsed = end_time - start_time

print("\nTotal time elapsed: {} minutes".format(round(elapsed / 60, 2)))