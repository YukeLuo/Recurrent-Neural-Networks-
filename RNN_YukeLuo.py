# -*- coding: utf-8 -*-
"""
Some code adapted from:  https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
@author: apblossom
Small LSTM Network to learn text sequences
"""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import CuDNNLSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import datetime
import matplotlib.pyplot as plt
from random import sample
import random
import keras
import string
import sys

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Parameters Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
seq_length = 20
epoch_num =50
batch_size_num=5000

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

filename = "abcnews-date-text.csv"
df = pd.read_csv(filename, 
            header=0, 
            names=['publish_date', 'headline_text'])
#random.seed(1)
df=df.sample(n=100000,replace=False,random_state=1)

df=df.drop(columns=['publish_date'])
df.to_csv('cleanedup-news-file.csv',
          index=False,header=False)


filename="cleanedup-news-file.csv"
raw_text = open(filename, encoding="utf8").read()
raw_text = raw_text.lower()
dropPunctuation = str.maketrans("", "", string.punctuation)
raw_text = raw_text.translate(dropPunctuation)
start_time1 = datetime.datetime.now()


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Pretreat Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print ("Total Characters: ", n_chars)
print ("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers

dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])


n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

'''
del dataX
del dataY
del n_chars
del n_vocab
'''

'''
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Define Model 1 - CuDNN version Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
epoch_num =50
batch_size_num=5000
# define the LSTM model
model = Sequential()
model.add(CuDNNLSTM(32, input_shape=(X.shape[1], X.shape[2]),return_sequences=True))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(64))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.summary()
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Train Model 1 Section
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
model.compile(loss='categorical_crossentropy', optimizer='adam')
#filepath="C:/datasets/Model-1-CuDNN-weights-improvement-{epoch:02d}-{loss:.4f}-dropout.hdf5"
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-dropout.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, period = 10, mode='min')
callbacks_list = [checkpoint]
#model.load_weights(filename)
history1 = model.fit(X, y, epochs=epoch_num, batch_size=batch_size_num, callbacks=callbacks_list)
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print ("Seed:")
print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

# generate characters
for j in range(30):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print ("\nDone.")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Show output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
stop_time1 = datetime.datetime.now()

print("Model 1 Summary")
print("Batch Size:",batch_size_num,"\nNumber of Epochs:",epoch_num)
model.summary()
print("Last loss score:",history1.history['loss'][-1] )
print ("Time required for training:",stop_time1 - start_time1)

# summarize history for loss
plt.plot(history1.history['loss'])
plt.title('model 1 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc='upper right')
plt.show()


for i in range(10,41,10):
    history2 = model.fit(X, y, epochs=i, batch_size=batch_size_num)
    start = numpy.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    print ("Seed:")
    print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
    
    # generate characters
    for j in range(30):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print ("\nDone.")

'''




'''
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#MODEL 1
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Define Model 1 Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

epoch_num =50
batch_size_num=5000


# define the LSTM model
model = Sequential()
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2]),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.summary()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Train Model 1 Section
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
model.compile(loss='categorical_crossentropy', optimizer='adam')
filepath="C:/datasets/Model-1-weights-improvement-{epoch:02d}-{loss:.4f}-dropout.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=2, period = 5, mode='min')
callbacks_list = [checkpoint]

# fit the model
history1 = model.fit(X, y, epochs=epoch_num, batch_size=batch_size_num, callbacks=callbacks_list)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Show output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
stop_time1 = datetime.datetime.now()

print("Model 1 Summary")
print("Batch Size:",batch_size_num,"\nNumber of Epochs:",epoch_num)
model.summary()
print("Last loss score:",history1.history['loss'][-1] )
print ("Time required for training:",stop_time1 - start_time1)

# summarize history for loss
plt.plot(history1.history['loss'])
plt.title('model 1 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc='upper right')
plt.show()
'''
'''
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#MODEL 2 - Try Halfing the number of hidden units
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
epoch_num =50
batch_size_num=5000
# define the LSTM model
model = Sequential()
model.add(CuDNNLSTM(16, input_shape=(X.shape[1], X.shape[2]),return_sequences=True))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(32))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.summary()
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Train Model 1 Section
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
model.compile(loss='categorical_crossentropy', optimizer='adam')
#filepath="C:/datasets/Model-1-CuDNN-weights-improvement-{epoch:02d}-{loss:.4f}-dropout.hdf5"
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-dropout.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, period = 10, mode='min')
callbacks_list = [checkpoint]
#model.load_weights(filename)
history1 = model.fit(X, y, epochs=epoch_num, batch_size=batch_size_num, callbacks=callbacks_list)
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print ("Seed:")
print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

# generate characters
for j in range(30):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print ("\nDone.")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Show output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
stop_time1 = datetime.datetime.now()

print("Model 1 Summary")
print("Batch Size:",batch_size_num,"\nNumber of Epochs:",epoch_num)
model.summary()
print("Last loss score:",history1.history['loss'][-1] )
print ("Time required for training:",stop_time1 - start_time1)

# summarize history for loss
plt.plot(history1.history['loss'])
plt.title('model 1 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc='upper right')
plt.show()


for i in range(10,41,10):
    history2 = model.fit(X, y, epochs=i, batch_size=batch_size_num)
    start = numpy.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    print ("Seed:")
    print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
    
    # generate characters
    for j in range(30):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print ("\nDone.")
'''

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#MODEL 3 - Try Doubling hidden Units
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
epoch_num =50
batch_size_num=5000
# define the LSTM model
model = Sequential()
model.add(CuDNNLSTM(64, input_shape=(X.shape[1], X.shape[2]),return_sequences=True))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.summary()
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Train Model 1 Section
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
model.compile(loss='categorical_crossentropy', optimizer='adam')
#filepath="C:/datasets/Model-1-CuDNN-weights-improvement-{epoch:02d}-{loss:.4f}-dropout.hdf5"
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-dropout.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, period = 10, mode='min')
callbacks_list = [checkpoint]
#model.load_weights(filename)
history1 = model.fit(X, y, epochs=epoch_num, batch_size=batch_size_num, callbacks=callbacks_list)
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print ("Seed:")
print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

# generate characters
for j in range(30):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print ("\nDone.")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Show output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
stop_time1 = datetime.datetime.now()

print("Model 1 Summary")
print("Batch Size:",batch_size_num,"\nNumber of Epochs:",epoch_num)
model.summary()
print("Last loss score:",history1.history['loss'][-1] )
print ("Time required for training:",stop_time1 - start_time1)

# summarize history for loss
plt.plot(history1.history['loss'])
plt.title('model 1 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc='upper right')
plt.show()


for i in range(10,41,10):
    history2 = model.fit(X, y, epochs=i, batch_size=batch_size_num)
    start = numpy.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    print ("Seed:")
    print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
    
    # generate characters
    for j in range(30):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print ("\nDone.")

'''
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#MODEL 4 - add a hidden dense layer
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
epoch_num =50
batch_size_num=5000
# define the LSTM model
model = Sequential()
model.add(CuDNNLSTM(32, input_shape=(X.shape[1], X.shape[2]),return_sequences=True))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(64))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dense(y.shape[1], activation='softmax'))
model.summary()
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Train Model 1 Section
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
model.compile(loss='categorical_crossentropy', optimizer='adam')
#filepath="C:/datasets/Model-1-CuDNN-weights-improvement-{epoch:02d}-{loss:.4f}-dropout.hdf5"
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-dropout.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, period = 10, mode='min')
callbacks_list = [checkpoint]
#model.load_weights(filename)
history1 = model.fit(X, y, epochs=epoch_num, batch_size=batch_size_num, callbacks=callbacks_list)
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print ("Seed:")
print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

# generate characters
for j in range(30):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print ("\nDone.")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Show output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
stop_time1 = datetime.datetime.now()

print("Model 1 Summary")
print("Batch Size:",batch_size_num,"\nNumber of Epochs:",epoch_num)
model.summary()
print("Last loss score:",history1.history['loss'][-1] )
print ("Time required for training:",stop_time1 - start_time1)

# summarize history for loss
plt.plot(history1.history['loss'])
plt.title('model 1 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc='upper right')
plt.show()


for i in range(10,41,10):
    history2 = model.fit(X, y, epochs=i, batch_size=batch_size_num)
    start = numpy.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    print ("Seed:")
    print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
    
    # generate characters
    for j in range(30):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print ("\nDone.")
'''
'''
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#MODEL 5 - Doubling the Sequence Length vs Original Model
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
seq_length = 40
epoch_num =50
batch_size_num=5000
# define the LSTM model
model = Sequential()
model.add(CuDNNLSTM(32, input_shape=(X.shape[1], X.shape[2]),return_sequences=True))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(64))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.summary()
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Train Model 1 Section
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
model.compile(loss='categorical_crossentropy', optimizer='adam')
#filepath="C:/datasets/Model-1-CuDNN-weights-improvement-{epoch:02d}-{loss:.4f}-dropout.hdf5"
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-dropout.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, period = 10, mode='min')
callbacks_list = [checkpoint]
#model.load_weights(filename)
history1 = model.fit(X, y, epochs=epoch_num, batch_size=batch_size_num, callbacks=callbacks_list)
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print ("Seed:")
print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

# generate characters
for j in range(30):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print ("\nDone.")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Show output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
stop_time1 = datetime.datetime.now()

print("Model 1 Summary")
print("Batch Size:",batch_size_num,"\nNumber of Epochs:",epoch_num)
model.summary()
print("Last loss score:",history1.history['loss'][-1] )
print ("Time required for training:",stop_time1 - start_time1)

# summarize history for loss
plt.plot(history1.history['loss'])
plt.title('model 1 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc='upper right')
plt.show()


for i in range(10,41,10):
    history2 = model.fit(X, y, epochs=i, batch_size=batch_size_num)
    start = numpy.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    print ("Seed:")
    print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
    
    # generate characters
    for j in range(30):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print ("\nDone.")
'''
'''
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#MODEL 6 - Halfing the Sequence Length
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
seq_length = 10
epoch_num =50
batch_size_num=5000
# define the LSTM model
model = Sequential()
model.add(CuDNNLSTM(32, input_shape=(X.shape[1], X.shape[2]),return_sequences=True))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(64))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.summary()
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Train Model 1 Section
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
model.compile(loss='categorical_crossentropy', optimizer='adam')
#filepath="C:/datasets/Model-1-CuDNN-weights-improvement-{epoch:02d}-{loss:.4f}-dropout.hdf5"
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-dropout.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, period = 10, mode='min')
callbacks_list = [checkpoint]
#model.load_weights(filename)
history1 = model.fit(X, y, epochs=epoch_num, batch_size=batch_size_num, callbacks=callbacks_list)
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print ("Seed:")
print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

# generate characters
for j in range(30):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print ("\nDone.")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Show output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
stop_time1 = datetime.datetime.now()

print("Model 6 Summary")
print("Batch Size:",batch_size_num,"\nNumber of Epochs:",epoch_num)
model.summary()
print("Last loss score:",history1.history['loss'][-1] )
print ("Time required for training:",stop_time1 - start_time1)

# summarize history for loss
plt.plot(history1.history['loss'])
plt.title('model 1 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc='upper right')
plt.show()


for i in range(10,41,10):
    history2 = model.fit(X, y, epochs=i, batch_size=batch_size_num)
    start = numpy.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    print ("Seed:")
    print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
    
    # generate characters
    for j in range(30):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print ("\nDone.")
'''
'''
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Last Question
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
epoch_num =15
batch_size_num=5000
# define the LSTM model
model = Sequential()
model.add(CuDNNLSTM(64, input_shape=(X.shape[1], X.shape[2]),return_sequences=True))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.summary()
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Train Model 1 Section
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
model.compile(loss='categorical_crossentropy', optimizer='adam')
#filepath="C:/datasets/Model-1-CuDNN-weights-improvement-{epoch:02d}-{loss:.4f}-dropout.hdf5"
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-dropout.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, period = 1, mode='min')
callbacks_list = [checkpoint]
#model.load_weights(filename)
history1 = model.fit(X, y, epochs=epoch_num, batch_size=batch_size_num, callbacks=callbacks_list)
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print ("Seed:")
print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

# generate characters
for j in range(30):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print ("\nDone.")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Show output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
stop_time1 = datetime.datetime.now()

print("Model 1 Summary")
print("Batch Size:",batch_size_num,"\nNumber of Epochs:",epoch_num)
model.summary()
print("Last loss score:",history1.history['loss'][-1] )
print ("Time required for training:",stop_time1 - start_time1)

# summarize history for loss
plt.plot(history1.history['loss'])
plt.title('model 1 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc='upper right')
plt.show()


for i in range(1,16,1):
    history2 = model.fit(X, y, epochs=i, batch_size=batch_size_num)
    start = numpy.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    print ("Seed:")
    print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
    
    # generate characters
    for j in range(30):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print ("\nDone.")
'''