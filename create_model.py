import tensorflow as tf
import numpy as np
import sys

import EER_mod
import read_mod

# provide the file with the speeches passed trough openl3 in .npy format
fl_inp = open(sys.argv[1],'r')


# load .npy files and create labels from the names of the files
# the number before the '-' sighn stands for the speaker, e.g.,
# in 19-xxxx-xxx.npy, 19 stands for the speaker's ID
n_S, n_speeches, emb, labels = read_mod.read_speeches(fl_inp)
n_data = len(labels)


# neurons in the input layer
n_in = emb.shape[1]
print('number of neurons in the input layer = ', emb.shape[1])


# shuffle data
rand_perm = np.arange(n_data)
np.random.shuffle( rand_perm )
labels = labels[rand_perm]
emb = emb[rand_perm]


# divide into training and test sets
# The training set is 10% from the whole set
n_test = int(n_data * 0.1)
print('size of the testing set = ', n_test)

labels_test = labels[:n_test]
data_test = emb[:n_test,:]

labels_train = labels[n_test:]
data_train = emb[n_test:,:]

print('size of the training set = ', labels_train.shape[0])


# Construct the model with only 1 hidden layer
n_hidden = 512
model = tf.keras.Sequential([
    tf.keras.layers.Dense(n_hidden, activation='relu'),
    tf.keras.layers.Dense(n_S)
])


# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# Train the model
epochs=20
model.fit(data_train, 
          labels_train, 
          epochs=epochs,
          verbose=0)


# Test the model
test_loss, test_acc = model.evaluate(data_test,  labels_test)
print('test_loss = ', test_loss)
print('test_loss = ', test_acc)


# Create the new model for features extracting 
# from the layer before the last one
new_model = tf.keras.Model(inputs=model.input,
                           outputs=model.layers[-2].output)


# save new model for usage in speakers comparison
model_name = 'model_S' + str(n_S) + '_speeches' + str(n_speeches)
model.save(model_name)


# Calculate EER on the test set
# all data points from the test set will be compared with each other
hidden_layer_pred = new_model.predict(data_test)


EER, thres = EER_mod.EER(hidden_layer_pred, labels_test)
print('EER = ', f'{EER*100: .1f}','%')
print('Threshold = ', f'{thres: .3f}')


# The name of the model, threshold, and list of speeches, 
# which were used for training are saved

fl_model = open(model_name+'.dat','w')
print('model_name = ', model_name, file = fl_model)
print('thres = ', thres, file = fl_model)


fl_inp.seek(0)
for speech in fl_inp.readlines():
    print('speech = ', speech.strip(), file = fl_model)
