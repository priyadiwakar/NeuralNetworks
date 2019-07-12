
import os
import keras
import numpy as np 
import struct
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pylab as plt
import csv

random.seed(0)
batch_size = 128
num_classes = 10
epochs = 10
img_rows, img_cols = 28, 28
learn_rate=0.001
activation='relu'
train_acc=[]
val_acc=[]
train_loss=[]
val_loss=[]
finaltacc=[]
finaltloss=[]
finalvacc=[]
finalvloss=[]



def read(dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)

    
#load MNIST database from files train-images.idx3-ubyte,train-labels.idx1-ubyte
#t10k-images.idx3-ubyte,t10k-labels.idx1-ubyte 

#train data read into list
trainingdata=list(read(dataset = "training", path = "."))
#test data read into list
testingdata=list(read(dataset = "testing", path = "."))

#initialize array to store image pixels and label data
N=len(trainingdata)
x_train=np.zeros((N,28,28))
y_train=np.zeros(N)

N1=len(testingdata)
X_test=np.zeros((N1,28,28))
Y_test=np.zeros(N1)

#read list data into the arrays
for i in range(N):
    
    label,pixels=trainingdata[i]
    #a=np.reshape(pixels,(1,784))
    x_train[i,:]=pixels
    y_train[i]=label

for i in range(N1):
    
    label1,pixels1=testingdata[i]
    #a=np.reshape(pixels1,(1,784))
    X_test[i,:]=pixels1
    Y_test[i]=label1


x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
X_test = X_test.astype('float32')
x_train /= 255
X_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


#used during hyperparameter tuning
X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train,
                                                    stratify=y_train, 
                                                    test_size=0.33,random_state=42)



# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below
Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_val = keras.utils.to_categorical(Y_val, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)
y_train = keras.utils.to_categorical(y_train, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),activation=activation,input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation=activation))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation=activation))
model.add(Dense(num_classes, activation='softmax'))
#myfile=open('resultf.csv','a', newline='')

#for i in range(len(learn_rate)):
    #for j in range(len(batch_size)):
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(lr=learn_rate),
              metrics=['accuracy'])


history=model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)
#    f= open('outputf.txt', 'a')
#    f.write(str(history.history) )
#    f.close()
#
#    tacc=history.history['acc']
#    vacc=history.history['val_acc']
#    tloss=history.history['loss']
#    vloss=history.history['val_loss']
#
#    wr = csv.writer(myfile)
#    wr.writerow(tacc)
#    wr.writerow(vacc)
#    wr.writerow(tloss)
#    wr.writerow(vloss)
#    train_acc.append(tacc)
#    val_acc.append(vacc)
#    train_loss.append(tloss)
#    val_loss.append(vloss)
#    finaltacc.append(tacc[-1])
#    finaltloss.append(tloss[-1])
#    finalvacc.append(vacc[-1])
#    finalvloss.append(vloss[-1])

#wr.writerow(finaltacc)
#wr.writerow(finalvacc)
#wr.writerow(finaltloss)
#wr.writerow(finalvloss)
#
#myfile.close()

# summarize history for accuracy
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#title='Model accuracy for activation function '+activation
#plt.title(title)
#plt.ylabel('Accuracy')
#plt.xlabel('Epochs')
#plt.legend(['training set', 'validation set'], loc='upper left')
#plt.savefig(title)
#plt.show()
## summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#title='Model Loss for activation function '+activation
#plt.title(title)
#plt.ylabel('Loss')
#plt.xlabel('Epochs')
#plt.legend(['training set', 'validation set'], loc='upper left')
#plt.savefig(title)
#plt.show()
#
#
##
score = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
y_pred = model.predict_classes(X_test)


onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = y_pred.reshape(len(y_pred), 1)
pred = onehot_encoder.fit_transform(integer_encoded)
pred=pred.astype(int)
#store result in CSV file
with open('mnist.csv', 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(pred)

csvFile.close()

