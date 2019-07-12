import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from keras.datasets import mnist
from scipy.stats import norm
import os

import keras
from keras import layers
from keras.models import Model
from keras import metrics
from keras import backend as K   # 'generic' backend so code works with either tensorflow or theano

# construct a custom layer to calculate the loss
class CustomVariationalLayer(keras.layers.Layer):

    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        # Reconstruction loss
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        # KL divergence
        kl_loss = -5e-4 * K.mean(1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma), axis=-1)
        return K.mean(xent_loss + kl_loss)

    # adds the custom loss to the class
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x
    
    

def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector
    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """
    global latent_dim
    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean= encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(1,figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()  

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            if latent_dim == 2:
               z_sample = np.array([[xi, yi]])
            elif latent_dim == 10:
               z_sample = np.array([[xi, yi, 1, 1, 1, 1, 1, 1, 1, 1]])
            elif latent_dim == 20:
               z_sample = np.array([[xi, yi, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(2,figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


def display_reconstructed(x_test, decoded_imgs,model_name):

    filename="recontructed_imgs"+str(latent_dim)+".png"
    filename = os.path.join(model_name, filename)
    plt.figure(3)
    for i in range(100):
        
        
        # display reconstruction
        ax = plt.subplot(10,10,i+1)
        plt.imshow(decoded_imgs[i].reshape(28, 28),cmap='gray')
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
    plt.savefig(filename)
    plt.show()



# sampling function
def sampling(args):
    z_mu, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mu)[0], latent_dim),
                              mean=0., stddev=1.)
    return z_mu + K.exp(z_log_sigma) * epsilon


K.clear_session()

np.random.seed(237)

# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = np.reshape(x_train, [-1,28,28,1])
x_test = np.reshape(x_test, [-1,28,28,1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train=np.append(x_train,x_test[0:5000,:,:,:],axis=0)
x_test=x_test[5000:,:,:,:]

y_train=np.append(y_train,y_test[0:5000],axis=0)
y_test=y_test[5000:]


img_shape = (28, 28, 1)    # for MNIST
batch_size = 256
latent_dim = 20  # Number of latent dimension parameters
model_name="vae_mlp_"+str(latent_dim)
model_weights=model_name+".h5"

# Encoder architecture: Input -> Conv2D*4 -> Flatten -> Dense
input_img = keras.Input(shape=img_shape)

x = layers.Conv2D(32, 3,
                  padding='same', 
                  activation='relu')(input_img)
x = layers.Conv2D(64, 3,
                  padding='same', 
                  activation='relu',
                  strides=(2, 2))(x)
x = layers.Conv2D(64, 3,
                  padding='same', 
                  activation='relu')(x)
x = layers.Conv2D(64, 3,
                  padding='same', 
                  activation='relu')(x)
# need to know the shape of the network here for the decoder
shape_before_flattening = K.int_shape(x)

x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)

# Two outputs, latent mean and (log)variance
z_mu = layers.Dense(latent_dim)(x)
z_log_sigma = layers.Dense(latent_dim)(x)

z = layers.Lambda(sampling)([z_mu, z_log_sigma])



# decoder takes the latent distribution sample as input
decoder_input = layers.Input(K.int_shape(z)[1:])

# Expand to 784 total pixels
x = layers.Dense(np.prod(shape_before_flattening[1:]),
                 activation='relu')(decoder_input)

# reshape
x = layers.Reshape(shape_before_flattening[1:])(x)

# use Conv2DTranspose to reverse the conv layers from the encoder
x = layers.Conv2DTranspose(32, 3,
                           padding='same', 
                           activation='relu',
                           strides=(2, 2))(x)

x = layers.Conv2D(1, 3,
                  padding='same', 
                  activation='sigmoid')(x)

# decoder model statement
decoder = Model(decoder_input, x)

# apply the decoder to the sample from the latent distribution
z_decoded = decoder(z)

# apply the custom loss to the input images and the decoded latent distribution sample
y = CustomVariationalLayer()([input_img, z_decoded])

# VAE model statement
vae = Model(input_img, y,name=model_name)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()

history = vae.fit(x=x_train, y=None,
        shuffle=    True,
        epochs=15,
        batch_size=batch_size,
        validation_data=(x_test, None))
vae.save_weights(model_weights)
encoder = Model(input_img, z_mu)
models = (encoder, decoder)
data = (x_test, y_test)

plot_results(models,
                 data,
                 batch_size=batch_size,
                 model_name=model_name)

#encoder = Model(inputs, z_mean)
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)
display_reconstructed(x_test, decoded_imgs,model_name)
print(history.history.keys())

#  "Loss" graphs
filename = os.path.join(model_name, "vae_training_loss.png")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model training loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(filename)
plt.show()
