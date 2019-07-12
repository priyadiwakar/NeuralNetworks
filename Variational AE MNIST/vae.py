from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import binary_crossentropy
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import os

def sample_z(args):
    z_mean,z_log_var=args
    batch = K.shape(z_mean)[0]
    dim =  K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


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

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(1,figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()
    if latent_dim==2:
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
                z_sample = np.array([[xi, yi]])
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


np.random.seed(237)

# MNIST dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

# network parameters
image_size = X_train.shape[1]
original_dim = image_size
input_shape = (original_dim, )
intermediate_dim = 512
batch_size = 128
latent_dim = 2
epochs = 50
model_name="vae_mlp_"+str(latent_dim)
model_weights=model_name+".h5"


inputs = Input(shape=input_shape, name='encoder_input')
x=Dense(intermediate_dim,activation='relu')(inputs)
z_mean=Dense(latent_dim,name='z_mean')(x)
z_log_var=Dense(latent_dim,name='z_log_var')(x)


z=Lambda(sample_z,output_shape=(latent_dim,),name='z')([z_mean,z_log_var])


encoder=Model(inputs,[z_mean,z_log_var,z],name='encoder')
encoder.summary()

latent_inputs=Input(shape=(latent_dim,), name='z_sampling')
x=Dense(intermediate_dim,activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()


outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name=model_name)



if __name__ == '__main__':
    
    models = (encoder, decoder)
    data = (X_test, Y_test)

    
    reconstruction_loss = binary_crossentropy(inputs,
                                                  outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    
    
    
    vae.fit(X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, None))
    vae.save_weights(model_weights)
    
    plot_results(models,
                 data,
                 batch_size=batch_size,
                 model_name=model_name)
    
    
    encoder = Model(inputs, z_mean)
    encoded_imgs = encoder.predict(X_test)
    decoded_imgs = decoder.predict(encoded_imgs)
    display_reconstructed(X_test, decoded_imgs,model_name)