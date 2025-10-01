import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import losses
from keras.datasets import mnist
from keras.layers import Dense, Input, Lambda
from keras.models import Model

BATCH_SIZE: int = 100
ORIGINAL_DIM: int = 28 * 28
LATENT_DIM: int = 2
INTERMEDIATE_DIM: int = 256
NUMBER_EPOCH: int = 5
EPSILON_STD: float = 1.0


# --- Creating the encoder ---
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(
        shape=(batch, dim), mean=0.0, stddev=1.0
    )  # Changed from K.random_normal
    return z_mean + tf.math.exp(z_log_var / 2) * epsilon


x = Input(shape=(ORIGINAL_DIM,), name="input")  # Input to our encoder
h = Dense(INTERMEDIATE_DIM, activation="relu", name="encoding")(x)  # Intermediate Layer
z_mean = Dense(LATENT_DIM, name="mean")(h)  # Defines the mean of the latent space
z_log_var = Dense(LATENT_DIM, name="log-variance")(
    h
)  # Defines the log variance of the latent space
z = Lambda(sampling, output_shape=(LATENT_DIM,))([z_mean, z_log_var])
encoder = Model(
    x, [z_mean, z_log_var, z], name="encoder"
)  # Defines the econder as a Keras model


# --- Creating the decoder ---
input_decoder = Input(shape=(LATENT_DIM,), name="decoder_input")  # Input to the decoder
decoder_h = Dense(INTERMEDIATE_DIM, activation="relu", name="decoder_h")(
    input_decoder
)  # Takes the latente space to the intermediate dimension
x_decoded = Dense(ORIGINAL_DIM, activation="sigmoid", name="flat_decoded")(
    decoder_h
)  # Gets the mean from the original dimension
decoder = Model(
    input_decoder, x_decoded, name="decoder"
)  # Defines the decoder as Keras model

# --- Combining the model ---
output_combined = decoder(
    encoder(x)[2]
)  # Grabs the output. Recall that we need to grab the third element our sampling z.
vae = Model(x, output_combined)  # Link the input and the overall output
print(vae.summary())


# --- Defining our loss function ---
def vae_loss(y_true, y_pred):
    # Get encoder outputs for the current batch
    z_mean, z_log_var, _ = encoder(y_true)

    # Reconstruction loss
    xent_loss = ORIGINAL_DIM * losses.binary_crossentropy(y_true, y_pred)

    # KL divergence loss
    kl_loss = -0.5 * tf.math.reduce_sum(
        1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var), axis=-1
    )

    return tf.reduce_mean(xent_loss + kl_loss)


# --- Compile the model ---
vae.compile(optimizer="rmsprop", loss=vae_loss)


# --- Creating train/test split ---
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


# --- Train model ---
history = vae.fit(
    x_train,
    x_train,
    shuffle=True,
    epochs=NUMBER_EPOCH,
    batch_size=BATCH_SIZE,
    validation_data=(x_test, x_test),
    verbose=1,
)


# --- Plot and save training history ---
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Generate some sample reconstructions
plt.subplot(1, 2, 2)
n = 10
digit_size = 28
figure = np.zeros((digit_size * 2, digit_size * n))
for i in range(n):
    x_test_sample = x_test[i : i + 1]
    x_decoded = vae.predict(x_test_sample)

    digit = x_test_sample[0].reshape(digit_size, digit_size)
    figure[:digit_size, i * digit_size : (i + 1) * digit_size] = digit

    digit = x_decoded[0].reshape(digit_size, digit_size)
    figure[digit_size:, i * digit_size : (i + 1) * digit_size] = digit

plt.imshow(figure, cmap="Greys_r")
plt.title("Original (top) vs Reconstructed (bottom)")
plt.axis("off")

plt.tight_layout()
plt.savefig("vae_results.png", dpi=150, bbox_inches="tight")

print("Plot saved as 'vae_results.png'")
