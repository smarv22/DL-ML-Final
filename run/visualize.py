import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


"""
Computes the loss
"""
def compute_loss(model, img, fr):
    features = model(img)
    loss = tf.zeros(shape=())
    if fr == "all":
        #Loops over all filters in the layer
        for i in range(features.shape[-1]):
            activation = features[:, :, :, i]
            #This scaling appears to be broken? I'm just ignoring it
            #scaling = tf.reduce_prod(tf.cast(tf.shape(activation), "float32"))
            #Indexing tries and removes padding, needs to be specified per model, or just ignored
            #loss += (tf.reduce_sum(tf.square(activation[:, 2:-2, 2:-2])) / scaling)
            #Used by tensorflows example (and literally everywhere else I have seen, we may need to evaluate which is best)
            loss += tf.math.reduce_mean(activation[:, 2:-2, 2:-2])
    else:
        #Loss for a specific filter
        activation = features[:, :, :, fr]
        loss = tf.math.reduce_mean(activation)
    return loss


"""
Performs a step of gradient ascent
"""
def gradient_ascent_step(model, img, learning_rate, fr):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(model, img, fr)
    grads = tape.gradient(loss, img)
    grads /= tf.maximum(tf.reduce_mean(tf.abs(grads)), 1e-6)
    #Used by tensorflows example
    #grads /= tf.math.reduce_std(grads) + 1e-8
    if fr == "all":
        img += learning_rate * grads
    else:
        #When just visualizing filters, I have seen people normalize grads = tf.math.norm(grads) / the l2 norm instead of what is below
        #When visualizing just the filter, the gradients can be really small, may want to do 10 * grads to speed up learning
        img += learning_rate * grads * 10
    return loss, img


"""
Performs gradient ascent for the number of iterations passed
"""
def gradient_ascent_loop(model, img, iterations, learning_rate, max_loss=None, fr="all"):
    for i in range(iterations):
        loss, img = gradient_ascent_step(model, img, learning_rate, fr)
        if max_loss is not None and loss > max_loss:
            break
        print("Loss value at step {}: {}".format(i, loss))
    return img


"""
Converts the tensor into an image to display
"""
def display_learned_image(img):
    img = img - tf.math.reduce_min(img)
    img = img / tf.math.reduce_max(img)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    """
    img = img.numpy() * 255
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.show()
    """
