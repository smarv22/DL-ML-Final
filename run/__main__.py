from keras.applications import InceptionResNetV2, InceptionV3
import tensorflow as tf
import sys

import visualize


if __name__ == "__main__":
    pre_model = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(299,299,3))
    #pre_model = InceptionV3(weights="imagenet")

    """
    #Useful for finding the layer index you want to visualize
    for layer in pre_model.layers[:20]:
        print(layer)
    sys.exit(0)
    """

    #Create sub model
    outputs = pre_model.layers[300].output
    sub_model = tf.keras.models.Model(pre_model.layers[0].input, outputs)

    start_image = tf.zeros((1, 299,299,3))

    #fr is the index of the filter being visualized, or "all" to visualize the total layer activation
    img = visualize.gradient_ascent_loop(sub_model, start_image, 100, 0.001, None, fr=0)
    visualize.display_learned_image(img[0])
