import sklearn
import numpy as np
import json


"""
Loads the learned images and their associated true model layer numbers
"""
def load_learned_images(model_names):
    learned_images = {}
    true_layer_indexes = {}
    for name in model_names:
       imgs = json.load(open("visualizations/{}.json".format(name), "r"))
       true_layer_indexes[name] = [x[0] for x in imgs]
       learned_images[name] = np.array([x[1] for x in imgs])
    return (learned_images, true_layer_indexes)


"""
Graysacles the given image(s)
"""
def greyscale(images):
    return np.dot(images[...,:3], [0.2989, 0.5870, 0.1140])


"""
Performs cluserting analysis
"""
def cluster_analysis(model_names):
    learned_images, true_layer_indexes = load_learned_images(model_names)
    for name in model_names:
        print(learned_images[name].shape)
        print(greyscale(learned_images[name]).shape)
