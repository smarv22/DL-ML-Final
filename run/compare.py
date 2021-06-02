from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture 
from sklearn.cluster import KMeans
import numpy as np
import json
import sys


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
Clustering
"""
def cluster(fit_data, model_names, learned_images, cluster_function):
    clustered_images = cluster_function(fit_data)
    models_clusters = {}
    for i, name in enumerate(model_names):
        if i == 0:
            models_clusters[name] = clustered_images[:learned_images[name].shape[0]]
        else:
            models_clusters[name] = clustered_images[learned_images[model_names[i-1]].shape[0]:learned_images[model_names[i-1]].shape[0] + learned_images[name].shape[0]]
    return models_clusters


"""
Gmm fit predict callback
Only required becuase sklearn does not provide a fit_predict method on GMM class
"""
def build_gmm_fit_predict(n_components):
    gmm = GaussianMixture(n_components=n_components)
    def gmm_fit_predict(fit_data):
        gmm2 = gmm.fit(fit_data)
        return gmm2.predict(fit_data)
    return gmm_fit_predict



"""
Performs cluserting analysis
"""
def cluster_analysis(model_names):
    learned_images, true_layer_indexes = load_learned_images(model_names)
    for i, name in enumerate(model_names):
        learned_images[name] = greyscale(learned_images[name])
        learned_images[name] = np.reshape(learned_images[name], (learned_images[name].shape[0], 9801))
        #May want to remove, check if it actually is better
        for q in range(learned_images[name].shape[0]):
            learned_images[name][q] = learned_images[name][q] - np.amin(learned_images[name][q])
            learned_images[name][q] = learned_images[name][q] - np.amax(learned_images[name][q])

    #Perform PCA
    pca = PCA(n_components=50)
    pca_trainable = np.concatenate([x for x in learned_images.values()], axis=0)
    reduced_images = pca.fit_transform(pca_trainable)

    #Clustering
    #KMeans
    models_kmeans_clusters = cluster(reduced_images, model_names, learned_images, KMeans(n_clusters=50).fit_predict)
    #Gaussion Mixture Models
    models_gmms_clusters = cluster(reduced_images, model_names, learned_images,  build_gmm_fit_predict(n_components=50))
    """
    both models_kmeans_clusters and models_gmms_clusters = {
        vgg16: [
            kmeans/gmms cluster for convolutional layer 1,
            kmeans/gmms cluster for convolutional  layer 2,
            kmeans/gmms cluster for convolutional  layer 3,
            etc....
        ],
        vgg19: [
            kmeans/gmms cluster for convolutional layer 1,
            kmeans/gmms cluster for convolutional  layer 2,
            kmeans/gmms cluster for convolutional  layer 3,
            etc....
        ],
        etc...
    }
    """
