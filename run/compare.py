from sklearn.decomposition import PCA, KernelPCA
from sklearn.mixture import GaussianMixture 
from sklearn.cluster import KMeans
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from tabulate import tabulate
from itertools import combinations
import numpy as np
import json
import sys

"""
Loads the learned images and their associated true model layer numbers
"""
def load_learned_images(model_names, image_types="layers"):
    learned_images = {}
    true_layer_indexes = {}
    for name in model_names:
        if image_types == "layers":
            imgs = json.load(open("visualizations/layers/{}.json".format(name), "r"))
            true_layer_indexes[name] = [x[0] for x in imgs]
            learned_images[name] = np.array([x[1] for x in imgs])
        else:
            learned_images[name] = {}
            imgs = json.load(open("visualizations/filters/{}.json".format(name), "r"))
            true_layer_indexes = [int(x) for x in list(imgs.keys())]
            for layer, filters in imgs.items():
                learned_images[name][int(layer)] = np.array(filters)
    return (learned_images, true_layer_indexes)


"""
Graysacles the given image(s)
"""
def greyscale(images):
    return np.dot(images[...,:3], [0.2989, 0.5870, 0.1140])


"""
Use traditional metrics to measure similarity
"""
def comp_metrics(model_names):
    learned_images, true_layer_indexes = load_learned_images(model_names)
    #Layer indices compared for every combination of models
    layer_indices = [x for x in range(-3, 3)]
    #Preset list for columns used by tabulate -- can add more for each metric used
    metrics = [['Layers', 'Models', 'SSIM', 'L2']]
    for i, name in enumerate(model_names):
        learned_images[name] = greyscale(learned_images[name])
        learned_images[name] = np.reshape(learned_images[name], (learned_images[name].shape[0], 9801))

    for i in layer_indices:
        for subset in combinations(model_names, 2):
            metrics.append([i if i < 0 else i + 1, subset, structural_similarity(learned_images[subset[0]][i], learned_images[subset[1]][i]), np.linalg.norm(learned_images[subset[0]][i] - learned_images[subset[1]][i])])

    with open("metrics.txt", "w") as f:
        f.write(tabulate(metrics, headers='firstrow', tablefmt="fancy_grid"))

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
def build_gmm_fit_predict(n_components, n_init=10, max_iter=300):
    gmm = GaussianMixture(n_components=n_components, n_init=n_init, max_iter=max_iter)
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
        for q in range(learned_images[name].shape[0]):
            #Alteration where we change on specific color channels
            learned_images[name][q][:,:,0] -= np.amin(learned_images[name][q][:,:,0])
            learned_images[name][q][:,:,0] /= np.amax(learned_images[name][q][:,:,0])
            learned_images[name][q][:,:,1] -= np.amin(learned_images[name][q][:,:,1])
            learned_images[name][q][:,:,1] /= np.amax(learned_images[name][q][:,:,1])
            learned_images[name][q][:,:,2] -= np.amin(learned_images[name][q][:,:,2])
            learned_images[name][q][:,:,2] /= np.amax(learned_images[name][q][:,:,2])
        learned_images[name] = greyscale(learned_images[name])
        learned_images[name] = np.reshape(learned_images[name], (learned_images[name].shape[0], 9801))

    #Perform linear and non-linear decomposition
    decomp_trainable = np.concatenate([x for x in learned_images.values()], axis=0)
    #Perform PCA
    pca = PCA(n_components=200)
    pca_reduced_images = pca.fit_transform(decomp_trainable)
    #Perform KPCA
    kpca = KernelPCA(n_components=200, kernel="rbf")
    kpca_reduced_images = kpca.fit_transform(decomp_trainable)

    #Clustering
    #KMeans
    models_kmeans_clusters = cluster(kpca_reduced_images, model_names, learned_images, KMeans(n_clusters=13, n_init=2000, max_iter=700).fit_predict)
    #Gaussion Mixture Models
    models_gmms_clusters = cluster(kpca_reduced_images, model_names, learned_images,  build_gmm_fit_predict(n_components=13, n_init=2000, max_iter=700))
    print(models_kmeans_clusters)
    print()
    print(models_gmms_clusters)
    print()
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


"""
Filters cluster analysis
"""
def cluster_filters_analysis(model_names, compares):
    learned_images, true_layer_indexes = load_learned_images(model_names, "filters")
    #Preprocess the learned images
    for name, layers in learned_images.items():
        for layer, filters in layers.items():
            new_filters = np.zeros((filters.shape[0], 99, 99))
            for i in range(len(filters)):
                #filters[i] -= np.amin(filters[i])
                #filters[i] /= np.amax(filters[i])
                new_filters[i] = greyscale(filters[i])
            learned_images[name][layer] = new_filters
    for compare in compares:
        filters1 = learned_images[compare[0][0]][compare[0][1]]
        filters2 = learned_images[compare[1][0]][compare[1][1]]
        matches = []
        for i in range(len(filters1)):
            best_match = (0,-100)
            for q in range(len(filters2)):
                similarity = structural_similarity(filters1[i], filters2[q])
                if similarity > best_match[1]:
                    best_match = (q, similarity)
            matches.append((i, *best_match))
        print(compare)
        print(matches)
        print()
        #Find the nearest neighber by copmuting the ssim difference
    """
    true_layer_index is the layer index the filters are for, refer to layers_filters in __main__ for more information
    learned_images = {
        vgg16: {
            true_layer_index: [
                learned image for filter 0 for true_layer_index, 
                learned image for filter 1 for true_layer_index,
                learned image for filter 2 for true_layer_index, 
                etc...
            ],
            true_layer_index: [
                learned image for filter 0 for true_layer_index, 
                learned image for filter 1 for true_layer_index,
                learned image for filter 2 for true_layer_index, 
                etc...
            ],
            etc...
        }
        vgg19: {
            true_layer_index: [
                learned image for filter 0 for true_layer_index, 
                learned image for filter 1 for true_layer_index,
                learned image for filter 2 for true_layer_index, 
                etc...
            ],
            true_layer_index: [
                learned image for filter 0 for true_layer_index, 
                learned image for filter 1 for true_layer_index,
                learned image for filter 2 for true_layer_index, 
                etc...
            ],
            etc..
        }
        etc...
    }
    """
