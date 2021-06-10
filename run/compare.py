from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tabulate import tabulate
from itertools import combinations
import numpy as np
import json
import sys


def preprocess(model_names, learned_images, flatten=True):
    """
    for i, name in enumerate(model_names):
        learned_images[name] = greyscale(learned_images[name])
        learned_images[name] = np.reshape(learned_images[name], (learned_images[name].shape[0], 9801))
        for q in range(learned_images[name].shape[0]):
            learned_images[name][q] = learned_images[name][q] - np.amin(learned_images[name][q])
            learned_images[name][q] = learned_images[name][q] - np.amax(learned_images[name][q])
    """
    for i, name in enumerate(model_names):
        for q in range(learned_images[name].shape[0]):
            # Alteration where we change on specific color channels
            learned_images[name][q][:, :,
                                    0] -= np.amin(learned_images[name][q][:, :, 0])
            learned_images[name][q][:, :,
                                    0] /= np.amax(learned_images[name][q][:, :, 0])
            learned_images[name][q][:, :,
                                    1] -= np.amin(learned_images[name][q][:, :, 1])
            learned_images[name][q][:, :,
                                    1] /= np.amax(learned_images[name][q][:, :, 1])
            learned_images[name][q][:, :,
                                    2] -= np.amin(learned_images[name][q][:, :, 2])
            learned_images[name][q][:, :,
                                    2] /= np.amax(learned_images[name][q][:, :, 2])
        learned_images[name] = greyscale(learned_images[name])
        if flatten:
            learned_images[name] = np.reshape(
                learned_images[name], (learned_images[name].shape[0], 9801))
    return learned_images


"""
Loads the learned images and their associated true model layer numbers
"""


def load_learned_images(model_names, image_types="layers"):
    learned_images = {}
    true_layer_indexes = {}
    for name in model_names:
        if image_types == "layers":
            imgs = json.load(
                open("visualizations/layers/{}.json".format(name), "r"))
            true_layer_indexes[name] = [x[0] for x in imgs]
            learned_images[name] = np.array([x[1] for x in imgs])
        else:
            learned_images[name] = {}
            imgs = json.load(
                open("visualizations/filters/{}.json".format(name), "r"))
            true_layer_indexes = [int(x) for x in list(imgs.keys())]
            for layer, filters in imgs.items():
                learned_images[name][int(layer)] = np.array(filters)
    return (learned_images, true_layer_indexes)


"""
Graysacles the given image(s)
"""


def greyscale(images):
    return np.dot(images[..., :3], [0.2989, 0.5870, 0.1140])


"""
Use traditional metrics to measure similarity
"""


def comp_metrics(model_names):
    learned_images, true_layer_indexes = load_learned_images(model_names)
    learned_images = preprocess(model_names, learned_images, flatten=False)
    # Layer indices compared for every combination of models
    layer_indices = [x for x in range(-3, 3)]
    # Preset list for columns used by tabulate -- can add more for each metric used
    metrics = [['Layers', 'Models', 'SSIM', 'L2']]

    for i in layer_indices:
        for subset in combinations(model_names, 2):
            metrics.append([i if i < 0 else i + 1, subset, structural_similarity(learned_images[subset[0]][i],
                                                                                 learned_images[subset[1]][i]), np.linalg.norm(learned_images[subset[0]][i] - learned_images[subset[1]][i])])

    with open("metrics.txt", "w") as f:
        f.write(tabulate(metrics, headers='firstrow', tablefmt="fancy_grid"))


"""
Clustering
"""


def cluster(fit_data, model_names, learned_images, cluster_function):
    clustered_images = cluster_function(fit_data)
    models_clusters = {}
    size = []
    for i, name in enumerate(model_names):
        if i == 0:
            models_clusters[name] = clustered_images[:learned_images[name].shape[0]]
            size.append(learned_images[name].shape[0])
            left = size[i]
        else:
            size.append(learned_images[name].shape[0])
            right = left + size[i]
            models_clusters[name] = clustered_images[left:right]
            left = right

    return models_clusters, size


"""
Gmm fit predict callback
Only required becuase sklearn does not provide a fit_predict method on GMM class
"""


def build_gmm_fit_predict(n_components, n_init=10, max_iter=300):
    gmm = GaussianMixture(n_components=n_components,
                          n_init=n_init, max_iter=max_iter)

    def gmm_fit_predict(fit_data):
        gmm2 = gmm.fit(fit_data)
        return gmm2.predict(fit_data)
    return gmm_fit_predict


"""
Generate scatter plot for PCA

"""


def visualize_data(X, fig_title='plot.jpg'):
    fig_str = fig_title + '.jpg'
    plt.figure()
    plt.scatter(X[:, 0],
                X[:, 1], alpha=0.8)
    plt.title(fig_title)
    plt.savefig(fig_str)
    plt.close()


"""
Performs cluserting analysis
"""


def cluster_analysis(model_names):
    learned_images, true_layer_indexes = load_learned_images(model_names)
    learned_images = preprocess(model_names, learned_images)

    # Perform linear and non-linear decomposition
    decomp_trainable = np.concatenate(
        [x for x in learned_images.values()], axis=0)

    # Perform PCA
    pca = PCA(n_components=100)
    pca_reduced_images = pca.fit_transform(decomp_trainable)
    # Perform KPCA
    kpca = KernelPCA(n_components=100, kernel="poly")
    kpca_reduced_images = kpca.fit_transform(decomp_trainable)

    # Generate scatter plots for data, PCA, and Kernel PCA
    visualize_data(decomp_trainable, fig_title='Initial Data Plot')
    visualize_data(pca_reduced_images, fig_title='PCA_plot')
    visualize_data(kpca_reduced_images, fig_title='Kernel PCA_plot')
    print('Generated dataset plots')

    # Clustering
    # KMeans
    k = 13
    model = KMeans(k, n_init=2000, max_iter=700)
    # model2 = KMeans(k, n_init=2000, max_iter=700)
    models_kmeans_clusters, vals = cluster(
        kpca_reduced_images, model_names, learned_images, model.fit_predict)
    # models2_kmeans_clusters, vals2 = cluster(
    #     decomp_trainable, model_names, learned_images, model2.fit_predict)

    # Gaussion Mixture Models
    # models_gmms_clusters = cluster(kpca_reduced_images, model_names, learned_images,  build_gmm_fit_predict(
    #     n_components=15, n_init=100, max_iter=700))
    print(models_kmeans_clusters)
    # print(models_gmms_clusters)

    centroids = model.cluster_centers_
    # centroids2 = model2.cluster_centers_  # Centroids for pre-kernel PCA

    # Generate tables for layer matches between models
    find_layer_matches(models_kmeans_clusters)
    find_layer_matches(models_kmeans_clusters,
                       model_str1='mobilenet', model_str2='mobilenetv2')
    print('Generated layer tables')

    # Generate Horizontal Bar figures of layer to cluster distribution between models
    generate_distribution_hbar(k, model_names, models_kmeans_clusters)
    generate_distribution_hbar(k, model_names, models_kmeans_clusters,
                               model_str1='mobilenet', model_str2='mobilenetv2')
    print('Generated layer distributions')

    # Generate kmeans scatter plot figures for pre-Kernel PCA data
    # Maybe we can use these figures to explain why Kernel PCA was used
    # generate_kmeans_figures(k, model_names, decomp_trainable,
    #                         models2_kmeans_clusters, centroids2, vals2, str_add='_no kpca')

    # Generate kmeans scatter plot figures
    generate_kmeans_figures(k, model_names, kpca_reduced_images,
                            models_kmeans_clusters, centroids, vals)

    print('Generated Kmeans figures')


"""
Use Clusters to find matching layers
"""


def find_layer_matches(models_kmeans_clusters, model_str1='vgg16', model_str2='vgg19'):
    first_model = models_kmeans_clusters[model_str1]
    second_model = models_kmeans_clusters[model_str2]
    top_str1 = model_str1 + '_Layer'
    top_str2 = model_str2 + '_Matches'
    matches = [[top_str1, top_str2, 'Cluster']]
    for i, layer in enumerate(first_model):
        val = np.where(layer == second_model)
        val = list(val[0])
        matches.append([i, val, layer])

    save_str = model_str1 + '_' + model_str2 + '_layer_matches.txt'
    with open(save_str, "w", encoding='utf-8') as f:
        f.write(tabulate(matches, headers='firstrow', tablefmt="fancy_grid"))


"""
Plot kmeans for each model
"""


def generate_kmeans_figures(k, model_names, images, models_kmeans_clusters, centroids, layer_count, str_add=''):
    # Generate Full data plot with centroids
    plt.figure()
    plt.scatter(images[:, 0],
                images[:, 1], alpha=0.3)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                c='black', alpha=0.8, marker='*')
    fig_title = 'Centroids Plot' + str_add
    fig_str = fig_title + '.jpg'
    plt.title(fig_title)
    plt.savefig(fig_str)
    plt.close()

    # Generate figures for each model's kmeans plot
    for i, name in enumerate(model_names):
        fig_title = str(k) + 'k'+'_kmeans_' + name + str_add
        fig_str = fig_title + '.jpg'

        if i == 0:
            plt.figure()
            plt.scatter(images[:layer_count[i], 0],
                        images[:layer_count[i], 1], c=models_kmeans_clusters[name], cmap='rainbow')
            plt.scatter(centroids[:, 0], centroids[:, 1],
                        c='black', alpha=0.8, s=40, marker='*')
            plt.title(fig_title)
            plt.savefig(fig_str)
            plt.close()
            left = layer_count[i]

        else:
            end = layer_count[i]+left
            plt.figure()
            plt.scatter(images[left:end, 0],
                        images[left:end, 1], c=models_kmeans_clusters[name], cmap='rainbow')
            plt.scatter(centroids[:, 0], centroids[:, 1],
                        c='black', alpha=0.8, s=40, marker='*')
            plt.title(fig_title)
            plt.savefig(fig_str)
            plt.close()
            left = end


"""
Generate horizontal bar figure of layer distribution between two models
"""


def generate_distribution_hbar(k, model_names, models_kmeans_clusters, model_str1='vgg16', model_str2='vgg19'):
    # Generate layer distribution
    title_str = model_str1 + '_' + model_str2 + '_layer_distribution'
    figure_str = title_str + '.jpg'
    labels = list(range(1, k+1))
    counts = {}
    for i, name in enumerate(model_names):
        freq = {}
        for j in models_kmeans_clusters[name]:
            if j in freq:
                freq[j] += 1
            else:
                freq[j] = 1
        for j in range(k):
            if j not in freq:
                freq[j] = 0
        counts[name] = {k: freq[k] for k in sorted(freq)}
    xlen16 = max(counts[model_str1].values())
    xlen19 = max(counts[model_str2].values())
    x_total = xlen19 + xlen16 + 1

    plt.figure()
    plt.barh(labels, counts[model_str1].values(), color='c', alpha=0.7)
    plt.barh(labels, counts[model_str2].values(), alpha=0.7,
             color='m', left=list(counts[model_str1].values()))
    plt.xticks(list(range(0, x_total)))
    plt.xlabel('Distribution')
    plt.yticks(labels, labels)
    plt.ylabel('Cluster #')
    red_patch = mpatches.Patch(color='c', label=model_str1)
    blue_patch = mpatches.Patch(color='m', label=model_str2)
    plt.legend(handles=[red_patch, blue_patch], loc='best')
    plt.title(title_str)
    plt.savefig(figure_str)
    plt.close()

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
    learned_images, true_layer_indexes = load_learned_images(
        model_names, "filters")
    # Preprocess the learned images
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
            best_match = (0, -100)
            for q in range(len(filters2)):
                similarity = structural_similarity(filters1[i], filters2[q])
                if similarity > best_match[1]:
                    best_match = (q, similarity)
            matches.append((i, *best_match))
        print(compare)
        print(matches)
        print()
        # Find the nearest neighber by computing the ssim difference
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
