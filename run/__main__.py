import tensorflow as tf
import argparse
import sys

import visualize
import compare

#Add optional arguments for control flow
parser = argparse.ArgumentParser(description="ML DL Final Project")
parser.add_argument('--create-visualizations', action="store_true")
parser.add_argument('--visualize', nargs="+")
parser.add_argument('--cluster', action="store_true")
parser.add_argument('--metrics', action="store_true")


if __name__ == "__main__":
    #Parse the arguments
    args = parser.parse_args()

    image_shape = (99,99,3)
    models = {
        "vgg16": tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(image_shape)),
        "vgg19": tf.keras.applications.VGG19(weights="imagenet", include_top=False, input_shape=(image_shape)),
        "inception_resnetv2": tf.keras.applications.InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(image_shape))
    }

    #Create the layer visualizations
    if args.create_visualizations:
        visualize.create_visualizations(models, image_shape, fr="all")

    #Display a layer visualization given a model name and a filter number
    elif args.visualize:
        visualize.load_and_display_activation_image(args.visualize[0], int(args.visualize[1]))

    #Perform cluster analysis
    elif args.cluster:
        compare.cluster_analysis(list(models.keys()))

    elif args.metrics:
        compare.comp_metrics(list(models.keys()))
