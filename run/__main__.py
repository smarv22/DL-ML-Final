import tensorflow as tf
import argparse
import sys

import visualize
import compare

# Add optional arguments for control flow
parser = argparse.ArgumentParser(description="ML DL Final Project")
parser.add_argument('--create-visualizations', action="store_true")
parser.add_argument('--create-visualizations-filters', action="store_true")
parser.add_argument('--visualize', nargs="+")
parser.add_argument('--cluster', action="store_true")
parser.add_argument('--filters-analysis', action="store_true")
parser.add_argument('--create-highres-visualization', nargs="+")
parser.add_argument('--metrics', action="store_true")


if __name__ == "__main__":
    # Parse the arguments
    args = parser.parse_args()

    #Bump up image resolution if we are creating a high res version
    if args.create_highres_visualization:
        image_shape = (1000, 1000, 3)
    else:
        image_shape = (99, 99, 3)
    models = {
        "vgg16": tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(image_shape)),
        "vgg19": tf.keras.applications.VGG19(weights="imagenet", include_top=False, input_shape=(image_shape)),
        "inception_resnetv2": tf.keras.applications.InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(image_shape)),
        "mobilenet": tf.keras.applications.MobileNet(weights="imagenet", include_top=False, input_shape=(image_shape)),
        "mobilenetv2": tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=(image_shape))
    }

    # The layer index is the true layer index, not the conv layer index
    layers_filters = {
        "vgg16": [1, 4, 17],
        "vgg19": [1, 4, 20],
        "inception_resnetv2": []
    }
    layers_filters_compares = [
        (("vgg16", 1), ("vgg19", 1)),
        (("vgg16", 4), ("vgg19", 4)),
        (("vgg16", 17), ("vgg19", 20)),
    ]

    # Create the layer visualizations
    if args.create_visualizations:
        visualize.create_full_layer_visualizations(models, image_shape)

    # Create visualizations for individual filters grouped by layers
    elif args.create_visualizations_filters:
        visualize.create_layers_filters_visualizations(
            models, layers_filters, image_shape)

    # Display a layer visualization given a model name and a filter number
    elif args.visualize:
        visualize.load_and_display_activation_image(
            args.visualize[0], int(args.visualize[1]))

    # Perform cluster analysis
    elif args.cluster:
        compare.cluster_analysis(list(models.keys()))

    # Perform basic metrics analysis on filters from specified layers
    elif args.filters_analysis:
        compare.filters_analysis(list(models.keys()), layers_filters_compares)

    #Perform basic metrics like l2 and ssim
    elif args.metrics:
        compare.comp_metrics(list(models.keys()))

    #Creates a high res visualization of a layer or filter activation
    elif args.create_highres_visualization:
        #Note, layer is the true layer index not the conv2d layer index
        fr = args.create_highres_visualization[2] 
        if not fr == "all":
            fr = int(args.create_highres_visualization[2])
        visualize.create_highres_visualization(models[args.create_highres_visualization[0]], int(args.create_highres_visualization[1]), fr, image_shape)
