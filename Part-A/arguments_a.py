import argparse

def parse_int_list(value):
        # Split the input string by commas and convert each split element to an integer
        int_list = list(map(int, value.split(',')))
        return int_list

def parsArg():
    parser = argparse.ArgumentParser(description="Image Classifier CNN.")

    # Add arguments to the parser
    parser.add_argument('-wp', '--wandb_project', default='image_classifier', type=str, help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity', default='cs22s027', type=str, help='WandB Entity used to track experiments in the Weights & Biases dashboard.')
    parser.add_argument('-e', '--epochs', default=8, type=int, help='Number of epochs to train neural network.')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='Batch size used to train neural network.')
    parser.add_argument('-a', '--activation', default='Mish', type=str, choices=['Mish', 'ReLU', 'SiLU', 'GELU'], help='Activation function to be used.')
    parser.add_argument('-ubn', '--use_batch_norm', default='No', type=str, choices=['Yes', 'No'], help='Whether to use batch normalization.')
    parser.add_argument('-ad', '--augment_data', default='Yes', type=str, choices=['Yes', 'No'], help='Whether to add data augmentation.')
    parser.add_argument('-fo', '--filter_organization', default=[128,64,32,64,128], type=parse_int_list, help='Number of filters in each layer as comma-separated values e.g., 128,64,...')
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='Learning rate used to optimize model parameters.')
    parser.add_argument('-dr', '--dropout_rate', default=0.3, type=float, help='Dropout rate used to train neural network.')
    parser.add_argument('-k', '--kernel', default=3, type=int, help='Size of kernel.')
    parser.add_argument('-dn', '--dense_neurons', default=128, type=int, help='Number of neurons in dense layer.')
    parser.add_argument('-kn', '--kernl', default=3, type=int, help='Size of kernel.')
    # Parse the arguments
    args = parser.parse_args()
    return args
