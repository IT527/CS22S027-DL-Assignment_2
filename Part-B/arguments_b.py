import argparse


def parsArg():
    parser = argparse.ArgumentParser(description="Fine-tune Pre-trained CNN Model.")

    # Add arguments to the parser
    parser.add_argument('-wp', '--wandb_project', default='finetuned', type=str, help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity', default='cs22s027', type=str, help='WandB Entity used to track experiments in the Weights & Biases dashboard.')
    parser.add_argument('-e', '--epochs', default=8, type=int, help='Number of epochs to train neural network.')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='Batch size used to train neural network.')
    parser.add_argument('-ad', '--augment_data', default='Yes', type=str, choices=['Yes', 'No'], help='Whether to add data augmentation.')
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='Learning rate used to optimize model parameters.')
    parser.add_argument('-m', '--momentum', default=0.9, type=float, help='Momentum used in momentum based SGD optimizer to optimize model parameters.')

    # Parse the arguments
    args = parser.parse_args()
    return args