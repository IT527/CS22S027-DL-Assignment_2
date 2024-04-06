"""#Importing Libraries"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import Subset, DataLoader, random_split
import os
import random
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from arguments_a import *


args = parsArg()
key = '90512e34eff1bdddba0f301797228f7b64f546fc'
batch_size = args.batch_size
activation_type = args.activation
use_batch_norm = args.use_batch_norm
augment_data = args.augment_data
layer_filters = args.filter_organization
learning_rate = args.learning_rate
dropout_rate = args.dropout_rate
filter_size = args.kernl
dense_neurons = args.dense_neurons
epochs = args.epochs
wandb.login(key = key)

"""#Importing Dataset"""
# %%capture
# !curl -SL https://storage.googleapis.com/wandb_datasets/nature_12K.zip > nature_12K.zip
# !unzip nature_12K.zip

train_dir='inaturalist_12K/train/'
test_dir='inaturalist_12K/val/'
categories=['Amphibia','Animalia','Arachnida','Aves','Fungi','Insecta','Mammalia','Mollusca','Plantae','Reptilia']

"""#Data Pre-processing"""
#Ensuring equal representation of each class in the validation set
def load_and_split_data(directory, validation_ratio=0.2):
    #Load the full dataset
    full_dataset = datasets.ImageFolder(root=directory)

    #Get indices for each class
    targets = torch.tensor(full_dataset.targets)
    classes = full_dataset.classes
    class_to_indices = {class_name: torch.where(targets == class_index)[0]
                        for class_index, class_name in enumerate(classes)}

    #Initialize the lists for train and validation indices
    train_indices = []
    val_indices = []

    #Seed the random number generator for reproducibility
    random.seed(52)

    #Split indices for each class
    for indices in class_to_indices.values():
        indices = indices.tolist()
        random.shuffle(indices)  #Randomly shuffle the indices
        split = int(len(indices) * (1 - validation_ratio))  #Calculate split point
        train_indices += indices[:split]  #Indices for training
        val_indices += indices[split:]  #Indices for validation

    # Create subsets for training and validation
    training_data = Subset(full_dataset, train_indices)
    validation_data = Subset(full_dataset, val_indices)

    return training_data, validation_data

train_dataset, val_dataset = load_and_split_data(train_dir)

def initialize_data_loaders(train_dataset, val_dataset, augment_data, batch_size, resize_dim=224):
    """
    Prepares and returns DataLoader instances for training and validation data sets.

    Args:
    - train_dataset: Dataset for training.
    - val_dataset: Dataset for validation.
    - augment_data: String indicating if data augmentation ('Yes') should be applied to training data.
    - batch_size: Number of samples per batch.
    - resize_dim: Target size for each image dimension.

    Returns:
    - Training and validation DataLoader instances.
    """

    #Define data preprocessing steps
    training_preprocess = transforms.Compose([
        transforms.RandomResizedCrop(resize_dim) if augment_data == "Yes" else transforms.Resize((resize_dim, resize_dim)),
        transforms.RandomHorizontalFlip() if augment_data == "Yes" else transforms.Lambda(lambda img: img),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    validation_preprocess = transforms.Compose([
        transforms.Resize((resize_dim, resize_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    #Assign transformations to datasets
    train_dataset.dataset.transform = training_preprocess
    val_dataset.dataset.transform = validation_preprocess

    #Initialize DataLoader for both datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader

#Initialize DataLoader for test dataset
def create_test_loader(test_dir, batch_size, resize_dim=224):
    test_transforms = transforms.Compose([
        transforms.Resize((resize_dim, resize_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return test_loader

# Create Test DataLoader
test_loader = create_test_loader(test_dir, batch_size=batch_size)

"""# CNN Model"""

class ImageClassifier(pl.LightningModule):

    def __init__(self, learning_rate, activation_type, use_batch_norm, augment_data, layer_filters, dropout_rate, filter_size, dense_neurons):
        super(ImageClassifier, self).__init__()
        self.val_losses = [] #To keep track of validation loss
        self.train_losses = []  #To keep track of training loss

        self.lr = learning_rate #Learning rate
        self.activation_type = activation_type  #Acticvation function to be used
        # self.use_batch_norm = use_batch_norm #Batch Normalization
        self.use_batch_norm = True if use_batch_norm == "Yes" else False
        self.augment_data = augment_data #Data Augmentation
        self.layer_filters = layer_filters #Number of layers in each filter
        self.dropout_rate = dropout_rate #Dropouts
        self.filter_size = filter_size  #Size of filter/Kernel
        self.dense_neurons = dense_neurons #Number of neurons in dense layer
        self.define_hyperparameters()

        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        self.choose_activation()
        # Input channels for iNaturalist images (RGB)
        in_channels = 3
        # Construct Convolutional Blocks
        for out_channels in layer_filters:
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, filter_size, padding=0))
            self.pool_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            if use_batch_norm:
                self.batch_norm_layers.append(nn.BatchNorm2d(out_channels))
            else:
                self.batch_norm_layers.append(nn.Identity())
            in_channels = out_channels  # Update in_channels for the next layer

        conv_output_size = 224 #Image size considered to be 224x224
        for _ in self.layer_filters:
            # Stride=1 and padding=0 for convolution layers, by default.
            conv_output_size = (conv_output_size - self.filter_size + 1)  # Convolution effect
            conv_output_size = conv_output_size // 2  # Pooling effect simplification

        self.flatten = nn.Flatten()
        flat_features = conv_output_size * conv_output_size * self.layer_filters[-1]
        self.dense_layer = nn.Linear(flat_features, self.dense_neurons)

        # Correct batch normalization after the dense layer if enabled
        if self.use_batch_norm:
            self.batch_norm_dense = nn.BatchNorm1d(self.dense_neurons)
        else:
            self.batch_norm_dense = nn.Identity()

        # The output layer that will contain 10 neurons for the 10 classes
        self.output_layer = nn.Linear(self.dense_neurons, 10)

        self.dropout = nn.Dropout(self.dropout_rate)

    #Forward Propagation
    def _forward_features(self, x):
        # Pass input through convolutional blocks
        for conv_layer, bn_layer, pool_layer in zip(self.conv_layers, self.batch_norm_layers, self.pool_layers):
            x = conv_layer(x)
            x = self.activation(x)
            x = bn_layer(x)
            x = pool_layer(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.flatten(x)
        x = self.dense_layer(x)
        x = self.batch_norm_dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

    #Activation Functions
    def choose_activation(self):
        activations = {
            "ReLU": nn.ReLU(),
            "GELU": nn.GELU(),
            "SiLU": nn.SiLU(),
            "Mish": nn.Mish()
        }
        self.activation = activations.get(self.activation_type, nn.ReLU())

    def define_hyperparameters(self):
        self.save_hyperparameters()

    #Training loss and accuracy logging
    def training_step(self, train_batch):
        inputs, targets = train_batch
        predictions = self(inputs)
        loss = F.cross_entropy(predictions, targets)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        acc = (predictions.argmax(dim=1) == targets).float().mean()
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    #Test loss and accuracy logging
    def validation_step(self, val_batch):
        inputs, targets = val_batch
        predictions = self(inputs)
        loss = F.cross_entropy(predictions, targets)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        acc = (predictions.argmax(dim=1) == targets).float().mean()
        self.log('val_acc', acc, on_epoch=True, prog_bar=True, logger=True)
        return loss

    #Test loss and accuracy logging
    def test_step(self, test_batch):
        inputs, targets = test_batch
        predictions = self(inputs)
        loss = F.cross_entropy(predictions, targets)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        acc = (predictions.argmax(dim=1) == targets).float().mean()
        self.log('test_acc', acc, on_epoch=True, prog_bar=True, logger=True)
        return loss

    #Using optimizer function
    def configure_optimizers(self):
       return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)

    #Logging results to wandb and pytorch lightning
    def on_train_epoch_end(self):
        validation_result = self.trainer.callback_metrics.get('val_loss'), self.trainer.callback_metrics.get('val_acc')
        training_result = self.trainer.callback_metrics.get('train_loss'), self.trainer.callback_metrics.get('train_acc')

        # Optionally, log the epoch number as well
        wandb.log({"val_accuracy": validation_result[1].item(),
                  'val_loss': validation_result[0].item(),
                   'train_accuracy': training_result[1].item(),
                   'train_loss': training_result[0].item(),
                   'epoch': self.current_epoch})


# Initialize your model with hyperparameters from wandb
config = {
    "learning_rate": args.learning_rate,
    "activation_type": args.activation,
    "use_batch_norm": args.use_batch_norm,
    "augment_data": args.augment_data,
    "layer_filters": args.filter_organization,
    "dropout_rate": args.dropout_rate,
    "filter_size": args.kernel,
    "dense_neurons": args.dense_neurons,
    "epochs": args.epochs
}
run = wandb.init(config=config, project=args.wandb_project, entity=args.wandb_entity)

model = ImageClassifier(
    learning_rate=learning_rate,
    activation_type=activation_type,
    use_batch_norm=use_batch_norm,
    augment_data=augment_data,
    layer_filters=layer_filters,
    dropout_rate=dropout_rate,
    filter_size=filter_size,
    dense_neurons=dense_neurons
) 
wandb_logger = WandbLogger()
wandb.run.name = 'lr_' + str(learning_rate) + '_act_' + str(activation_type) + '_bn_' + str(use_batch_norm) + '_augment_' + str(augment_data) + '_lf_' + str(
            layer_filters) + '_kernel_' + str(filter_size) + '_dp_' + str(dropout_rate) + '_dn_' + str(dense_neurons) + '_epoch_' + str(epochs)

trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu", devices=1)
train_loader, val_loader = initialize_data_loaders(train_dataset, val_dataset, augment_data = augment_data, batch_size=batch_size)
trainer.fit(model, train_loader, val_loader)
#Printing accruacy and loss for training and validation datasets
train_loss = trainer.logged_metrics['train_loss']
train_acc = trainer.logged_metrics['train_acc'] * 100  # Convert proportion to percentage
val_loss = trainer.logged_metrics['val_loss']
val_acc = trainer.logged_metrics['val_acc'] * 100  # Convert proportion to percentage

print(f"Training Accuracy: {train_acc:.2f}%")
print(f"Training Loss: {train_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")

test_results = trainer.test(model,  test_loader)
#Printing accuracy and loss for test dataset
if test_results:
    test_loss = test_results[0]["test_loss"]
    test_acc = test_results[0]["test_acc"] * 100  # Convert proportion to percentage

    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

wandb.finish()
