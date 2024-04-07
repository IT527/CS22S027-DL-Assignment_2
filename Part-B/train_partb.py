import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
import wandb
import random
import timm #To get pre-trained py-torch models
from arguments_b import *

args = parsArg()
key = '90512e34eff1bdddba0f301797228f7b64f546fc'
batch_size = args.batch_size
augment_data = args.augment_data
learning_rate = args.learning_rate
momentum = args.momentum
epochs = args.epochs
wandb.login(key = key)

"""#Data Pre-processing"""

# %%capture
# !curl -SL https://storage.googleapis.com/wandb_datasets/nature_12K.zip > nature_12K.zip
# !unzip nature_12K.zip

train_dir='inaturalist_12K/train/'
test_dir='inaturalist_12K/val/'
categories=['Amphibia','Animalia','Arachnida','Aves','Fungi','Insecta','Mammalia','Mollusca','Plantae','Reptilia']

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

    # Define data preprocessing steps
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

    # Assign transformations to datasets
    train_dataset.dataset.transform = training_preprocess
    val_dataset.dataset.transform = validation_preprocess

    # Initialize DataLoader for both datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader

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

train_loader, val_loader = initialize_data_loaders(train_dataset, val_dataset, augment_data= augment_data, batch_size=batch_size, resize_dim=224)

run = wandb.init(project=args.wandb_project, entity=args.wandb_entity)

class ViTFinetuner(pl.LightningModule):
    def __init__(self, num_classes, learning_rate, momentum):
        super().__init__()
        self.save_hyperparameters()

        # Load the pre-trained Vision Transformer model
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)

        # Freeze all the layers
        for param in self.vit.parameters():
            param.requires_grad = False

        # Replace the classifier head with a new one
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)

    def training_step(self, train_batch):
        images, labels = train_batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, val_batch):
        images, labels = val_batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def test_step(self, test_batch):
        inputs, targets = test_batch
        predictions = self(inputs)
        loss = F.cross_entropy(predictions, targets)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        acc = (predictions.argmax(dim=1) == targets).float().mean()
        self.log('test_acc', acc, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=self.hparams.momentum)


    def on_train_epoch_end(self):
        validation_result = self.trainer.callback_metrics.get('val_loss'), self.trainer.callback_metrics.get('val_acc')
        training_result = self.trainer.callback_metrics.get('train_loss'), self.trainer.callback_metrics.get('train_acc')

        # Optionally, log the epoch number as well
        wandb.log({"val_accuracy": validation_result[1].item(),
                  'val_loss': validation_result[0].item(),
                   'train_accuracy': training_result[1].item(),
                   'train_loss': training_result[0].item(),
                   'epoch': self.current_epoch})

# Train the model
model = ViTFinetuner(num_classes=10, learning_rate=learning_rate, momentum = momentum)
trainer = pl.Trainer(max_epochs=epochs,  accelerator="gpu", devices=1)
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