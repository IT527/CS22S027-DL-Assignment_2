# CS22S027-DL-Assignment-2, Part-A

### Details
Name: Ishaan Taneja </br>
Roll No.: CS22S027 </br>
</br>

This project is an implementation of a convolution neural network from scratch using Python. It is designed to be flexible, allowing adjustments to various parameters such as network architecture, kernel size, number of filters, activation functions, and experiment tracking using wandb.


### Dependencies
 - python
 - wandb library
 - torch library
 - torchvision library 
 - pytorch_lightning library
 - matplotlib (Optional: if you want to plot grid of sample test prediction)

To download all the necessary dependencies, you can run: `pip install -r requirements_a.txt`


### Clone and Download Instructions
Clone the repository or download the project files. Ensure that python, other required packages along with the 'iNaturalist dataset' are installed in the project directory.</br>
To download and unzip dataset, run the command: 
</br>
`curl -O -L https://storage.googleapis.com/wandb_datasets/nature_12K.zip && unzip nature_12K.zip`
</br>
To clone the repository directly to you local machine, ensure git is installed, run the command: 
</br>
`git clone https://github.com/IT527/CS22S027-DL-Assignment_2.git`
</br>
</br>
Alternatively, you can download the entire repository as a .zip file from the Download ZIP option provided by github.


### Usage
To run the python script, navigate to the project directory and run: `python train_parta.py [OPTIONS]`
</br>
Ensure to connect to GPU to run, in case using CPU, replace 'gpu' by 'cpu' in line 293 of python script.
</br>
The 'OPTIONS' can take different values for parameters to select dataset, modify network architecture, select activation function and many more.</br>
The possible arguments and respective values for 'OPTIONS' are shown in the table below:</br>

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | image_classifier | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | cs22s027  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-e`, `--epochs` | 8 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 64 | Batch size used to train neural network. | 
| `-a`, `--activation` | Mish | choices:  ["ReLU", "GELU", "SiLU", "Mish"] |
| `-ubn`, `--use_batch_norm` | No | choices:  ["Yes", "No"] |
| `-ad`, `--augment_data` | Yes | choices:  ["Yes", "No"] |
| `-fo`, `--filter_organization` | [128,64,32,64,128] | Number of filters in each layer as comma-separated values e.g., 128,64,32,64,128 |
| `-lr`, `--learning_rate` | 0.001 | Learning rate used to optimize model parameters | 
| `-dr`, `--dropout_rate` | 0.3 | Dropout rate used to train the model | 
| `-dn`, `--dense_neurons` | 128 | Number of neurons in the dense layer of the neural network. | 
| `-kn`, `--kernl` | 3 | Size of kernel used in convolution layers |



An example run with epochs 5 and dropout rate of 0.5 : `python train_parta.py --epochs 5 --dropout_rate 0.5`

</br>

On execution of the file as shown above, loss and accuracies for the train, validation and test dataset will be printed on the terminal. Along with it, the plots highlighting the loss and accuracies for each epochs, for both train and validation dataset, will be logged onto the wandb project.</br>
To access plots in wandb, ensure to replace the given key with your wandb API key.</br>


### Adding new activation function and optimiser
To add any new optimiser, you can create add a code for it in function 'choose_activation()' within class 'ImageClassifier'.
Similarly, any new optimizer function can be added within the function 'configure_optimizers()'.

### Additional Resources and help
Included in the project is DL_Assignment_2_A.ipynb, compatible with Jupyter Notebook or Google Colab. It encompasses CNN code, sweep operations, and logging utilities like test prediction sample images. For tailored runs, you may need to adjust configurations and uncomment sections in the notebook to log specific metrics or plots. The notebook serves as a practical reference for understanding the project's workflow. </br>
All the plots are generated and logged to wandb using this file only, while for a new configuration one can run the train_parta.py file as shown above.
</br>
</br>
The sweep details for choosing the hyperparameters, runs, sample images, and related plots can be viewed at: ``




