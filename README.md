# Fashion-MNIST Machine Learning Project (93.6% accuracy)

### Description
This project explores three machine learning techniques to classify the [Fashion-MNIST dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist), achieving an accuracy of 93.6% on unseen data. 
The techniques explored are Multilayer Perceptrons (MLP), Convolutional Neural Networks (CNN), and Visual Transformers (ViT). 
For more information on the optimal architectures we found and a discussion of our results, check out our [report](https://github.com/leonpetrinos/Fashion-MNIST/blob/main/report.pdf).

### How to Run the Project
To clone the repository, type:
```bash
git clone https://github.com/leonpetrinos/Fashion-MNIST.git
```

The dataset is provided in the root directory as a zip file. Unzip before doing any other command.
Navigate to the `/project` directory and here are some sample commands to execute:
#### MLP
```bash
python main.py --data ../dataset --nn_type mlp --lr 2e-4 --max_iters 30
```

#### CNN
```bash
python main.py --data ../dataset --nn_type cnn --lr 1e-3 --max_iters 200
```

#### ViT
```bash
python main.py --data ../dataset --nn_type transformer --lr 2e-4 --max_iters 30
```

#### Additional Arguments
* `--device <device_name>`: For CNN and MLP, specify the device (cpu by default, mps for MacOS).
* `--use_pca`: Allows the use of Principal Component Analysis (dimensionality reduction technique) and only works for the MLP. 
* `--pca_d <number_of_principal_components>`: Number of principal components for PCA.
* `--nn_batch_size <batch_size>`: Modify the batch size for stochastic gradient descent.
* `--test`: Train on the whole training set and output predictions on the test set. Without this argument, the model trains on 80% of the data (20% for validation) and prints validation accuracies and F1 scores.

### Requirements
Install the necessary packages using:
```bash
pip install torch torchinfo matplotlib numpy
```

If case of errors even when all the packages are installed, run:
```bash
pip install opencv-python==4.6.0.66
```

### Collaborators
| Name | GitHub |
| ---- | ------ |
| Leon Petrinos | [leonpetrinos](https://github.com/leonpetrinos) |
| Andrea Trugenberger | [AndreaTrugenberger](https://github.com/AndreaTrugenberger) |
| Youssef Chelaifa | [chelaifa](https://github.com/chelaifa) |
