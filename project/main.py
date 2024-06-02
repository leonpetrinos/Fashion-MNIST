import argparse

import numpy as np
from torchinfo import summary
import torch

from src.data import load_data
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer, MyViT, MyViTBlock
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes
import matplotlib.pyplot as plt

def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """

    ## We load our data and flatten the images into vectors
    xtrain, xtest, ytrain = load_data(args.data)
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)

    ## Normalizing the data
    mean = np.mean(xtrain, axis=0, keepdims=True)
    std = np.std(xtrain, axis=0, keepdims=True)
    xtrain = normalize_fn(xtrain, mean, std)
    xtest = normalize_fn(xtest, mean, std) 

    ## Make a validation set
    if not args.test:
         validation_percentage = 0.2
         N = xtrain.shape[0]
         num_elements = int(N * validation_percentage)
         indices = np.arange(N)

         np.random.shuffle(indices)

         valid_ind = indices[:num_elements]  # 20%
         train_ind = indices[num_elements:]  # 80%

         x_train_copy = np.copy(xtrain)
         y_train_copy = np.copy(ytrain)

         xtrain, xval = xtrain[train_ind], x_train_copy[valid_ind]
         ytrain, yval = ytrain[train_ind], y_train_copy[valid_ind]

    ## Dimensionality reduction (MS2)
    if args.use_pca and args.nn_type == "mlp":
        print("Using PCA")
        pca_obj = PCA(d=args.pca_d)
        pca_obj.find_principal_components(xtrain)
        xtrain = pca_obj.reduce_dimension(xtrain)
        xtest = pca_obj.reduce_dimension(xtest)
        if not args.test:
            xval = pca_obj.reduce_dimension(xval)

    ## Initialize the method 
    n_classes = get_n_classes(ytrain)
    C = 1 # channels
    dim = int(np.sqrt(xtrain.shape[1]))

    if args.nn_type == "mlp":
        model = MLP(input_size=xtrain.shape[1], n_classes=n_classes, hidden_layers=[512, 256, 128])
        model.to(args.device)
        method_obj = Trainer(model=model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size, device=args.device)

    elif args.nn_type == "cnn":
        xtrain = xtrain.reshape((xtrain.shape[0], C, dim, dim)) # NCHW
        xtest = xtest.reshape((xtest.shape[0], C, dim, dim)) # NCHW
        if not args.test:
            xval = xval.reshape((xval.shape[0], C, dim, dim))
        model = CNN(input_channels=C, n_classes=n_classes)
        model.to(args.device)
        method_obj = Trainer(model=model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size, device=args.device)

    elif args.nn_type == "transformer":
        xtrain = xtrain.reshape((xtrain.shape[0], C, dim, dim))
        xtest = xtest.reshape((xtest.shape[0], C, dim, dim))
        if not args.test:
            xval = xval.reshape((xval.shape[0], C, dim, dim))
        model = MyViT(chw=(C, dim, dim), n_patches=7, n_blocks=10, hidden_d=64, n_heads=8, out_d=n_classes)
        model.to('cpu')
        method_obj = Trainer(model=model, lr=3e-4, epochs=5, batch_size=128, device='cpu')

    summary(model)

    ## Train and evaluate the method
    preds_train = method_obj.fit(xtrain, ytrain) # Fit on training data

    ## Performance on training data
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    ## Performance on validation data
    if not args.test:
        preds_val = method_obj.predict(xval) # Predict on unseen data
        acc = accuracy_fn(preds_val, yval)
        macrof1 = macrof1_fn(preds_val, yval)
        print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
    else: 
        preds_test = method_obj.predict(xtest)
        np.save("predictions", preds_test)

def plot_mlp(xtrain, ytrain, xval, yval, input_size, n_classes, hidden_layers, params, args):
    AccuracyTrain_PCA = []
    AccuracyTest_PCA = []
    AccuracyTrain = []
    AccuracyTest = []

    for param in params:       
        model = MLP(input_size, n_classes, hidden_layers)     
        method_obj = Trainer(model, lr=param, epochs=args.max_iters, batch_size=args.nn_batch_size)
        trainPrediction = method_obj.fit(xtrain, ytrain)
        AccuracyTrain.append(accuracy_fn(trainPrediction, ytrain))
        testPrediction = method_obj.predict(xval)
        AccuracyTest.append(accuracy_fn(testPrediction, yval))

    if not args.use_pca:
      print("NOW USING PCA")
      pca_obj = PCA(d=args.pca_d)
      pca_obj.find_principal_components(xtrain)
      xtrain_pca = pca_obj.reduce_dimension(xtrain)
      xtest_pca = pca_obj.reduce_dimension(xval)
      input_size_pca = xtrain_pca.shape[1]
     
      for param in params:
        model = MLP(input_size_pca, n_classes, hidden_layers)      
        method_obj = Trainer(model, lr=param, epochs=args.max_iters, batch_size=args.nn_batch_size)
        trainPredictions = method_obj.fit(xtrain_pca, ytrain)
        AccuracyTrain_PCA.append(accuracy_fn(trainPredictions, ytrain))
        testPrediction = method_obj.predict(xtest_pca)
        AccuracyTest_PCA.append(accuracy_fn(testPrediction, yval))

    plt.semilogx(params, AccuracyTrain, 'r', label="Train_data_NO_PCA")
    plt.semilogx(params, AccuracyTest, 'b', label="Test_data_NO_PCA")
    plt.semilogx(params, AccuracyTrain_PCA, 'g', label="Train_data_PCA")
    plt.semilogx(params, AccuracyTest_PCA, 'y', label="Test_data_PCA")
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Learning Rate')
    plt.legend()
    plt.show()

def plot_transformer(xtrain, ytrain, xval, yval, chw, n_patches, hidden_d, n_heads, out_d, params, args):
     AccTrain = []
     AccTest = []

     for param in params:  
        model = MyViT(chw, n_patches, param, hidden_d, n_heads, out_d)
        method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)
        trainPreds = method_obj.fit(xtrain, ytrain)
        AccTrain.append(accuracy_fn(trainPreds, ytrain))
        testPreds = method_obj.predict(xval)
        AccTest.append(accuracy_fn(testPreds, yval))

     plt.plot(params, AccTrain, 'r', label="Train data")
     plt.plot(params, AccTest, 'b', label="Test data")
     plt.xlabel('Number of blocks')
     plt.ylabel('Accuracy')
     plt.title('Accuracy vs Number of blocks')
     plt.legend()
     plt.show()

def plot_cnn(n_classes, xtrain, ytrain, xval, yval):
    train_accuracy, val_accuracy = [], []
    train_f1, val_f1 = [], []
    learning_rates = [1e-4, 1e-3, 1e-2]
    ep = 100
    for lr in learning_rates:
        print(f"lr: {lr}")
        model = CNN(input_channels=1, n_classes=n_classes)
        model.to('mps')
        method_obj = Trainer(model=model, lr=lr, epochs=ep, batch_size=args.nn_batch_size, device='mps')
        # Get training accuracy
        preds_train = method_obj.fit(xtrain, ytrain)
        acc = accuracy_fn(preds_train, ytrain)
        train_accuracy.append(acc)
        macrof1 = macrof1_fn(preds_train, ytrain)
        train_f1.append(macrof1)
        # Get validation accuracy
        preds_val = method_obj.predict(xval)
        acc = accuracy_fn(preds_val, yval)
        val_accuracy.append(acc)
        macrof1 = macrof1_fn(preds_val, yval)
        val_f1.append(macrof1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, train_accuracy, label='Training Accuracy', marker='o')
    plt.plot(learning_rates, val_accuracy, label='Validation Accuracy', marker='o')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--data', default="dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp",
                        help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")
    parser.add_argument('--use_pca', action="store_true", help="use PCA for feature reduction")
    parser.add_argument('--pca_d', type=int, default=100, help="the number of principal components")


    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")


    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
    