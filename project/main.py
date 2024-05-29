import argparse

import numpy as np
from torchinfo import summary
import torch

from src.data import load_data
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer, MyViT, MyViTBlock
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes

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
        model = MLP(input_size=xtrain.shape[1], n_classes=n_classes, hidden_layers=[512, 512])
        model.to(args.device)
        method_obj = Trainer(model=model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size, device=args.device)

    elif args.nn_type == "cnn":
        xtrain = xtrain.reshape((xtrain.shape[0], C, dim, dim)) # NCHW
        xtest = xtest.reshape((xtest.shape[0], C, dim, dim)) # NCHW
        if not args.test:
            xval = xval.reshape((xval.shape[0], C, dim, dim))
        model = CNN(input_channels=C, n_classes=n_classes, filters=[32, 64], hidden_layers=[128], image_size=28)
        model.to(args.device)
        method_obj = Trainer(model=model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size, device=args.device)

    elif args.nn_type == "transformer":
        xtrain = xtrain.reshape((xtrain.shape[0], C, dim, dim))
        xtest = xtest.reshape((xtest.shape[0], C, dim, dim))
        if not args.test:
            xval = xval.reshape((xval.shape[0], C, dim, dim))
        model = MyViT(chw=(C, dim, dim), n_patches=7, n_blocks=4, hidden_d=8, n_heads=8, out_d=n_classes)
        model.to('cpu')
        method_obj = Trainer(model=model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size, device='cpu')

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
    