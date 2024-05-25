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

    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain = load_data(args.data)
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)

    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    ## Normalizing the data
    mean = np.mean(xtrain, axis=0, keepdims=True)
    std = np.std(xtrain, axis=0, keepdims=True)
    xtrain = normalize_fn(xtrain, mean, std)
    xtest = normalize_fn(xtest, mean, std) 
    
    # Make a validation set
    if not args.test:
         validation_percentage = 0.2
         N = xtrain.shape[0]
         num_elements = int(N * validation_percentage)
         indices = np.arange(N)

         ### Shuffling the elements
         np.random.permutation(indices)

         ### validation indices & training_indices

         valid_ind = indices[:num_elements]  # 20%
         train_ind = indices[num_elements:]  # 80%

         x_train_copy = np.copy(xtrain)
         y_train_copy = np.copy(ytrain)

         xtrain, xtest = xtrain[train_ind], x_train_copy[valid_ind]
         ytrain, ytest = ytrain[train_ind], y_train_copy[valid_ind]

    ### WRITE YOUR CODE HERE to do any other data processing

    # Dimensionality reduction (MS2)
    if args.use_pca:
        print("Using PCA")
        pca_obj = PCA(d=args.pca_d)
        pca_obj.find_principal_components(xtrain)
        xtrain = pca_obj.reduce_dimension(xtrain)
        xtest = pca_obj.reduce_dimension(xtest)
        ### WRITE YOUR CODE HERE: use the PCA object to reduce the dimensionality of the data

    ## 3. Initialize the method you want to use.
    n_classes = get_n_classes(ytrain)
    N_train, D_train = xtrain.shape
    N_test, D_test = xtest.shape
    C = 1 # channels
    dim = int(np.sqrt(D_train))

    if args.nn_type == "mlp":
        model = MLP(input_size=D_train, n_classes=n_classes, hidden_layers=[512, 512]) ### WRITE YOUR CODE HERE

    elif args.nn_type == "cnn":
        xtrain = xtrain.reshape((N_train, C, dim, dim)) # NCHW
        xtest = xtest.reshape((N_test, C, dim, dim)) # NCHW
        model = CNN(input_channels=C, n_classes=n_classes, filters=[32, 64], hidden_layers=[128], image_size=28)

    elif args.nn_type == "transformer":
        xtrain = xtrain.reshape((N_train, C, dim, dim))
        xtest = xtest.reshape((N_test, C, dim, dim))
        model = MyViT(chw=(C, dim, dim), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=n_classes)

    summary(model)

    # Trainer object
    method_obj = Trainer(model=model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)

    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(xtrain, ytrain)

    # Predict on unseen data
    preds = method_obj.predict(xtest)

    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    ## As there are no test dataset labels, check your model accuracy on validation dataset.
    # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.

    if not args.test:
        acc = accuracy_fn(preds, ytest)
        macrof1 = macrof1_fn(preds, ytest)
        print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
    
    if args.test:
        np.save("predictions", preds) 

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.

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