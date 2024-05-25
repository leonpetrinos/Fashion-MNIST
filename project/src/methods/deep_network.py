import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from ..utils import accuracy

## MS2

##############################################################################################################################
##############################################################################################################################

class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes, hidden_layers=[512, 256, 128]):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)
        
        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        in_size = input_size
    
        # create a list containing all layers including hidden ones
        layers = []
        for out_size in hidden_layers:
            layers.append(nn.Linear(in_size, out_size)) # add layer
            layers.append(nn.ReLU())
            in_size = out_size
            
        layers.append(nn.Linear(in_size, n_classes)) # append the final layer 
        self.mlp_model = nn.Sequential(*layers) 
    
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = self.mlp_model(x)
        return x

##############################################################################################################################
##############################################################################################################################

class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes, filters=[6, 16], hidden_layers=[120, 84], image_size=28):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        in_channels = input_channels
        conv_layers = []

        final_size = image_size
        for out_channels in filters:
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)) # perform convolution
            conv_layers.append(nn.ReLU()) # apply activation function
            conv_layers.append(nn.MaxPool2d(kernel_size=2)) # apply max pooling reducing dimension 
            in_channels = out_channels
            final_size = final_size // 2 # final dimension divided by two because of max pooling 

        self.convolution_model = nn.Sequential(*conv_layers) # convolution model is ready 

        in_size = in_channels * final_size * final_size # input size is the flattened version of the convolutions output
        mlp_layers = []
        for out_size in hidden_layers:
            mlp_layers.append(nn.Linear(in_size, out_size)) # add layer
            mlp_layers.append(nn.ReLU()) # apply actvation function
            in_size = out_size
        mlp_layers.append(nn.Linear(in_size, n_classes))

        self.mlp_model = nn.Sequential(*mlp_layers) # mlp model is ready
        
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """

        x = self.convolution_model(x)
        x = torch.flatten(x, 1)
        x = self.mlp_model(x)

        return x
    

##############################################################################################################################
##############################################################################################################################

class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"
        d_head = int(d / n_heads)
        self.d_head = d_head

        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):

                # Select the mapping associated to the given head.
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]

                # Map seq to q, k, v.
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                # Calculate attention scores
                attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
                
                # Apply softmax to get attention weights
                attention_weights = self.softmax(attention_scores)
                
                # Apply attention weights to the values
                seq_result.append(torch.matmul(attention_weights, v))
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super().__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out

class MyViT(nn.Module):
    """
    A Transformer-based neural network
    """

    def __init__(self, chw, n_patches, n_blocks, hidden_d, n_heads, out_d):
        """
        Initialize the network.
        
        """
        super().__init__()

        # Attributes
        self.chw = chw # (C, H, W)
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        # Input and patches sizes
        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # Positional embedding
        self.positional_embeddings = self.get_positional_embeddings(n_patches ** 2 + 1, hidden_d) 

        # Transformer blocks
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        # Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        n, c, h, w = x.shape

        # Divide images into patches.
        patches = self.patchify(x, self.n_patches) 

        # Map the vector corresponding to each patch to the hidden size dimension.
        tokens = self.linear_mapper(patches) 

        # Add classification token to the tokens.
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Add positional embedding.
        out =  tokens + self.positional_embeddings.repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Get the classification token only.
        out = out[:, 0]

        # Map to the output distribution.
        out = self.mlp(out) 

        return out

    def patchify(self, images, n_patches):
        n, c, h, w = images.shape

        assert h == w # We assume square image.

        patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
        patch_size = h // n_patches 

        for idx, image in enumerate(images):
            for i in range(n_patches):
                for j in range(n_patches):
                    # Extract the patch of the image.
                    patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size] 
                    # Flatten the patch and store it.
                    patches[idx, i * n_patches + j] = patch.flatten() 

        return patches

    def get_positional_embeddings(self, sequence_length, d):
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                if j % 2 == 0:
                    result[i, j] = np.sin(i / (10000 ** (j / d)))
                else:
                    result[i, j] = np.cos(i / (10000 ** ((j - 1) / d)))
        return result


##############################################################################################################################
##############################################################################################################################

class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size

        self.criterion = nn.CrossEntropyLoss()
        if isinstance(model, MyViT): self.optimizer = torch.optim.Adam(params=model.parameters(), lr=lr) 
        else: self.optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader, ep)

    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        self.model.train()
        train_loss = 0.0
        for it, batch in enumerate(dataloader):
            # Load a batch, break it down in images and targets
            x, y = batch
            # Run forward pass
            logits = self.model(x) 

            # logits = F.softmax(logits, dim=1) ## NOT SURE ABOUT THIS
            
            # Compute loss 
            loss = self.criterion(logits, y.long()) 
            # Run backward pass.
            loss.backward() 
            # Accumulate and average the training loss
            train_loss += loss.detach().cpu().item() / len(dataloader)
            # Update the weights
            self.optimizer.step() 
            # Zero-out the accumulated gradients.
            self.model.zero_grad() 
            
            print(f'\rEpoch {ep + 1}/{self.epochs} | '
                  f'Batch {it + 1}/{len(dataloader)} | '
                  f'Loss: {loss:.4f} | '
                  f'Accuracy: {accuracy(logits, y):.2f}%', end='')
    
        # Print averaged training loss at the end of each epoch
        # print(f'\nEpoch {ep + 1}/{self.epochs} | Average Training Loss: {train_loss:.4f}')
        print(f'\n{"-"*65}')
        

    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation, 
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for it, batch in enumerate(dataloader):
                logits = self.model(batch[0])
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds)

        pred_labels = torch.cat(all_preds)
        return pred_labels
    
    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        # First, prepare data for pytorch
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(), 
                                      torch.from_numpy(training_labels))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.cpu().numpy()