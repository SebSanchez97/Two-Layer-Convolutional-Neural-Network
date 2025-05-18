import torch
from torch import nn
import matplotlib.pyplot as plt
import torch
from numpy import load
import matplotlib.pyplot as plt

def read_npz(filename):
    """
    Reads the MNIST dataset from a numpy archive.
    
    Arguments:
    ----------
        filename: str
            - Filename for numpy archive

    Returns:
    --------
        X : np.ndarray
            - Normalised feature vectors
        Y : np.ndarray
            - One-hot encoded target class vectors
    """
    data = load(filename)

    X = torch.tensor(data['X'], dtype=torch.float32)
    Y = torch.tensor(data['Y'])

    Y = torch.argmax(Y,-1).type(torch.long)

    return X, Y

def hits(y_hat,y):
    """
    Calculate the number of correct predictions in a batch

    Arguments:
    ----------
        y_hat : torch.tensor
        y : torch.tensor

    Returns:
    --------
        hits : torch.tensor
            - Number of correct predictions
    """
    pred_ids = torch.argmax(y_hat, -1)
    corr = torch.eq(pred_ids, y)
    hits = torch.sum(corr)

    return hits

def plot_filters(X,H,z_1,v_1):
    """
    Filter visualisation

    Arguments:
    ----------
        X : torch.tensor                                   
            - Feature vector         
        H : torch.tensor                                   
            - CNN weights/filters          
        z_1 : torch.tensor                                 
            - CNN layer one output after bias    
        v_1 : torch.tensor                                 
            - CNN layer one output after activation        
    """

    C = H.shape[0]
    
    fig, axs = plt.subplots(4,C+1,figsize=(C+1,4))

    for i in range(C):
        axs[0,i+1].imshow(X.reshape(28,28))

    for i in range(C):
        axs[1,i+1].imshow(H[i,0,:,:].detach().numpy())

    for i in range(C):
        axs[2,i+1].imshow(z_1[0,i,:,:].detach().numpy())

    for i in range(C):
        axs[3,i+1].imshow(v_1[0,i,:,:].detach().numpy())

    plt.text(0.5, 0.5, "Input \nImage", horizontalalignment='center',
         verticalalignment='center', transform=axs[0,0].transAxes)
    
    plt.text(0.5, 0.5, "Filter", horizontalalignment='center',
            verticalalignment='center', transform=axs[1,0].transAxes)
    
    plt.text(0.5, 0.5, "$Z[1]$", horizontalalignment='center',
            verticalalignment='center', transform=axs[2,0].transAxes)
    
    plt.text(0.5, 0.5, "$V[1]$", horizontalalignment='center',
            verticalalignment='center', transform=axs[3,0].transAxes)

    for ax in fig.axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig("filters.png", bbox_inches='tight')

class CNN(nn.Module):
    def __init__(self, M_1, M_2, M_3, K, stride):
        super().__init__()
        """        
        Arguments:
        ---------
            M_1 : int
                - Kernel height
            M_2 : int
                - Kernel width
            M_3 : int
                - Kernel output channels
            K : int
                - Model output dimensions
            stride : int
                - Convolution step size
        """
        # CONVOLUTIONAL LAYER
        kernel_sz = (M_1, M_2)
        P_1 = 1
        self.conv_1 = nn.Conv2d(in_channels=P_1, out_channels=M_3, kernel_size=kernel_sz, stride=stride)

        # FULLY-CONNECTED LAYER
        input_shape = int(M_3 * (((28 - M_1) / stride) + 1) * (((28 - M_2) / stride) + 1))
        output_shape = K
        self.dense = nn.Linear(in_features=input_shape, out_features=output_shape)

        # ACTIVATIONS
        self.relu = nn.ReLU()
 
    def forward(self, x):
        """
        Arguments:
            x : Input vector (N,D)

        Returns:
            y_hat : torch.tensor
                - Output predcition (N,K)
            z_1 : torch.tensor
                - Output matrix after convolutional layer
            v_1 : torch.tensor
                - Output matrix after activation layer
        """
        new_shape = (-1, 1, 28, 28)
        x = torch.reshape(x, new_shape)

        z_1 = self.conv_1(x)
        v_1 = self.relu(z_1)

        v1_flat = torch.reshape(v_1, (v_1.shape[0], -1))
        y_hat = self.dense(v1_flat)

        return y_hat, z_1, v_1
