import torch
import torch.nn as nn

def create_dynamic_conv_model(n_features, n_timesteps, n_outputs, num_conv_layers=2, base_channels=32, kernel_size=3):

    """Create a convolutional neural network

    Arguments
    -------------

    n_features: int = number of time series (here three for x, y, z acceleration axis)
    n_timesteps: int = number of time steps in each time series
    n_outputs: int = number of class for classification tasks
    num_conv_layers: int = number of CNN layers 
    base_channels: int = number of output channels in the first CNN layer. No. of output channels increase by a factor of 2 for each subsequent CNN layer
    kernel_size: int
    """

    if num_conv_layers == 0:
        return nn.Sequential(nn.Flatten(), nn.Linear(n_features*n_timesteps, n_outputs))
   
    layers = []
    padding = (kernel_size - 1) // 2

    # Add Conv1d layers based on the specified number
    for i in range(num_conv_layers):
        if i == 0:
            # First Conv1d layer
            layers.append(nn.Conv1d(in_channels=n_features, out_channels=base_channels, kernel_size=kernel_size, padding=padding))
        else:
            # Subsequent Conv1d layers
            layers.append(nn.Conv1d(in_channels=base_channels * (2**(i-1)), out_channels=base_channels * (2**i), kernel_size=kernel_size, padding=padding))
        layers.extend([nn.ReLU(), nn.MaxPool1d(2)])

    layers.append(nn.Dropout(0.5))

    # Flatten layer
    layers.append(nn.Flatten())

    # Calculate the input size for the first Linear layer after flattening
    input_size = int(base_channels * (2**(num_conv_layers - 1)) * (n_timesteps // (2**num_conv_layers)))


    # # Add remaining layers
    layers.append(nn.Linear(input_size, n_outputs))
    layers.append(nn.Sigmoid())

    # Create the model
    model = nn.Sequential(*layers)

    return model

def test_create_dynamic_conv_model(n_features, n_timesteps, n_outputs, num_conv_layers=1, base_channels=32, kernel_size=3):

    model = create_dynamic_conv_model(n_features, n_timesteps, n_outputs, num_conv_layers, base_channels, kernel_size)
    X = torch.zeros((5, n_features, n_timesteps))
    out = model(X)
    print(out.shape)


def create_dynamic_mlp(n_features, n_timesteps, n_outputs, num_layers=2, width=32):

    """Create a multi-layer perceptron

    Arguments
    -------------

    n_features: int = number of time series (here three for x, y, z acceleration axis)
    n_timesteps: int = number of time steps in each time series
    n_outputs: int = number of class for classification tasks
    num_layers: int = number of hidden layers 
    width: int = out[put size of the first hidden layer. Size of output increase by a factor of 2 for each subsequent hidden layer
    """

    if num_layers == 1:
        return nn.Sequential(nn.Flatten(), nn.Linear(n_features*n_timesteps, n_outputs))
    
    layers = [nn.Flatten()]

    # Add Conv1d layers based on the specified number
    for i in range(num_layers-1):
        if i == 0:
            # First Linear layer
            layers.append(nn.Linear(n_features*n_timesteps, width))
        else:
            # Subsequent Linear layers
            layers.append(nn.Linear(width * (2**(i-1)), width * (2**i)))
        
        layers.append(nn.ReLU())

    # Add remaining layers
    layers.append(nn.Linear(width * (2**(num_layers-2)), n_outputs))

    # Create the model
    model = nn.Sequential(*layers)

    return model

class dynamic_Conv2D(nn.Module):
    def __init__(self, 
                channels: int = None, 
                height: int = None, 
                width: int = None, 
                n_outputs: int = None, 
                num_conv_layers: int = 2, 
                base_channels: int = 32, 
                kernel_size: int = 4):
        super(dynamic_Conv2D, self).__init__()

        self.channels = channels
        self.height = height
        self.width = width
        self.n_outputs = n_outputs
        
        if self.channels is None or self.height is None or self.width is None or self.n_outputs is None:
            raise ValueError
        
        layers = []

        # Add Conv1d layers based on the specified number
        for i in range(num_conv_layers):
            if i == 0:
                # First Conv1d layer
                layers.append(nn.Conv2d(in_channels=self.channels, out_channels=base_channels, kernel_size=kernel_size))
            else:
                # Subsequent Conv1d layers
                layers.append(nn.Conv2d(in_channels=base_channels * (2**(i-1)), out_channels=base_channels * (2**i), kernel_size=kernel_size))
            layers.append(nn.ReLU())

        layers.extend([
            nn.Dropout(0.5),
            nn.MaxPool2d(2),
            nn.Flatten()
        ])
        self.CNN = nn.Sequential(*layers)
        self.fc_input_size = self.calculate_fc_input_size()


        # Define the fully connected linear layers
        self.linear = nn.Linear(self.fc_input_size, self.n_outputs)

    def calculate_fc_input_size(self):
        x = torch.randn(1, self.channels, self.height, self.width)
        x = self.CNN(x)
        return x.size(1)

    def forward(self, x):
        x = self.CNN(x)
        return self.linear(x)
