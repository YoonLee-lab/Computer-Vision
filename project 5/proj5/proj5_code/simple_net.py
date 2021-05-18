import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        """
        Constructor for SimpleNet class to define the layers and loss function.

        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch's
        documention to understand what this means.
        """
        super().__init__()

        self.conv_layers = None
        self.fc_layers = None
        self.loss_criterion = None

        #######################################################################
        # Student code begins
        #######################################################################

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=(5,5)),
            nn.MaxPool2d(3,3),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=(5,5)),
            nn.MaxPool2d(3,3),
            nn.ReLU())

        self.fc_layers = nn.Sequential(
            nn.Linear(500,100),
            nn.ReLU(),
            nn.Linear(100,15))

        self.loss_criterion = nn.CrossEntropyLoss(reduction = 'sum')

        #######################################################################
        # Student code ends
        #######################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass with the network.

        Args:
            x: the (N,C,H,W) input images

        Returns:
            y: the (N,15) output (raw scores) of the net
        """
        model_output = None
        #######################################################################
        # Student code begins
        #######################################################################

        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1) # flatten
        model_output = self.fc_layers(x)

        #######################################################################
        # Student code ends
        #######################################################################
        return model_output
