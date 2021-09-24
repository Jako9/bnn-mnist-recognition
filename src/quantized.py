import torch
from torch.autograd import Function
from torch import nn


def Binarize(tensor, quant_mode='det'):
    """Binarize a tensor

    Args:
        tensor (torch.Tensor): A tensor to be binarized
        quant_mode (str, optional): A choice of quantisation mode (at times there were other quantisation methodes). Defaults to 'det'.

    Returns:
        torch.Tensor: The binarized tensor
    """
    if quant_mode == 'det':
        return tensor.sign()

class BinarizeLinear(nn.Linear):
    """An implementation of a binarized LinearLayer

    Args:
        nn : The pytorch implementation of LinearLayer
    """

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        """Forward data though the layer

        Args:
            input (torch.Tensor): The output from the layer before

        Returns:
            torch.Tensor: The activation values of this layer
        """
        # If this is not the first layer, binarize theoutput of the layer before
        if input.size(1) != 784:
            input.data = Binarize(input.data)
        # binarize the weights and bias
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = Binarize(self.weight.org)
        if not hasattr(self.bias, 'org'):
            self.bias.org = self.bias.data.clone()
        self.bias.data = Binarize(self.bias.org)
        # feed the binarized weights and inputs to nn.linear to calculate the neuron activations
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out
