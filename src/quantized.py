from torch.autograd import Function
from torch import nn

class Quantize(Function):
    @staticmethod
    def forward(ctx, input, quantization):

        output = input.clone().detach()
        output = quantization.applyQuantization(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


quantize = Quantize.apply


class Quantization:
    def __init__(self, method):
        self.method = method

    def applyQuantization(self, input):
        return self.method(input)


def Binarize(tensor, quant_mode='det'):
    if quant_mode == 'det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0, 1).round().mul_(2).add_(-1)

class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        if input.size(1) != 784:
            input.data = Binarize(input.data)
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = Binarize(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out


class QuantizedLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        self.name = "QuantizedLinear"
        self.layerNR = kwargs.pop('layerNr', None)
        self.quantization = kwargs.pop('quantization', None)
        super(QuantizedLinear, self).__init__(*args, **kwargs)

    def forward(self, input):
        if self.bias is None:
            quantized_weight = quantize(self.weight, self.quantization)
            output = F.linear(input, quantized_weight)
            return output
        else:
            quantized_weight = quantize(self.weight, self.quantization)
            quantized_bias = quantize(self.bias, self.quantization)
            return F.linear(input, quantized_weight, quantized_bias)


class QuantizedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.name = "QuantizedConv2d"
        self.layerNR = kwargs.pop('layerNr', None)
        self.quantization = kwargs.pop('quantization', None)
        super(QuantizedConv2d, self).__init__(*args, **kwargs)

    def forward(self, input):
        if self.bias is None:
            quantized_weight = quantize(self.weight, self.quantization)
            output = F.conv2d(input, quantized_weight, self.bias, self.stride,
                              self.padding, self.dilation, self.groups)
            return output
        else:
            quantized_weight = quantize(self.weight, self.quantization)
            quantized_bias = quantize(self.bias, self.quantization)
            output = F.conv2d(input, quantized_weight, quantized_bias, self.stride,
                              self.padding, self.dilation, self.groups)
            return output


class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale
