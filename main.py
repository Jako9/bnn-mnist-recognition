import torch
import argparse
import torchvision
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.autograd import Function
#import binarizePM1

###run 'pip install matplotlib' in console
import matplotlib.pyplot as plt


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

#binarizepm1 = Quantization(binarizePM1.binarize)
# BNN Definition
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = QuantizedConv2d(1, 32, 3, 1,quantization=Quantization(torch.sign))
        self.conv2 = QuantizedConv2d(32, 64, 3, 1,quantization=Quantization(torch.sign))
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = QuantizedLinear(9216, 4096,quantization=Quantization(torch.sign))
        self.fc2 = QuantizedLinear(4096, 10,quantization=Quantization(torch.sign))
        self.scale = Scale()

    def forward(self, x):
        x = self.conv1(x)
        #x = torch.sign(x)
        x = torch.relu(x)


        x = self.conv2(x)
        #x = torch.sign(x)
        x = torch.relu(x)

        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)


        x = self.fc1(x)
        #x = torch.sign(x)
        #x = torch.relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        #output = self.scale(x)
        return output

def train(args, model, train_loader, optimizer, device, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
                    #print(weight)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    #with torch.no_grad():

#with torch.no_grad():
    #for layer in model.children():
    #    if(type(layer) == type(nn.Linear(1,1))):
        #    layer.weight.data = torch.sign(layer.weight.data)
        #    layer.weight.data = torch.relu(layer.weight.data)
def test(model, device, test_loader):
    hit = 0
    total = 0
    print("Genauigkeit wird ausgewertet")

    with torch.no_grad():
        for data in test_loader:
            image, label = data
            result = model(image.to(device))
            for i, j in enumerate(result):
                if torch.argmax(j) == label[i]:
                    hit += 1
                total += 1

    for layer in model.children():
        if(type(layer) == type(QuantizedConv2d(1, 32, 3, 1,quantization=Quantization(torch.sign)))):
            for weight in layer.weight:
                print(weight)
    print(f"Genauigkeit: {100 * hit / total}%")



def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")  # you can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    # MNIST Datenset (herunter-) laden
    train_data = datasets.MNIST(
        "", train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),
                                                                    ThresholdTransform(thr_255=180)
                                                                    ]))

    # Trainingsdaten einlesen
    training_set = DataLoader(
        train_data, batch_size=256, shuffle=True)

    ############Testing image binarization##################
    for data in training_set:
        break;

    plt.imshow(data[0][0].view(28,28))
    plt.show()

    test_data = datasets.MNIST(
        "", train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),
                                                                    ThresholdTransform(thr_255=180)
                                                                    ]))

    test_set = DataLoader(
        test_data, batch_size=10, shuffle=True)

    # BNN Instanz
    bnn = NeuralNetwork().to(device)

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                            help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=3000, metavar='N',
                            help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                            help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.5, metavar='LR',
                            help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=1.5, metavar='M',
                            help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                            help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                            help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                            help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)


    # Anzahl Epochen zum Trainieren
    epochs = 1
    # Fortschritt (UI)
    optimizer = optim.Adadelta(bnn.parameters(), lr=args.lr)
    for epoch in range(epochs):
        train(args, bnn,training_set,optimizer,device,epoch)
        print(f"Fortschritt: {epoch+1}/{epochs}")


    # Statistik
    test(bnn,device,test_set)

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

class ThresholdTransform(object):
  def __init__(self, thr_255):
    self.thr = thr_255 / 255.  # input threshold for [0..255] gray level, convert to [0..1]

  def __call__(self, x):
    return (x > self.thr).to(x.dtype)  # do not change the data type

if __name__ == '__main__':
    main()
