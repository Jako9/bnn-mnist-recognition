import torch
import argparse
import torchvision
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.autograd import Function
from torch.optim.lr_scheduler import StepLR
#import binarizePM1

# run 'pip install matplotlib' in console
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
        self.infl_ratio = 1
        self.fc1 = BinarizeLinear(784, 500)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(500)
        self.fc2 = BinarizeLinear(500, 1024)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = BinarizeLinear(1024, 1024)
        self.htanh3 = nn.Hardtanh()
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 10)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.htanh3(x)

        x = self.fc4(x)
        x = self.logsoftmax(x)

        return x


def train(args, model, train_loader, optimizer, device, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # print(weight)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    # with torch.no_grad():

# with torch.no_grad():
    # for layer in model.children():
    #    if(type(layer) == type(nn.Linear(1,1))):
        #    layer.weight.data = torch.sign(layer.weight.data)
        #    layer.weight.data = torch.relu(layer.weight.data)


def test(model, device, test_loader):
    hit = 0
    total = 0
    print("Evaluating accuracy")
    model.eval()

    with torch.no_grad():
        for data in test_loader:
            image, label = data
            result = model(image.to(device))
            for i, j in enumerate(result):
                if torch.argmax(j) == label[i]:
                    hit += 1
                total += 1

    print(f"Accuracy: {100 * hit / total}%")
    model.train()


def exportThreshold(model):
    print("Export thresholds")
    # Clear File
    f = open("export/thresholds.txt", "w")
    f.write("")
    f.close()
    f = open("export/thresholds.txt", "a")
    for layer in model.modules():
        if(type(layer) == type(nn.BatchNorm1d(1024))):
            layerThreshold = "["
            for node in layer.weight:
                layerThreshold += (str(node.item())+",")
            layerThreshold = layerThreshold[:-1]
            layerThreshold += "]"
            f.write(layerThreshold)
    f.close()


def export(model):
    print("Starting export...")
    # Clear File
    f = open("export/weights.txt", "w")
    f.write("")
    f.close()
    #torch.save(model.state_dict(), "export/model.pt")
    f = open("export/weights.txt", "a")
    cnt = 0
    layerCount = 1
    # f.write(model.fc1.weight)
    # print(model.fc2.weight)
    torch.set_printoptions(profile="full")
    for layer in model.modules():
        if(type(layer) == type(BinarizeLinear(2048, 2048))):
            layerWeights = ""
            print("----Exporting layer " + str(layerCount) + "----")
            layerCount += 1
            finishedNodes = 0
            #nextStep = 0
            totalLayerNodes = layer.weight.size()[0]
            printProgressBar(0, totalLayerNodes,
                             prefix='Progress:', suffix='Complete', length=50)
            for node in layer.weight:
                #cnt +=1
                # f.write(str(node))
                # if((finishedNodes / totalLayerNodes) >= nextStep):
                #nextStep += 0.05
                #print("{:.1f}".format((finishedNodes / totalLayerNodes) * 100) + "% done")
                for cnt, edge in enumerate(node):
                    # print(edge)
                    layerWeights += str(max(int(edge.item()), 0))
                    #layerWeights += str(edge)
                    # break
                finishedNodes += 1
                printProgressBar(finishedNodes, totalLayerNodes,
                                 prefix='Progress:', suffix='Complete', length=50)
            f.write(layerWeights[:-1])

    f.close()


def main():
    if torch.cuda.is_available():
        # you can continue going on here, like cuda:1 cuda:2....etc.
        device = torch.device("cuda")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='input batch size for training (default: 200)')
    parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                        help='input batch size for testing (default: 10) ')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.5, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step-size', type=int, default=25, metavar='M',
                        help='Learning step size (default: 5)')
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

    # MNIST Datenset (herunter-) laden
    train_data = datasets.MNIST(
        "", train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),
                                                                     ThresholdTransform(
                                                                         thr_255=128)
                                                                     ]))

    # Trainingsdaten einlesen
    training_set = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True)

    ############Testing image binarization##################
    for data in training_set:
        break

    # plt.imshow(data[0][0].view(28,28))
    # plt.show()

    test_data = datasets.MNIST(
        "", train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),
                                                                      ThresholdTransform(
                                                                          thr_255=128)
                                                                      ]))

    test_set = DataLoader(
        test_data, batch_size=args.test_batch_size, shuffle=True)

    # BNN Instanz
    bnn = NeuralNetwork().to(device)
    # optimizer = optim.Adam(bnn.parameters(), lr=args.lr)
    # # optimizer = Clippy(model.parameters(), lr=args.lr)

    # scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Fortschritt (UI)
    optimizer = optim.Adadelta(bnn.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        train(args, bnn, training_set, optimizer, device, epoch)
        print(f"Progress: {epoch+1}/{args.epochs}")

    # Statistik
    test(bnn, device, test_set)

    export(bnn)
    exportThreshold(bnn)
    print("Done!")


def Binarize(tensor, quant_mode='det'):
    if quant_mode == 'det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0, 1).round().mul_(2).add_(-1)


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


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


class ThresholdTransform(object):
    def __init__(self, thr_255):
        self.thr = thr_255 / \
            255.  # input threshold for [0..255] gray level, convert to [0..1]

    def __call__(self, x):
        return (x > self.thr).to(x.dtype)  # do not change the data type


if __name__ == '__main__':
    main()
