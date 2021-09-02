import torch
import argparse
import torchvision
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from config import *
from quantized import BinarizeLinear
from transformation import *
from export import exportThreshold, export

#run 'pip install matplotlib' in console
import matplotlib.pyplot as plt

# define BNN-structure
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
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


def train(args, model, train_loader, optimizer, device, epoch, iteration):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {}, Iteration: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, iteration, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


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
    f = open("../measurements/batchsize_"+str(BATCH_SIZE)+".txt", "a")
    f.write(str(100 * hit / total) + ",")
    f.close()
    model.train()
    return (100 * hit / total)


def main():
    if torch.cuda.is_available():
        # you can continue going on here, like cuda:1 cuda:2....etc.
        device = torch.device("cuda")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, metavar='N',
                        help='input batch size for training (default: 200)')
    parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                        help='input batch size for testing (default: 10) ')
    parser.add_argument('--epochs', type=int, default=EPOCHS, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step-size', type=int, default=STEP_SIZE, metavar='M',
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

    # Configure MNIST-Dataset => Training and Testset(s)
    iterationData = []
    for i in range (0, REPETITIONS):
        if(USE_PROBABILITY_TRANSFORM):
            iterationData.append(datasets.MNIST(
                "", train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),
                                                                              ProbabilityTransform()
                                                                                 ])))
        else:
            iterationData.append(datasets.MNIST(
                "", train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),
                                                                              ThresholdTransform(
                                                                                  max_val=THRESHOLD)
                                                                                 ])))

    if(SHOW_PROCESSED_NUMBERS):
        for iteration in iterationData:
            i = 0
            for data in iteration:
                if(i==SELECTED_NUMBER_INDEX):
                    break
                i+=1
            plt.imshow(data[0][0].view(28,28))
            plt.show()

    training_setData = []
    # Trainingsdaten einlesen
    for i in range (0, REPETITIONS):
        training_setData.append(DataLoader(
            iterationData[i], batch_size=args.batch_size, shuffle= not SHOW_PROCESSED_NUMBERS))

    test_data = datasets.MNIST(
        "", train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),
                                                                      ThresholdTransform(
                                                                                  max_val=THRESHOLD)

                                                                      ]))
    test_set = DataLoader(
        test_data, batch_size=args.test_batch_size, shuffle=True)


    #run the training
    bnn = NeuralNetwork().to(device)
    optimizer = optim.Adadelta(bnn.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for iteration in range (0, REPETITIONS):
            train(args, bnn, training_setData[iteration], optimizer, device, epoch, iteration)
            test(bnn,device,test_set)
            print(f"Progress: Epoch: {epoch+1}/{args.epochs}, Iteration: {iteration+1}/{REPETITIONS}")

    #evaluate calculated BNN
    accuracy = test(bnn, device, test_set)

    #exporting Weights
    #export(bnn)
    #exportThreshold(bnn)
    print("Done!")
    return accuracy


if __name__ == '__main__':
    accuracies = []
    for x in range(MEASUREMENT_RUNS):
        accuracies.append(main())

    print("-----Summary-----")
    for i,val in enumerate(accuracies):
        print("Run " + str(i) + ": " + str(val) +"%")
