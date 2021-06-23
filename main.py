import torch
import argparse
import torchvision
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader



# BNN Definition
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
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
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

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

    print(f"Genauigkeit: {100 * hit / total}%")

    

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")  # you can continue going on here, like cuda:1 cuda:2....etc. 
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    # MNIST Datenset (herunter-) laden
    mnist_data = datasets.MNIST(
        "", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    # Trainingsdaten einlesen
    training_set = DataLoader(
        mnist_data, batch_size=10, shuffle=True)

    # BNN Instanz
    bnn = NeuralNetwork().to(device)

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                            help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                            help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                            help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                            help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
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
    epochs = 3
    # Fortschritt (UI)
    optimizer = optim.Adadelta(bnn.parameters(), lr=args.lr)
    for epoch in range(epochs):
        train(args, bnn,training_set,optimizer,device,epoch)
        print(f"Fortschritt: {epoch+1}/{epochs}")


    # Statistik
    test(bnn,device,training_set)

if __name__ == '__main__':
    main()





