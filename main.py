import torch
import argparse
import torchvision
from torch import nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader



# BNN Definition
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        # Berechnungspipeline
        self.run = nn.Sequential(
            # Layer: 784 -> 64
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            # Layer: 64 -> 64
            nn.Linear(64, 64),
            nn.ReLU(),
            # Layer: 64 -> 10
            nn.Linear(64, 10),
            nn.ReLU(),
        )

    # Berechnung
    def forward(self, x):
        x = self.flatten(x)
        x = self.run(x)
        return x

def train(model, train_loader, optimizer, device, epoch):
    for data in train_loader:
        image, label = data
        model.zero_grad()
        result = model(image.to(device))
        loss = nn.functional.nll_loss(result.to(device), label.to(device))
        loss.backward()

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
        train(bnn,training_set,optimizer,device,epoch)
        print(f"Fortschritt: {epoch+1}/{epochs}")


    # Statistik
    test(bnn,device,training_set)

if __name__ == '__main__':
    main()





