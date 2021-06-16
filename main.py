import torch
import torchvision
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

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

# BNN Instanz
bnn = NeuralNetwork().to(device)

# Anzahl Epochen zum Trainieren
epochs = 3
# Fortschritt (UI)
epochs_done = 0
for epoch in range(epochs):
    for data in training_set:
        image, label = data
        bnn.zero_grad()
        result = bnn(image.to(device))
        loss = nn.functional.nll_loss(result.to(device), label.to(device))
        loss.backward()

    epochs_done += 1
    print(f"Fortschritt: {epochs_done}/{epochs}")

# Statistik
hit = 0
total = 0
print("Genauigkeit wird ausgewertet")

with torch.no_grad():
    for data in training_set:
        image, label = data
        result = bnn(image.to(device))
        for i, j in enumerate(result):
            if torch.argmax(j) == label[i]:
                hit += 1

            total += 1

print(f"Genauigkeit: {100 * hit / total}%")
