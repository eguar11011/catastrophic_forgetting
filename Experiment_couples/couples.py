import torchvision
from torchvision import transforms
import torch
from torch import nn
from torch.utils.data import DataLoader

import os



total_fase = 2 
save_model = True
load_model = False
device = "cuda"
data_dir = "../data" 
batch_size = 64

# Conjunto de datos MNIST
train_data = torchvision.datasets.MNIST(
    root=data_dir,
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
eval_data = torchvision.datasets.MNIST(
    root=data_dir,
    train=False,
    download=True,
    transform=transforms.ToTensor()
)


# Conjunto de entrenamiento
train_indices_1_2 = [i for i in range(len(train_data)) if train_data.targets[i] == 1 or train_data.targets[i] == 2]
train_indices_5_7 = [i for i in range(len(train_data)) if train_data.targets[i] == 7 or train_data.targets[i] == 5]

train_1_2 = torch.utils.data.Subset(train_data, train_indices_1_2)
train_5_7 = torch.utils.data.Subset(train_data, train_indices_5_7)

# Conjunto de evaluación
eval_indices_1_2_5_7 = [i for i in range(len(eval_data)) if (eval_data.targets[i] == 1 or eval_data.targets[i] == 2 or eval_data.targets[i] == 7 or eval_data.targets[i] == 5)]

eval_1_2_5_7 = torch.utils.data.Subset(eval_data, eval_indices_1_2_5_7)

# Dataloader
train_1_2_dataloader = DataLoader(train_1_2, batch_size, shuffle=True)
train_5_7_dataloader = DataLoader(train_5_7, batch_size, shuffle=True)
eval_1_2_5_7_dataloader = DataLoader(eval_1_2_5_7, batch_size, shuffle=True)
print("Se cargaron los datos correctamente")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)

if load_model: model.load_state_dict(torch.load(f'etapa_1.pth')) ; print("Se cargo un modelo")
else: print("No se cargo ningun modelo")
learning_rate = 1e-3
# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    tarjet_prediction = []  # Lista para almacenar las etiquetas reales y predicciones

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            # Guardar la etiqueta real y la predicción en la lista de tuplas
            tarjet_prediction.extend(list(zip(y.cpu().numpy(), pred.argmax(1).cpu().numpy())))

    test_loss /= num_batches
    correct /= size
    log_accuracy_loss.append((100*correct, test_loss))
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    with open(f'logs/epoch_{i}_fase_{t}.txt', 'w') as archivo:
            # Escribe el valor de la variable en el archivo
            archivo.write(str(tarjet_prediction))
    print(f'El valor prediciones se ha guardado en el archivo.txt')
    


epochs = 16
data_1 = train_1_2_dataloader
data_2 = train_5_7_dataloader
_eval = eval_1_2_5_7_dataloader
log_accuracy_loss = []
for t in range(total_fase):
    print("*" *250)
    print(f"Etapa {t}")
    print("*" *250)

    for i in range(epochs):
        if t == 0:
            print(f"Epoch {i+1}\n-------------------------------")
            train_loop(data_1, model, loss_fn, optimizer)
            test_loop(_eval, model, loss_fn)
        elif t == 1:
            print(f"Epoch {i+1}\n-------------------------------")
            train_loop(data_2, model, loss_fn, optimizer)
            test_loop(_eval, model, loss_fn)

        if save_model: torch.save(model.state_dict(), f'Fase_{t}.pth')

with open(f'logs/log_accuracy_loss.txt', 'w') as archivo:
        # Escribe el valor de la variable en el archivo
        archivo.write(str(log_accuracy_loss))
print(f'Se guardo correctamente el acurracy')

print("Done!")