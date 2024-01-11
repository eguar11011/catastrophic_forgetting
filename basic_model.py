import torchvision
from torchvision import transforms

import torch
from torch import nn
from torch.utils.data import DataLoader
import os

from utils import save_tarjet_predictions

num_fase = "completo"
save_model = True
load_model = False
device = "cuda"






data_dir = "data" 

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


# Separar las clases 0-4 y 5-9 para los conjuntos de entrenamiento y evaluación

# Conjunto de entrenamiento
train_indices_0_to_4 = [i for i in range(len(train_data)) if train_data.targets[i] < 5]
train_indices_5_to_9 = [i for i in range(len(train_data)) if train_data.targets[i] >= 5]

train_0_to_4 = torch.utils.data.Subset(train_data, train_indices_0_to_4)
train_5_to_9 = torch.utils.data.Subset(train_data, train_indices_5_to_9)

# Conjunto de evaluación
eval_indices_0_to_4 = [i for i in range(len(eval_data)) if eval_data.targets[i] < 5]
eval_indices_5_to_9 = [i for i in range(len(eval_data)) if eval_data.targets[i] >= 5]

eval_0_to_4 = torch.utils.data.Subset(eval_data, eval_indices_0_to_4)
eval_5_to_9 = torch.utils.data.Subset(eval_data, eval_indices_5_to_9)
 
train_dataloader = DataLoader(train_data, batch_size=500, shuffle=True)
train_0_to_4_dataloader = DataLoader(train_0_to_4, batch_size=500, shuffle=True)
train_5_to_9_dataloader = DataLoader(train_5_to_9, batch_size=500, shuffle=True)

eval_0_to_4_dataloader = DataLoader(eval_0_to_4, batch_size=300, shuffle=True)

eval_dataloader = DataLoader(eval_data, batch_size=10000, shuffle=True)
#-----------------------------------------------------------------------------------------
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)

if load_model: model.load_state_dict(torch.load(f'fase_1.pth')) ; print("Se cargo un modelo")
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

    if save_model: torch.save(model.state_dict(), f'fase_{num_fase}.pth')
    
    



def test_loop(dataloader, model, loss_fn,etapa):
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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


    with open(f'epoch_{etapa}_fase_{num_fase}.txt', 'w') as archivo:
            # Escribe el valor de la variable en el archivo
            archivo.write(str(tarjet_prediction))

    print(f'El valor prediciones se ha guardado en el archivo.txt')
    


epochs = 25 
data_1 = train_0_to_4_dataloader
data_2 = train_5_to_9_dataloader
data_t = train_dataloader
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(data_t, model, loss_fn, optimizer)
    test_loop(eval_dataloader, model, loss_fn,t)
    
#torch.save(model.state_dict(), 'fase_2.pth')

print("Done!")