import torchvision
from torchvision import transforms
import torch
from torch import nn
from torch.utils.data import DataLoader

import os
import torch.nn.functional as F
import torchvision.models as models

total_fase = 1
save_model = True
load_model = False
device = "cuda"
data_dir = "../data" 
batch_size = 10

print("GPU activa:", torch.cuda.is_available(), "\nCantidad de GPs", torch.cuda.device_count())
#------------------------------------------------------------------------------------------------

# Transformaciones para preprocesar los datos
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Descarga y carga el conjunto de datos CIFAR-10
train_data = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, num_workers=2)

eval_data = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
eval_dataloader = torch.utils.data.DataLoader(eval_data, batch_size=10000, shuffle=False, num_workers=2)


#-----------------------------------------------------------------------------------------


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()


        resnet = models.resnet18(weights= None)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Obtener todas las capas excepto la capa de clasificación

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        
    def forward(self, x):
        # Propagación hacia adelante en la red
        x = self.features(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)



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


def test_loop(dataloader, model, loss_fn, save=True):
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
    

    # Definir la ruta del archivo
    directorio= 'logs'
    if not os.path.exists(directorio):
        os.makedirs(directorio)
    # Guardar tarjet_predictions
    if save:    
        with open(f'logs/epoch_{i}_fase_{t}_cifar10:completo.txt', 'w') as archivo:
                # Escribe el valor de la variable en el archivo
                archivo.write(str(tarjet_prediction))
        print(f'El valor prediciones se ha guardado en el archivo.txt')
    

epochs = 10

data_1 = train_dataloader
# data_2 = train_5_to_9_dataloader
log_accuracy_loss = []
for t in range(total_fase):
    print("*" *200)
    print(f"Etapa {t}")


    for i in range(epochs):
        if t == 0:
            print(f"Epoch {i+1}\n-------------------------------")
            train_loop(data_1, model, loss_fn, optimizer)
            test_loop(eval_dataloader, model, loss_fn, t)
        elif t == 1:
            print(f"Epoch {i+1}\n-------------------------------")
            train_loop(data_2, model, loss_fn, optimizer)
            test_loop(eval_dataloader, model, loss_fn, t)

    if save_model: torch.save(model.state_dict(), f'Fase_{t}_cifar10_completo.pth'); print(f"Se guardo el modelo en la Fase:{t}_cifar10_completo")

with open(f'logs/log_accuracy_loss_cifar10_completo.txt', 'w') as archivo:
        archivo.write(str(log_accuracy_loss))

print(f'Se guardo correctamente el acurracy')
print("Done!")