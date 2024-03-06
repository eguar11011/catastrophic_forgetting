import torchvision
from torchvision import transforms
import torch
from torch import nn
import torch
from torch.utils.data import DataLoader, Subset
import os



total_fase = 2 
save_model = True
load_model = False
device = "cuda"
data_dir = "../data" 
batch_size = 32
#------------------------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------------------------
num_classes = 10  
class_dataloaders = []

# Itera sobre cada clase y crea un DataLoader para esa clase
for class_idx in range(num_classes):
    # Obtén los índices para la clase actual
    class_indices = [i for i in range(len(train_data)) if train_data.targets[i] == class_idx]
    
    # Subconjunto de datos para la clase actual
    class_subset = Subset(train_data, class_indices)
    
    # DataLoader para la clase actual
    class_dataloader = DataLoader(class_subset, batch_size=batch_size, shuffle=True)
    
    # Agrega el DataLoader al listado
    class_dataloaders.append(class_dataloader)

eval_dataloader_all = DataLoader(eval_data, batch_size=10000, shuffle=True)

#------------------------------------------------------------------------------------------------

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

if load_model: model.load_state_dict(torch.load(f'Fase_1.pth')) ; print("Se cargo un modelo")
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


def test_loop(dataloader, model, loss_fn, t):
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
    with open(f'logs/epoch_{i}_CC_{t}.txt', 'w') as archivo:
            # Escribe el valor de la variable en el archivo
            archivo.write(str(tarjet_prediction))
    print(f'El valor prediciones se ha guardado en el archivo.txt')
    

epochs = 10

log_accuracy_loss = []
for t, train_dataloader in enumerate(class_dataloaders):
    print("*" *200)
    print(f"Etapa {t}")

    for i in range(epochs):
            print(f"Epoch {i+1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer)
            test_loop(eval_dataloader_all, model, loss_fn, t)


    if save_model: torch.save(model.state_dict(), f'Fase_{t}.pth'); print(f"Se guardo el modelo en la Fase:{t}")


with open(f'logs/log_accuracy_loss.txt', 'w') as archivo:
        # Escribe el valor de la variable en el archivo
        archivo.write(str(log_accuracy_loss))
print(f'Se guardo correctamente el acurracy')

print("Done!")