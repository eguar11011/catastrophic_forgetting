import torchvision
from torchvision import transforms
import torch
from torch import nn
from torch.utils.data import DataLoader

import os



total_fase = 1
save_model = True
load_model = False
device = "cuda"
data_dir = "../data" 
batch_size = 32

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from functools import partial

# Conjunto de datos MNIST
train_data = torchvision.datasets.CIFAR10(
    root=data_dir,
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
eval_data = torchvision.datasets.CIFAR10(
    root=data_dir,
    train=False,
    download=True,
    transform=transforms.ToTensor()
)


 
train_dataloader = DataLoader(train_data, batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_data, batch_size=10000, shuffle=True)
print("Se cargaron los datos correctamente")


#-----------------------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):           
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( # input shape (3, 32, 32)
        nn.Conv2d(
        in_channels=3, # input height
        out_channels=16, # n_filters
        kernel_size=5, # filter size
        stride=1, # filter movement/step
        padding=2,
        # if want same width and length of this image after con2d,
        #padding=(kernel_size-1)/2 if stride=1
        ), # output shape (16, 16, 16)
        nn.ReLU(), # activation
        nn.MaxPool2d(kernel_size=2),
        # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential( # input shape (16, 14, 14)
        nn.Conv2d(16, 32, 5, 1, 2), # output shape (32, 8, 8)
        nn.ReLU(), # activation
        nn.MaxPool2d(2), # output shape (32, 7, 7)
        )
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32 *8 *8, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        # flatten the output of conv2 to (batch_size, 32 * 8 * 8)
        output = self.linear_relu_stack(x)
        return output 


model = CNN().to(device)
fase_1 = torch.load('Fase_1.pth')
model.load_state_dict(fase_1)

for idx,layer in enumerate(model.parameters()):
    if idx == len(fase_1) -2: break
    layer.requires_grad = False


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

    with open(f'logs/epoch_{i}_Tuning.txt', 'w') as archivo:
            # Escribe el valor de la variable en el archivo
            archivo.write(str(tarjet_prediction))
    print(f'El valor prediciones se ha guardado en el archivo.txt')
    


epochs = 20
data_t = train_dataloader
log_accuracy_loss = []
for t in range(total_fase):
    print("*" *200)
    print(f"Etapa {t}")

    for i in range(epochs):
        if t == 0:
            print(f"Epoch {i+1}\n-------------------------------")
            train_loop(data_t, model, loss_fn, optimizer)
            test_loop(eval_dataloader, model, loss_fn)

    if save_model: torch.save(model.state_dict(), f'Tuning.pth')


with open(f'logs/log_accuracy_loss_Tuning.txt', 'w') as archivo:
        # Escribe el valor de la variable en el archivo
        archivo.write(str(log_accuracy_loss))
print(f'Se guardo correctamente el acurracy')

print("Done!")