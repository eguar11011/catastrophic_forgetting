import torchvision
from torchvision import transforms
import torch
from torch import nn
from torch.utils.data import DataLoader

import os
import torch.nn.functional as F


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
 
train_dataloader = DataLoader(train_data, batch_size, shuffle=True)
train_0_to_4_dataloader = DataLoader(train_0_to_4, batch_size, shuffle=True)
train_5_to_9_dataloader = DataLoader(train_5_to_9, batch_size, shuffle=True)

eval_0_to_4_dataloader = DataLoader(eval_0_to_4, batch_size, shuffle=True)
eval_5_to_9_dataloader = DataLoader(eval_5_to_9, batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_data, batch_size=10000, shuffle=True)
print("Se cargaron los datos correctamente")
#-----------------------------------------------------------------------------------------


class WeightAligning_NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_old_classes):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_old_classes = num_old_classes

        self.flatten = nn.Flatten()
        # Definir las capas lineales
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Propagación hacia adelante en la red
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        if self.output_size != self.num_old_classes: 
            # Separar los pesos de las capas lineales para clases antiguas y nuevas
            weights_old = self.fc3.weight[:self.num_old_classes, :]
            weights_new = self.fc3.weight[self.num_old_classes:, :]
            weights_old.requires_grad_(True) ;weights_new.requires_grad_(True)

            # Calcular las normas de los vectores de peso para clases antiguas y nuevas
            norm_old = torch.norm(weights_old, dim=1)
            norm_new = torch.norm(weights_new, dim=1)
            
            # Calcular el factor de normalización γ
            gamma = torch.mean(norm_old) / torch.mean(norm_new)
            # Aplicar el alineamiento de pesos (Weight Aligning)
            weights_new_aligned = gamma * weights_new
            weights_new_aligned.requires_grad_(True)

            # Crear una copia del tensor de pesos de fc3
            new_fc3_weight = self.fc3.weight.clone()

            # Asignar los pesos alineados a la parte correspondiente del tensor de pesos de fc3
            new_fc3_weight[self.num_old_classes:, :] = weights_new_aligned

            # Asignar el tensor de pesos modificado a fc3
            self.fc3.weight = nn.Parameter(new_fc3_weight)
            
            
            # Aplicar los pesos alineados para calcular la salida
            logits = self.fc3(x)

        else: logits = self.fc3(x)
        
        return logits

# Definir las dimensiones de la red y el número de clases antiguas
input_size = 28*28  # Tamaño de entrada (por ejemplo, para imágenes de 28x28 píxeles)
hidden_size = 512  # Tamaño de las capas ocultas
output_size = 5  # Tamaño de salida (ejemplo: 20 clases nuevas)
num_old_classes = 5  # Número de clases antiguas


model = WeightAligning_NeuralNetwork(input_size, hidden_size, output_size, num_old_classes).to(device)



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
        with open(f'logs/epoch_{i}_fase_{t}.txt', 'w') as archivo:
                # Escribe el valor de la variable en el archivo
                archivo.write(str(tarjet_prediction))
        print(f'El valor prediciones se ha guardado en el archivo.txt')
    


epochs_first = 6
epochs_second = 10
data_1 = train_0_to_4_dataloader
data_2 = train_5_to_9_dataloader
data_t = train_dataloader
log_accuracy_loss = []

for t in range(total_fase):
    print("*" *250); print(f"Etapa {t}");  print("*" *250)

    if t == 0:
        for i in range(epochs_first):
            if t == 0:
                print(f"Epoch {i+1}\n-------------------------------")
                train_loop(data_1, model, loss_fn, optimizer)
                test_loop(eval_0_to_4_dataloader, model, loss_fn)

    elif t == 1:

        actual = torch.load("Fase_0.pth")
        #  scaling_factor = 0.01 
        tensor_weight = torch.rand(5,512).to(device)  #*scaling_factor
        tensor_bias = torch.rand(5).to(device)  #*scaling_factor
        actual["fc3.weight"] = torch.cat((actual["fc3.weight"],tensor_weight), dim=0 )
        actual["fc3.bias"] = torch.cat((actual["fc3.bias"],tensor_bias), dim=0 )

        
        output_size = 10  
        model = WeightAligning_NeuralNetwork(input_size, hidden_size, output_size , num_old_classes).to(device)
        model.load_state_dict(actual)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        for i in range(epochs_second):       
            print(f"Epoch {i+1}\n-------------------------------")
            train_loop(data_2, model, loss_fn, optimizer)
            #test_loop(eval_5_to_9_dataloader, model_second, loss_fn, save=False)
            test_loop(eval_dataloader, model, loss_fn)

    if t == 0: torch.save(model.state_dict(), f'Fase_{t}.pth'); print(f"Se guardo el modelo:\n Fase_{t}.pth ")
    if t == 1: torch.save(model.state_dict(), f'Fase_{t}.pth'); print(f"Se guardo el modelo:\n Fase_{t}.pth ") 

with open(f'logs/log_accuracy_loss.txt', 'w') as archivo:
        archivo.write(str(log_accuracy_loss))

print(f'Se guardo correctamente el acurracy')
print("Done!")