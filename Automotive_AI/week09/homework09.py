# CIFAR-10 CNN with PyTorch
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.init as init
import torch.nn.functional as F

# reproducibility
torch.manual_seed(777)

########################################
# 1. Hyper-parameters
########################################
learning_rate   = 0.001
training_epochs = 30
batch_size      = 128
device          = 'cuda' if torch.cuda.is_available() else 'cpu'

########################################
# 2. CIFAR-10 Dataset
########################################
# 평균·표준편차는 CIFAR-10 통계값
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

cifar10_train = dsets.CIFAR10(root='CIFAR10_data/',
                              train=True,
                              transform=transform_train,
                              download=True)

cifar10_test  = dsets.CIFAR10(root='CIFAR10_data/',
                              train=False,
                              transform=transform_test,
                              download=True)

train_loader = DataLoader(dataset=cifar10_train,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=2,
                          pin_memory=True)

test_loader  = DataLoader(dataset=cifar10_test,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=2,
                          pin_memory=True)

########################################
# 3. CNN Model
########################################
class CIFAR10_CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)  # 32→16
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)  # 16→8
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)  # 8→4
        )

        self.fc1 = torch.nn.Linear(4 * 4 * 256, 512, bias=True)
        self.fc2 = torch.nn.Linear(512, 10, bias=True)

        # Xavier/He 초기화 (Conv는 PyTorch 기본이 He-normal)
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)   # (N,  64,16,16)
        out = self.layer2(out) # (N, 128, 8, 8)
        out = self.layer3(out) # (N, 256, 4, 4)
        out = out.view(out.size(0), -1)  # Flatten
        out = F.relu(self.fc1(out))
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.fc2(out)    # logits
        return out

model = CIFAR10_CNN().to(device)

########################################
# 4. Loss & Optimizer
########################################
criterion  = torch.nn.CrossEntropyLoss()
optimizer  = torch.optim.Adam(model.parameters(), lr=learning_rate)

########################################
# 5. Train
########################################
print('Learning started. It may take a while...')
for epoch in range(1, training_epochs + 1):
    model.train()
    avg_loss = 0

    for X, Y in train_loader:
        X, Y = X.to(device), Y.to(device)

        optimizer.zero_grad()
        logits = model(X)              # hypothesis
        loss   = criterion(logits, Y)  # cost
        loss.backward()
        optimizer.step()

        avg_loss += loss.item() / len(train_loader)

    print(f'[Epoch {epoch:2d}/{training_epochs}] loss = {avg_loss:.4f}')

print('Learning Finished!')

########################################
# 6. Test Accuracy
########################################
model.eval()
correct = 0
total   = 0
with torch.no_grad():
    for X, Y in test_loader:
        X, Y = X.to(device), Y.to(device)
        logits = model(X)
        _, predicted = torch.max(logits, 1)
        total   += Y.size(0)
        correct += (predicted == Y).sum().item()

print(f'Accuracy on CIFAR-10 test images: {100 * correct / total:.2f}%')
