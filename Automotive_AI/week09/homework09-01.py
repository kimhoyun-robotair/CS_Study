import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.init as init
import torch.nn.functional as F

########################################
# 0. Device & Env
########################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)
if device.type == 'cuda':
    print(' -> GPU:', torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True  # conv 최적화

torch.manual_seed(777)

########################################
# 1. Hyper-parameters
########################################
lr              = 1e-3
epochs          = 30
batch_size      = 128
use_amp         = True   # 혼합정밀 사용 여부

########################################
# 2. CIFAR-10 Dataset
########################################
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

train_set = dsets.CIFAR10(root='CIFAR10_data/',
                          train=True,
                          transform=transform_train,
                          download=True)

test_set  = dsets.CIFAR10(root='CIFAR10_data/',
                          train=False,
                          transform=transform_test,
                          download=True)

train_loader = DataLoader(train_set,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=2,
                          pin_memory=True)

test_loader  = DataLoader(test_set,
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
            torch.nn.Conv2d(3, 64, 3, 1, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)           # 32→16
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, 1, 1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)           # 16→8
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 3, 1, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)           # 8→4
        )
        self.fc1 = torch.nn.Linear(4 * 4 * 256, 512)
        self.fc2 = torch.nn.Linear(512, 10)

        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.fc2(x)   # logits

model = CIFAR10_CNN().to(device)

########################################
# 4. Loss & Optimizer
########################################
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

########################################
# 5. Train
########################################
print('Training...')
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0

    for X, Y in train_loader:
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(X)
            loss   = criterion(logits, Y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f'[Epoch {epoch:2d}/{epochs}] loss = {avg_loss:.4f}')

print('Training finished!')

########################################
# 6. Test Accuracy
########################################
model.eval()
correct = 0
total   = 0
with torch.no_grad():
    for X, Y in test_loader:
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(X)

        _, pred = torch.max(logits, 1)
        total   += Y.size(0)
        correct += (pred == Y).sum().item()

print(f'Accuracy on CIFAR-10 test images: {100 * correct / total:.2f}%')