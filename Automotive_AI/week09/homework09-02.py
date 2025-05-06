import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

########################################
# 0. Device & Env
########################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)
if device.type == 'cuda':
    print(' ->', torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True

torch.manual_seed(777)

########################################
# 1. Hyper-parameters
########################################
lr         = 1e-3
epochs     = 30
batch_size = 128
use_amp    = True        # 자동 혼합정밀

def init_weights_xavier(m):
  if isinstance(m, nn.Linear):
      nn.init.xavier_uniform_(m.weight)  # Xavier-Uniform
      if m.bias is not None:
          nn.init.zeros_(m.bias)        

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

train_set = dsets.CIFAR10('CIFAR10_data', True,  transform_train, download=True)
test_set  = dsets.CIFAR10('CIFAR10_data', False, transform_test,  download=True)

train_loader = DataLoader(train_set, batch_size, True,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size, False, num_workers=2, pin_memory=True)

########################################
# 3. MLP Model
########################################
class CIFAR10_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32*32*3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 10)

        # 👇 전체 레이어를 순회하며 Xavier 초기화 적용
        self.apply(init_weights_xavier)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 3072차원
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, 0.5, self.training)
        return self.out(x)         # logits

model = CIFAR10_MLP().to(device)

########################################
# 4. Loss & Optimizer
########################################
criterion = nn.CrossEntropyLoss()
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
    print(f'[Epoch {epoch:2d}/{epochs}] loss = {running_loss/len(train_loader):.4f}')

print('Training finished!')

########################################
# 6. Test Accuracy
########################################
model.eval()
correct = total = 0
with torch.no_grad():
    for X, Y in test_loader:
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(X)
        pred = logits.argmax(1)
        total   += Y.size(0)
        correct += (pred == Y).sum().item()

print(f'Accuracy on CIFAR-10 test images: {100*correct/total:.2f}%')