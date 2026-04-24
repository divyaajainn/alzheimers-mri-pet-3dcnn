import torch
import torch.nn as nn
import torch.optim as optim
from step3_data_loader_3d import load_data

torch.manual_seed(42)

X, y = load_data()
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

perm = torch.randperm(len(X))
X = X[perm]
y = y[perm]

train_size = int(0.8 * len(X))
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

mri_train = X_train[:, 0:1]
pet_train = X_train[:, 1:2]

mri_val = X_val[:, 0:1]
pet_val = X_val[:, 1:2]

class FeatureCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )

    def forward(self, x):
        x = self.net(x)
        return x.view(x.size(0), -1)

class LateFusionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.mri_net = FeatureCNN()
        self.pet_net = FeatureCNN()

        self.fc = nn.Sequential(
            nn.Linear(2 * 64 * 4 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )

    def forward(self, mri, pet):
        f1 = self.mri_net(mri)
        f2 = self.pet_net(pet)
        fused = torch.cat([f1, f2], dim=1)
        return self.fc(fused)

model = LateFusionModel()

weights = torch.tensor([1.0, 1.0, 5.0])
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = optim.Adam(model.parameters(), lr=0.001)

batch_size = 4

for epoch in range(15):
    model.train()

    for i in range(0, len(mri_train), batch_size):
        m_batch = mri_train[i:i+batch_size]
        p_batch = pet_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        outputs = model(m_batch, p_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(mri_val, pet_val)
        acc = (preds.argmax(1) == y_val).float().mean()

    print(f"Epoch {epoch+1}, Val Acc: {acc:.4f}")