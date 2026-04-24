import torch
import torch.nn as nn
import torch.optim as optim
from step3_data_loader_3d import load_data
from step6_gan_experiment import Generator

torch.manual_seed(42)

X, y = load_data()
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

mri = X[:, 0:1]
pet = X[:, 1:2]

G = Generator()
G.eval()

fake_pet = G(mri).detach()

X_new = torch.cat([mri, pet, fake_pet], dim=1)

train_size = int(0.8 * len(X_new))
X_train, X_val = X_new[:train_size], X_new[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

class FinalModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = FinalModel()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_val)
        acc = (preds.argmax(1) == y_val).float().mean()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Val Acc: {acc:.4f}")