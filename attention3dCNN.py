import torch
import torch.nn as nn
import torch.optim as optim
from step3_data_loader_3d import load_data

torch.manual_seed(42)

X, y = load_data()

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

train_size = int(0.8 * len(X))
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

class Attention3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.sigmoid(self.conv(x))
        return x * attn

class AttentionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv3d(2, 16, 3, padding=1)
        self.attn1 = Attention3D(16)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = nn.Conv3d(16, 32, 3, padding=1)
        self.attn2 = Attention3D(32)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(32, 64, 3, padding=1)
        self.attn3 = Attention3D(64)
        self.pool3 = nn.MaxPool3d(2)

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.pool1(self.attn1(torch.relu(self.conv1(x))))
        x = self.pool2(self.attn2(torch.relu(self.conv2(x))))
        x = self.pool3(self.attn3(torch.relu(self.conv3(x))))

        x = x.view(x.size(0), -1)
        return self.fc(x)

model = AttentionCNN()

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