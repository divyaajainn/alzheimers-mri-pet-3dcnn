import torch
import torch.nn as nn
import torch.optim as optim
from step3_data_loader_3d import load_data

torch.manual_seed(42)

X, y = load_data()
X = torch.tensor(X, dtype=torch.float32)

mri = X[:, 0:1]
pet = X[:, 1:2]

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 1, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 16 * 64 * 64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

G = Generator()
D = Discriminator()

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=0.001)
optimizer_D = optim.Adam(D.parameters(), lr=0.001)

batch_size = len(X)

real_label = torch.ones((batch_size, 1)) * 0.9
fake_label = torch.zeros((batch_size, 1))

for epoch in range(5):
    D_real = D(pet)
    loss_real = criterion(D_real, real_label)

    fake_pet = G(mri)
    D_fake = D(fake_pet.detach())
    loss_fake = criterion(D_fake, fake_label)

    loss_D = loss_real + loss_fake

    optimizer_D.zero_grad()
    loss_D.backward()
    optimizer_D.step()

    D_fake = D(fake_pet)
    loss_G = criterion(D_fake, real_label) + 0.1 * torch.mean((fake_pet - pet) ** 2)

    optimizer_G.zero_grad()
    loss_G.backward()
    optimizer_G.step()

    print(f"Epoch {epoch+1}, D Loss: {loss_D.item():.4f}, G Loss: {loss_G.item():.4f}")