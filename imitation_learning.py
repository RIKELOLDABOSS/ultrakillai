import pickle
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from stable_baselines3 import PPO
from environment import UltrakillEnv

class UltrakillDataset(Dataset):
    def __init__(self, recordings_dir):
        self.samples = []
        for file in os.listdir(recordings_dir):
            if file.endswith(".pkl"):
                with open(os.path.join(recordings_dir, file), 'rb') as f:
                    data = pickle.load(f)
                    self.samples.extend(data)

        self.key_list = [
            'w', 'a', 's', 'd', 'space', 'shift', 'ctrl',
            'e', 'q', 'r', '1', '2', '3', '4', '5', '6'
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frame = sample['frame'].transpose(2, 0, 1) / 255.0 # CHW and normalized

        # Convert inputs to target actions
        inputs = sample['inputs']
        keys_pressed = [1 if k in inputs['keys'] else 0 for k in self.key_list]
        clicks = [1 if c else 0 for c in inputs['mouse_click']]
        move = inputs['mouse_move']

        # We'll return them as a single vector or a dict
        target = np.array(keys_pressed + clicks + list(move), dtype=np.float32)

        return torch.tensor(frame, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

def train_imitation():
    if not os.path.exists('recordings'):
        print("No recordings found. Run teacher.py first.")
        return

    dataset = UltrakillDataset('recordings')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Simple Behavioral Cloning model
    # Note: In a real scenario, we'd want to map this directly to the PPO model's weights
    # For now, we'll just demonstrate the training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We can use the same architecture as SB3's NatureCNN if we want consistency
    class SimplePolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 11 * 11, 512), nn.ReLU(),
                nn.Linear(512, 16 + 2 + 2) # Keys + Clicks + Mouse
            )
        def forward(self, x):
            return self.cnn(x)

    model = SimplePolicy().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    print("Training imitation model...")
    for epoch in range(10):
        total_loss = 0
        for frames, targets in dataloader:
            frames, targets = frames.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

    torch.save(model.state_dict(), "imitation_model.pth")
    print("Imitation model saved.")

if __name__ == "__main__":
    train_imitation()
