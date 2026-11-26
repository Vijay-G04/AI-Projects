# Handwritten Digit Recognizer (MNIST CNN)

**Level:** Beginner | **Estimated time:** 2 to 3 days | **Domain:** Deep learning (computer vision)

## What this project shows
Building and training a convolutional neural network from scratch, training loops, and evaluation.

## Dataset
**MNIST**: 60,000 training and 10,000 test grayscale images (28x28) of digits. Downloaded automatically by `torchvision.datasets.MNIST`.

## Tech stack
Python, PyTorch, torchvision, matplotlib, Gradio (optional drawing demo).

## Suggested folder structure
```
mnist-cnn/
├── data/                # auto-downloaded (gitignored)
├── src/
│   ├── model.py         # CNN definition
│   ├── train.py
│   └── predict.py
├── models/
│   └── mnist_cnn.pt
├── app.py               # optional Gradio demo
├── requirements.txt
└── README.md
```

## Step-by-step plan
1. Load MNIST with `torchvision`, normalize, and wrap in `DataLoader`s.
2. Visualize a grid of sample digits.
3. Define a small CNN: two conv layers, max pooling, dropout, two fully connected layers.
4. Write a training loop with cross-entropy loss and the Adam optimizer. Track loss per epoch.
5. Evaluate test accuracy and plot a confusion matrix. Look at the images the model gets wrong.
6. Save the weights and write `predict.py` that classifies a single image.
7. Optional: a Gradio sketchpad where users draw a digit and see the prediction.

## Starter code (`src/train.py`)
Treat this as a working skeleton. Run it, understand each line, then improve it.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_loader = DataLoader(datasets.MNIST("data", train=True, download=True, transform=tfm),
                          batch_size=64, shuffle=True)
test_loader = DataLoader(datasets.MNIST("data", train=False, transform=tfm), batch_size=1000)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)    # 28 -> 26
        self.conv2 = nn.Conv2d(32, 64, 3)   # 26 -> 24, pool -> 12
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.drop(torch.flatten(x, 1))
        return self.fc2(F.relu(self.fc1(x)))


model = CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
    print(f"Epoch {epoch + 1}: test accuracy = {correct / len(test_loader.dataset):.4f}")

torch.save(model.state_dict(), "models/mnist_cnn.pt")
```

## What to put in your README
- **Problem statement:** one paragraph on what you are predicting or building, and why it matters.
- **Dataset:** name, source link, size, and any cleaning you did.
- **Approach:** models tried and why.
- **Results:**
- Test accuracy and a training-loss curve.
- A confusion matrix and a few misclassified examples.
- A screenshot or GIF of the demo app, if you build one.
- **How to run:** `pip install -r requirements.txt`, then the exact command(s).
- **Limitations and next steps:** be honest about what the model gets wrong.

## Ways to extend it (after the basic version works)
- Add data augmentation (small rotations and shifts).
- Compare against a plain MLP to show why CNNs help.
- Try Fashion-MNIST or EMNIST.
- Export to ONNX.

## Resume line
Implemented a CNN in PyTorch for MNIST digit recognition reaching X% test accuracy; built an interactive drawing demo with Gradio.

## Before you push to GitHub
- Add a `.gitignore` (ignore `data/`, `__pycache__/`, `.venv/`, large model files).
- Do not commit large datasets; link to them and add a `download_data` note in the README.
- Never commit API keys. Use environment variables.
- Commit in small steps with clear messages (for example "Add baseline model", "Add evaluation plots").
