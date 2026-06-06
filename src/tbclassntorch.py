#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score


# In[6]:


bb = np.load('/kaggle/working/b_phipt.npy').reshape((-1, 64, 64, 1))
tt = np.load('/kaggle/working/t_phipt.npy').reshape((-1, 64, 64, 1))


# In[7]:


X = np.concatenate([bb, tt], axis=0)

#normalization
X_log  = np.log1p(X)
X_sqrt = np.sqrt(X)
X_norm =  X / (np.sum(X, axis=(1, 2), keepdims=True) + 1e-8)  # +epsilon to avoid div-by-zero for empty events

Y = np.concatenate([
    np.ones(len(bb)),
    np.zeros(len(tt))
]).astype(np.int64)


# In[8]:


X_inputs = {
    'Raw':  X,
    'Sqrt': X_sqrt,
    'Log':  X_log,
    'L1':   X_norm
}


# In[9]:


#Dataset Class

class TBDataset(Dataset):

    def __init__(self, X, y):

        X = np.transpose(X, (0, 3, 1, 2))
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# In[10]:


#cnn arch

class TBCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=0
            ),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 2048),
            nn.ReLU(),

            nn.Linear(2048, 512),
            nn.ReLU(),

            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# In[11]:


#create model

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# In[14]:


for name, X in X_inputs.items():

    print(f"\nTraining with {name}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y,
        test_size=0.2,
        stratify=Y,
        random_state=42
    )

    train_dataset = TBDataset(X_train, y_train)
    test_dataset = TBDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    # NEW MODEL FOR THIS NORMALIZATION
    model = TBCNN().to(device)

    # Optional
    model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    num_epochs = 30

    for epoch in range(num_epochs):

        model.train()

        running_loss = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

    # Evaluate
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
    
        for images, labels in test_loader:
    
            images = images.to(device)
            labels = labels.to(device)
    
            outputs = model(images)
    
            preds = torch.argmax(outputs, dim=1)
    
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    
    print(f"{name}: Test Accuracy = {accuracy:.4f}")
    
    torch.save(
        model.state_dict(),
        f"{name}_model.pth"
    )
    
    print(f"{name} model saved!")


# In[ ]:




