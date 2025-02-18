import torch as t
from .data import SolarDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
from ..models.model import ResNet
import pandas as pd
from sklearn.model_selection import train_test_split


# Load the data from the csv file and perform a train-test-split
data_path = "data.csv"
df = pd.read_csv(data_path,sep=';')

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# Data loading for the training and validation set 
train_dataset = SolarDataset(train_df, mode='train')
val_dataset = SolarDataset(val_df, mode='val')

train_loader = t.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = t.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

# Initialize model
resnet_model = ResNet()

# Optimizer
optimizer = t.optim.SGD(resnet_model.parameters(), lr=0.001, momentum= 0.99, weight_decay=1e-5)
# Loss criterion
criterion = t.nn.BCELoss()

# Train
trainer = Trainer(model=resnet_model, crit=criterion, optim=optimizer,
                  train_dl=train_loader, val_test_dl=val_loader, cuda=True,
                  early_stopping_patience=350)

res = trainer.fit(epochs=400)

plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')