import torch as t
from ..training.trainer import Trainer
from .model import ResNet
import torchvision as tv

epoch = int(354)

model=ResNet()

crit = t.nn.BCELoss()
trainer = Trainer(model, crit)
trainer.restore_checkpoint(epoch)
trainer.save_onnx('checkpoint_{:03d}.onnx'.format(epoch))

