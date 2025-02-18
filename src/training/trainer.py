import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import os
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR
import numpy as np


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience
        self.device = t.device("cuda:0" if t.cuda.is_available() else "cpu")


        if cuda:
            self._model = model.to(self.device)
            self._crit = crit.to(self.device)
            
    def save_checkpoint(self, epoch):
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def calculate_accuracy(self, predictions, labels):
        # Calculate accuracy
        accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        return accuracy


    def train_step(self, x, y):
        # Reset the gradients
        self._optim.zero_grad()
        # Propagate through the network
        output = self._model(x)
        # Calculate the loss
        loss = self._crit(output, y)
        # Compute gradient by backward propagation
        loss.backward()
        # Update weights
        self._optim.step()
        return loss.item()



    def val_test_step(self, x, y):
        # Predict and calculate the loss and predictions
        output = self._model(x)
        loss = self._crit(output, y)
        return loss.item() ,output

        
    def train_epoch(self):
        # Training
        total_loss = 0.0
        num_batches = len(self._train_dl)
        self._model.train()
        for batch_idx in tqdm(self._train_dl, desc='training', leave=False):
            x, y = batch_idx
            if self._cuda:
                x,y = x.to(self.device), y.to(self.device)
            loss = self.train_step(x, y)
            total_loss += loss
            del x, y, loss
            t.cuda.empty_cache()

        avg_loss = total_loss / num_batches
        return avg_loss
    
    def val_test(self):
        self._model.eval()
        all_predictions = []
        all_labels = []
        with t.no_grad():
            total_loss = 0.0
            num_batches = len(self._val_test_dl)
            for batch_idx in tqdm(self._val_test_dl, desc='Validation/Test', leave=False):
                x, y = batch_idx
                if self._cuda:
                    x, y = x.to(self.device), y.to(self.device)
                loss, predictions = self.val_test_step(x,y)
                total_loss += loss

                all_predictions.append(predictions.cpu())  # Move to CPU before converting to numpy
                all_labels.append(y.cpu())  # Move to CPU before converting to numpy

                del x, y, loss
                t.cuda.empty_cache()

            avg_loss = total_loss / num_batches
            print(f'Validation/Test Loss: {avg_loss:.4f}')

            # Flatten and concatenate the predictions and labels
            flat_predictions = t.cat(all_predictions, dim=0)
            flat_labels = t.cat(all_labels, dim=0)

            threshold = 0.5
            # Apply the threshold to predictions
            binary_predictions = (flat_predictions > threshold).float()

            # Calculate F1 score for each label separately
            f1 = f1_score(flat_labels, binary_predictions, average='weighted')
            
            print(f'F1 Score: {f1:.4f}')
            accuracy = self.calculate_accuracy(binary_predictions, flat_labels)
            print(f'Accuracy: {accuracy:.4f}')

            return avg_loss,f1

    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0

        train_losses = []
        val_losses = []
        epoch_counter = 0
        best_val_loss = float('inf')
        patience_counter = 0
        scheduler = t.optim.lr_scheduler.MultiStepLR(self._optim, milestones=[10, 20, 30, 40], gamma=0.1)
        # scheduler2 = t.optim.lr_scheduler.MultiStepLR(self._optim, milestones=[60, 80, 100, 130], gamma=0.5)
        while epoch_counter <= epochs:
            print(f'Epoch: {epoch_counter}')
            train_loss = self.train_epoch()
            train_losses.append(train_loss)

            # Validate/test the model
            val_loss,f1 = self.val_test()
            val_losses.append(val_loss)

            if f1 > 0.8:
                self.save_checkpoint(epoch_counter)

            # Check for early stopping
            if self._early_stopping_patience > 0:
                if val_loss <= best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
            
            
            
                else:
                    patience_counter += 1
            
                if patience_counter >= self._early_stopping_patience:
                    print(f'Early stopping after {epoch_counter} epochs without improvement.')
                    break

            epoch_counter += 1
            # Update the learning rate
            scheduler.step()
            # scheduler2.step()
        return train_losses, val_losses

                    
        
        
        
