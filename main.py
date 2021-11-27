import os
import copy
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from data import *
from models import EDL


class Experiment:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loss_type = 'digamma'
        assert self.loss_type in ('log', 'digamma', 'mse', 'softmax')

        # Load dataset
        dataset = load_mnist()
        assert set(dataset.keys()) >= {'train', 'val', 'classes'}
        self.train_loader = torch.utils.data.DataLoader(dataset['train'], batch_size=256, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(dataset['val'], batch_size=1024, shuffle=False)

        # Define model
        self.model = EDL(sample_shape=dataset['train'][0][0].shape, num_classes=dataset['classes'], loss_type=self.loss_type)
        self.model = self.model.to(self.device)

        # Define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.1)
        self.epochs = 50

    def train(self, saving_path=None):
        model = self.model
        best_valid_acc = 0.
        best_model_wts = model.state_dict()
        for epoch in range(self.epochs):
            model.train()
            train_loss, correct, num_samples = 0, 0, 0
            for batch, target in self.train_loader:
                batch = batch.to(self.device)
                target = target.to(self.device)
                evidence, loss = model(batch, target, epoch)
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
                train_loss += loss.mean().item() * len(target)
                correct += torch.sum(evidence.argmax(dim=-1).eq(target)).item()
                num_samples += len(target)
            self.scheduler.step()
            train_loss = train_loss / num_samples
            train_acc = correct / num_samples
            valid_acc = self.validate()
            if best_valid_acc < valid_acc:
                best_valid_acc = valid_acc
                best_model_wts = copy.deepcopy(model.state_dict())  # save the best model
            print(f'Epoch {epoch:2d}; train loss {train_loss:.4f}, train acc {train_acc:.4f}; val acc: {valid_acc:.4f}')

        model.load_state_dict(best_model_wts)
        print('Validation Accuracy:', self.validate())
        if saving_path is not None:
            os.makedirs(os.path.dirname(saving_path), exist_ok=True)
            torch.save(model, saving_path)
        return model

    def validate(self, loader=None):
        if loader is None:
            loader = self.valid_loader
        model = self.model
        model.eval()
        with torch.no_grad():
            correct, num_samples = 0, 0
            for batch, target in loader:
                batch = batch.to(self.device)
                target = target.to(self.device)
                evidence, loss = model(batch)
                correct += torch.sum(evidence.argmax(dim=-1).eq(target)).item()
                num_samples += len(target)
        acc = correct / num_samples
        return acc


if __name__ == '__main__':
    t = Experiment()
    t.train()
