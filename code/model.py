import torch
import torch.nn as nn
import numpy as np
from torch import optim

from torch.utils.tensorboard import SummaryWriter
# iter

from tqdm import tqdm

# ============================
# Model
# ============================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512, 1024]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Encoder
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Decoder
        for feature in reversed(features[:-1]):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for i, down in enumerate(self.downs):
            x = down(x)
            skip_connections.append(x)
            if i != len(self.downs)-1:
                x = nn.MaxPool2d(kernel_size=2)(x)

        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skip_connections[i//2+1]
            if x.shape != skip.shape:
                x = nn.functional.pad(x, (0, skip.shape[3]-x.shape[3], 0, skip.shape[2]-x.shape[2]))
            x = torch.cat((skip, x), dim=1)
            x = self.ups[i+1](x)

        x = self.final_conv(x)
        x = torch.sigmoid(x)
        # x = torch.where(x>0.5, torch.tensor(1), torch.tensor(0))
        return x

# ============================
# Loss
# ============================

def dice_coefficient(y_pred, y_true,eps=1e-5):
    y_pred = y_pred.contiguous().view(-1)
    y_true = y_true.contiguous().view(-1)
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum()
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice

def dice_loss(y_pred, y_true):
    return 1 - dice_coefficient(y_pred, y_true)

def get_optimiazer(model, learning_rate=0.001):
    return optim.Adam(model.parameters(), lr=learning_rate)


# ============================
# Trainer
# ============================

#with validation
def trainer_w_val(train_loader, val_loader,model, optimizer, loss_fn, num_epochs, device, save_path: str):
    if not save_path.endswith('pth'):
        raise Exception('Invalid save path')
    
    writer=SummaryWriter('./logs')
    best_loss = 1
    step=0
    for epoch in range(num_epochs):
        loss_record=[]

        loop = tqdm(enumerate(train_loader), ncols=100, total=len(train_loader))
        loop.set_description(f'Epoch [{epoch+1}/{num_epochs}]')

        for i, (inputs, labels) in loop:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # backward
            loss.backward()
            optimizer.step()
            step+=1
            
            loss_record.append(loss.detach().item())

            loop.set_postfix_str(f'loss/train={loss.detach().item():.6f}')
        

        mean_train_loss=sum(loss_record)/len(loss_record)
        writer.add_scalar('Loss/train',mean_train_loss,step)

       # validation
        model.eval() # Set your model to evaluation mode.
        loss_record = []
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                
                pred = model(x)
                loss = loss_fn(pred, y)
                loss_record.append(loss.item())

        mean_val_loss = sum(loss_record)/len(loss_record)
        print(f'Epoch [{epoch+1}/{num_epochs}]: Mean Train loss: {mean_train_loss:.4f}, Mean Valid loss: {mean_val_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_val_loss, step)


        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            torch.save(model.state_dict(), save_path)
            print(f'Save checkpoint in Epoch[{epoch+1}/{num_epochs}] with loss{best_loss}')

#without validation
def trainer_wo_val(train_loader, val_loader,model, optimizer, loss_fn, num_epochs, device, save_path: str):
    if not save_path.endswith('pth'):
        raise Exception('Invalid save path')
    
    writer=SummaryWriter('./logs')
    best_loss = 1
    step=0
    for epoch in range(num_epochs):
        loss_record=[]

        loop = tqdm(enumerate(train_loader), ncols=100, total=len(train_loader))
        loop.set_description(f'Epoch [{epoch+1}/{num_epochs}]')

        for i, (inputs, labels) in loop:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # backward
            loss.backward()
            optimizer.step()
            step+=1
            
            loss_record.append(loss.detach().item())

            loop.set_postfix_str(f'loss/train={loss.detach().item():.6f}')
        

        mean_train_loss=sum(loss_record)/len(loss_record)
        writer.add_scalar('Loss/train',mean_train_loss,step)

        # save model that performs best only on the train dataset 
        if mean_train_loss < best_loss:
            
            best_loss = mean_train_loss
            torch.save(model.state_dict(), save_path)
            print(f'Save checkpoint in Epoch[{epoch+1}/{num_epochs}] with loss{best_loss}')



