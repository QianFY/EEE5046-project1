from dataloader import *
from model import *

input_folder = './FAZ/Domain1/train/imgs'
label_folder = './FAZ/Domain1/train/mask'
val_input_folder = './FAZ/Domain1/valid/imgs'
val_label_folder = './FAZ/Domain1/valid/mask'
train_data_loader = get_Domain1_Dataloader(input_folder=input_folder, 
                                  label_folder=label_folder, 
                                  batch_size=4)
val_data_loader = get_val_Dataloader(input_folder=val_input_folder, 
                                  label_folder=val_label_folder, 
                                  batch_size=4)

# device = torch.device("mps")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = UNet(in_channels=3, out_channels=1)
model.to(device,dtype=torch.float32)

optimizer = get_optimiazer(model)
loss_fn = dice_loss
num_epochs = 800
save_path = './checkpoint/domain1_w_val5.pth'

trainer_w_val(train_loader=train_data_loader,
        val_loader=val_data_loader,
        model=model, optimizer=optimizer, loss_fn=loss_fn, 
        num_epochs=num_epochs,
        device=device, save_path=save_path)
