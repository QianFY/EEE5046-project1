from dataloader import *
from model import *

str_ratio = '0.7'
input_folders = ['../data/FAZ/Domain1/train/imgs']
label_folder = '../data/FAZ/Domain1/train/mask'
val_input_folders = ['../data/FAZ/Domain1/test/imgs']
val_label_folder = '../data/FAZ/Domain1/test/mask'

for i in range(2, 6):
    input_folder = '../data/FAZ/Domain1/train/imgs_ratio_' + str_ratio + '/toDomain' + str(i)
    val_input_folder = '../data/FAZ/Domain1/test/imgs_ratio_' + str_ratio + '/toDomain' + str(i)
    input_folders.append(input_folder)
    val_input_folders.append(val_input_folder)

train_data_loader = get_combined_Dataloader(input_folders=input_folders, 
                                  label_folder=label_folder, 
                                  batch_size=4)
val_data_loader = get_combined_Dataloader(input_folders=val_input_folders, 
                                  label_folder=val_label_folder, 
                                  batch_size=4)

# device = torch.device("mps")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = UNet(in_channels=3, out_channels=1)
model.to(device,dtype=torch.float32)

optimizer = get_optimiazer(model)
loss_fn = dice_loss
num_epochs = 20
save_path = f'./checkpoint/domain1_{str_ratio}.pth'

trainer_w_val(train_loader=train_data_loader,
        val_loader=val_data_loader,
        model=model, optimizer=optimizer, loss_fn=loss_fn, 
        num_epochs=num_epochs,
        device=device, save_path=save_path)
