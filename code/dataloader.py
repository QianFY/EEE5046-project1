import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import shutil

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ==========================
# DRIVE
# ==========================

# label_folder为None就是获取测试数据，不为None就是获取训练数据
class FAZ_Dataset(Dataset):
    def __init__(self, input_folder, label_folder=None):
        self.input_list = [os.path.join(input_folder, name) for name in sorted(os.listdir(input_folder))]
        self.label_list = None if label_folder is None else [os.path.join(label_folder, name) for name in sorted(os.listdir(label_folder))]

        self.x_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    def __getitem__(self, index):
        input = Image.open(self.input_list[index])
        input = self.x_transform(input)

        if self.label_list is not None:
            label = Image.open(self.label_list[index])
            label = np.asarray(label)[..., 0:1] > 0  # for label image, keep one channel, then convet to a binary image
            label = torch.from_numpy(np.moveaxis(label,-1,0).astype(np.float32))
            return input, label
        else:
            return input
    def __len__(self):
        return len(self.input_list)

# label_folder为None就是获取测试数据，不为None就是获取训练数据
def get_Domain1_Dataloader(input_folder, label_folder=None, batch_size=4):
    dataset = FAZ_Dataset(input_folder=input_folder, label_folder=label_folder)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def get_val_Dataloader(input_folder, label_folder=None, batch_size=4):
    dataset = FAZ_Dataset(input_folder=input_folder, label_folder=label_folder)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def get_test_Dataloader(input_folder, label_folder=None, batch_size=4):
    dataset = FAZ_Dataset(input_folder=input_folder, label_folder=label_folder)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False) 

    return dataloader







# ==========================
# Helper Function
# ==========================
domain_list=['../data/FAZ/Domain1','../data/FAZ/Domain2','../data/FAZ/Domain3','../data/FAZ/Domain4','../data/FAZ/Domain5']
domain_list=domain_list
input_folderL=[os.path.join(domain_list[i],'test/imgs') for i in range(len(domain_list))]
label_folderL=[os.path.join(domain_list[i],'test/mask') for i in range(len(domain_list))]
pred_folderL=[os.path.join(domain_list[i],'test/pred') for i in range(len(domain_list))]
val_img_folderL=[os.path.join(domain_list[i],'valid/imgs') for i in range(len(domain_list))]
val_msk_folderL=[os.path.join(domain_list[i],'valid/mask') for i in range(len(domain_list))]



def get_val_file_list(file_path:str):

    # 打开文件
    with open(file_path, 'r') as file:
        idx=0
        # 初始化二维列表
        file_names = [[] for _ in range(5)]

        for line in file:
            if line.strip():  # 使用strip()方法去除行首和行尾的空白字符
                file_names[idx].append(line.strip())      
            else:
                # 如果为空，进行相应的处理
                idx+=1
    return file_names
# 关闭文件（在使用with语句时不需要手动关闭文件）



def divide_val_from_test():
    source_img_file_list=[[] for _ in range(5)]
    source_msk_file_list=[[] for _ in range(5)]
    val_file_list=get_val_file_list('faz_val.txt')
    
    #get source_img_file_list , source_msk_file_list
    for idx in range(len(domain_list)):
        for item in val_file_list[idx]:
            source_img_file_list[idx].append(os.path.join(domain_list[idx],'test/imgs',item))
            source_msk_file_list[idx].append(os.path.join(domain_list[idx],'test/mask',item))
    #makedirs for validation
    for idx in range(len(domain_list)):
        os.makedirs(val_img_folderL[idx])
        os.makedirs(val_msk_folderL[idx])

        #move data from test to validation
        for s_img_pth, s_msk_pth in zip(source_img_file_list[idx],source_msk_file_list[idx]):
            shutil.move(s_img_pth,val_img_folderL[idx]) 
            shutil.move(s_msk_pth,val_msk_folderL[idx]) 
            # os.remove(s_img_pth) 
            # os.remove(s_msk_pth) 


# divide_val_from_test()


def show_result(inputs: torch.Tensor, outputs: torch.Tensor,labels:torch.Tensor, title: str=None):
    input = np.moveaxis(inputs.detach().cpu().numpy(), 1, -1)
    output = np.moveaxis(outputs.detach().cpu().numpy(), 1, -1)
    label = np.moveaxis(labels.detach().cpu().numpy(), 1, -1)
    
    num = input.shape[0]
    subtitle = ['input', 'label','output']
    plt.figure(figsize=(num*5+2,18))
    if title is not None:
        plt.suptitle(title, fontsize=20)
    for row , data in enumerate([input,label,output]):
        for col, group in enumerate(data):
            plt.subplot(3, num, (row)*num+(col+1))
            plt.title(f'{subtitle[row]} {col+1}')
            plt.imshow(group, cmap='gray')
    plt.show()

