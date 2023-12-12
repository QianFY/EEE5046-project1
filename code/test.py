
from dataloader import *
from model import *
from torch.cuda.amp import autocast as autocast
from medpy.metric import binary
import torch
domain_list=['../data/FAZ/Domain1','../data/FAZ/Domain2','../data/FAZ/Domain3','../data/FAZ/Domain4','../data/FAZ/Domain5']
domain_list=domain_list
input_folderL=[os.path.join(domain_list[i],'test/imgs') for i in range(len(domain_list))]
label_folderL=[os.path.join(domain_list[i],'test/mask') for i in range(len(domain_list))]
pred_folderL=[os.path.join(domain_list[i],'test/pred') for i in range(len(domain_list))]

#label_folder='./FAZ/Domain1/test/mask'
#input_folder='./FAZ/Domain1/test/imgs'
#pred_folder='./FAZ/Domain1/test/pred'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


save_path = './checkpoint/domain1_0.7.pth'
model_state = torch.load(save_path)

model = UNet(in_channels=3, out_channels=1)
model.load_state_dict(model_state)
model.eval() 
model.to(device,dtype=torch.float32)



with torch.no_grad():
    
    # one domain at a time
    for idx in range(len(domain_list)):
        dic_record=[]
        hd95_record=[]
        assd_record=[]

        test_data_loader = get_test_Dataloader(input_folder=input_folderL[idx], label_folder=label_folderL[idx], batch_size=4)

        
        for index, (inputs, labels) in enumerate(test_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
           
            outputs = model(inputs)

            diceScore = dice_coefficient(outputs, labels)
            dic_record.append(diceScore.cpu())

            hdScore=binary.hd95(outputs.cpu().numpy(), labels.cpu().numpy())
            hd95_record.append(hdScore)

            assdScore=binary.assd(outputs.cpu().numpy(), labels.cpu().numpy())
            assd_record.append(assdScore)

            # show_result(inputs, outputs, labels)

        dic_arr=np.array(dic_record)
        hd95_arr=np.array(hd95_record)
        assd_arr=np.array(assd_record)

       
        print(f"model:{save_path}") 
        print(f"diceScore of domain{domain_list[idx]} is {np.mean(dic_arr):.4f} +- {np.std(dic_arr):.4f}")
        print(f"hd95Score of domain{domain_list[idx]} is {np.mean(hd95_arr):.4f} +- {np.std(hd95_arr):.4f}")
        print(f"assdScore of domain{domain_list[idx]} is {np.mean(assd_arr):.4f} +- {np.std(assd_arr):.4f}\n")

        torch.cuda.empty_cache()


