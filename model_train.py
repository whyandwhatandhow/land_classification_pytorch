from torchvision.datasets import EuroSAT
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import copy
import time
import pandas as pd
import matplotlib.pyplot as plt
import model
import model_pro
import model_pro_max
import data_process
def train_process(model, train_loader, val_loader, epochs,model_name):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = torch.nn.CrossEntropyLoss()
    best_model_wts=copy.deepcopy(model.state_dict())
    best_acc=0.0
    train_acc_all=[]
    val_acc_all=[]
    train_loss_all=[]
    val_loss_all=[]
    since=time.time()

    for epoch in range(epochs):
        model.train()
        train_acc=0.0
        train_loss=0.0
        val_acc=0.0
        val_loss=0.0
        train_num=0
        val_num=0
        for step,(b_x,b_y) in enumerate(train_loader):
            model.train()
            b_x=b_x.to(device)
            b_y=b_y.to(device)
            output=model(b_x)
            loss=criterion(output,b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()*b_x.size(0)
            pred=torch.argmax(output,dim=1)
            train_acc+=(pred==b_y).sum().item()
            train_num+=b_x.size(0)

        for step,(b_x,b_y) in enumerate(val_loader):
            model.eval()
            b_x=b_x.to(device)
            b_y=b_y.to(device)
            with torch.no_grad():
                output=model(b_x)
                loss=criterion(output,b_y)
            val_loss+=loss.item()*b_x.size(0)
            pred=torch.argmax(output,dim=1)
            val_acc+=(pred==b_y).sum().item()
            val_num+=b_x.size(0)

        train_loss_all.append(train_loss/train_num)
        val_loss_all.append(val_loss/val_num)
        train_acc_all.append(train_acc/train_num)
        val_acc_all.append(val_acc/val_num)
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss/train_num:.4f} "
              f"Train Acc: {train_acc/train_num:.4f} "
              f"Val Loss: {val_loss/val_num:.4f} "
              f"Val Acc: {val_acc/val_num:.4f}")
        
        if val_acc_all[-1]>best_acc:
            best_acc=val_acc_all[-1]
            best_model_wts=copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(),f"best_model_{model_name}.pth")

    time_used=time.time()-since
    model.load_state_dict(best_model_wts)
    
    train_process_df=pd.DataFrame({
        'epoch':range(1,epochs+1),
        'train_acc': train_acc_all,
        'val_acc': val_acc_all,
        'train_loss': train_loss_all,
        'val_loss': val_loss_all,
        'time_used': [time_used]*epochs
    })
    train_process_df["time_used"]=time_used
    train_process_df["best_acc"]=best_acc
    return train_process_df

def matlib_acc_loss(train_process_df,model_name):
    epochs=train_process_df['epoch']
    train_acc=train_process_df['train_acc']
    val_acc=train_process_df['val_acc']
    train_loss=train_process_df['train_loss']
    val_loss=train_process_df['val_loss']

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)  
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1,2,2)  
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')        
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    plt.savefig(f"training_validation_accuracy_loss_{model_name}.png")


if __name__ == "__main__":
    torch.manual_seed(42)
    data_root = './data'
    train_loader, val_loader, _,class_names = data_process.eurosat_data_process(data_root)
    model=model_pro_max.get_resnet50(num_classes=len(class_names), in_channels=3)
    model_name="model_pro_max"
    train_process_df=train_process(model,train_loader,val_loader,epochs=40,model_name=model_name)
    matlib_acc_loss(train_process_df,model_name)

    print("\n训练结果")
    print(train_process_df)
    print(f"最佳验证集准确率: {train_process_df['best_acc'].iloc[0]:.4f}")
    print(f"训练时间：{train_process_df['time_used'].iloc[0]:.2f} 秒")
