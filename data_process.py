import torch
from torchvision import transforms
from torchvision.datasets import EuroSAT
from torch.utils.data import DataLoader

def eurosat_data_process(data_root='./data'):

    input_size = 224
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # 加载完整数据集
    full_dataset = EuroSAT(root=data_root, download=True, transform=transform)

    # 计算划分比例
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size   = int(0.1 * total_size)
    test_size  = total_size - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True,pin_memory=True, num_workers=4)
    val_loader   = DataLoader(val_set, batch_size=32, shuffle=False,pin_memory=True, num_workers=4)
    test_loader  = DataLoader(test_set, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)

    return train_loader, val_loader, test_loader, full_dataset.classes

if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = eurosat_data_process()
    print(f"类别数: {len(class_names)}")
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")