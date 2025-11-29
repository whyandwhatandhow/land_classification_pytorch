import torch
from torchsummary import summary
import torchvision.models as models

def get_resnet50(num_classes=21, in_channels=3):
    model = models.resnet50(weights=None)
    
    # 修改输入层(Conv1)以适配in_channels
    if in_channels != 3:
        conv1 = torch.nn.Conv2d(
            in_channels,
            model.conv1.out_channels,
            kernel_size=model.conv1.kernel_size,
            stride=model.conv1.stride,
            padding=model.conv1.padding,
            bias=model.conv1.bias is not None
        )
        with torch.no_grad():
            if in_channels < 3:
                conv1.weight[:, :in_channels] = model.conv1.weight[:, :in_channels]
            else:
                conv1.weight[:, :3] = model.conv1.weight
                if in_channels > 3:
                    # 新增通道通过循环复制RGB通道
                    for i in range(3, in_channels):
                        conv1.weight[:, i] = model.conv1.weight[:, i % 3]
        model.conv1 = conv1
    
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet50()
    model=model.to(device)
    if torch.cuda.is_available():
        print("当前使用的设备: CUDA (GPU)")
        print("CUDA 设备数量:", torch.cuda.device_count())
        print("当前CUDA设备索引:", torch.cuda.current_device())
        print("CUDA设备名称:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("当前使用的设备: CPU")
    print(summary(model, input_size=(3, 224, 224)))