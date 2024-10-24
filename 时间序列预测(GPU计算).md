```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset

# 检查是否有CUDA设备可用，如果有，使用第一个可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 1. 数据集定义
class BirdIllnessDataset(Dataset):
    def __init__(self, dates, geos, histories, images, labels):
        self.dates = dates
        self.geos = geos
        self.histories = histories
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        date = self.dates[idx]
        geo = self.geos[idx]
        history = self.histories[idx]
        image = self.images[idx]
        label = self.labels[idx]
        
        return {
            'date': torch.tensor(date, dtype=torch.float32).to(device),
            'geo': torch.tensor(geo, dtype=torch.float32).to(device),
            'history': torch.tensor(history, dtype=torch.float32).to(device),
            'image': torch.tensor(image, dtype=torch.float32).to(device),
            'label': torch.tensor(label, dtype=torch.float32).to(device)
        }

# 2. 模型定义
class BirdIllnessPredictionModelWithImage(nn.Module):
    def __init__(self, input_size_date, input_size_geo, input_size_history):
        super(BirdIllnessPredictionModelWithImage, self).__init__()
        
        # 使用预训练的ResNet模型作为图像特征提取器
        resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # 提取512维的图像特征
        
        # 日期特征嵌入
        self.date_embedding = nn.Sequential(
            nn.Linear(input_size_date, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # 经纬度嵌入
        self.geo_embedding = nn.Sequential(
            nn.Linear(input_size_geo, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # 历史发病数据处理
        self.history_embedding = nn.Sequential(
            nn.Linear(input_size_history, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # 融合层和预测层
        self.fc = nn.Sequential(
            nn.Linear(512 + 32 + 32 + 32, 64),  # 图像、日期、经纬度、历史发病数据
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出预测值
        )
    
    def forward(self, date, geo, history, images):
        # 提取图像特征
        img_features = self.resnet(images)
        img_features = img_features.view(img_features.size(0), -1)  # 将特征展平成 [batch_size, 512]
        
        # 处理日期特征
        date_features = self.date_embedding(date)
        
        # 处理经纬度特征
        geo_features = self.geo_embedding(geo)
        
        # 处理历史发病数据
        history_features = self.history_embedding(history)
        
        # 将所有特征拼接
        combined_features = torch.cat((img_features, date_features, geo_features, history_features), dim=1)
        
        # 预测发病数量
        predicted_count = self.fc(combined_features)
        return predicted_count

# 3. 数据加载和准备
dates = [[2024, 10, 22], [2024, 10, 23]]  # 示例日期
geos = [[35.6895, 139.6917], [34.0522, -118.2437]]  # 示例经纬度
histories = [[1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8]]  # 示例历史发病数据
images = torch.randn(2, 3, 224, 224)  # 示例图像数据
labels = [8, 6]  # 示例标签

# 创建数据集和数据加载器
dataset = BirdIllnessDataset(dates, geos, histories, images, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 4. 模型实例化、损失函数与优化器
model = BirdIllnessPredictionModelWithImage(input_size_date=3, input_size_geo=2, input_size_history=7)
model.to(device)  # 将模型移到GPU
criterion = nn.MSELoss()  # 回归问题，使用均方误差
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 训练过程
num_epochs = 100
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    
    for batch in dataloader:
        # 获取输入数据
        date_input = batch['date']
        geo_input = batch['geo']
        history_input = batch['history']
        image_input = batch['image']
        labels = batch['label']
        
        # 前向传播
        outputs = model(date_input, geo_input, history_input, image_input)
        loss = criterion(outputs, labels.view(-1, 1))  # 计算损失
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # 每10个epoch打印一次损失
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')

# 6. 验证过程
model.eval()  # 设置为评估模式
with torch.no_grad():
    total_loss = 0.0
    for batch in dataloader:
        date_input = batch['date']
        geo_input = batch['geo']
        history_input = batch['history']
        image_input = batch['image']
        labels = batch['label']
        
        outputs = model(date_input, geo_input, history_input, image_input)
        loss = criterion(outputs, labels.view(-1, 1))
        total_loss += loss.item()
    
    print(f'Validation Loss: {total_loss / len(dataloader):.4f}')

# 7. 预测功能
def predict_bird_illness(model, date, geo, history, image):
    # Ensure data is in tensor form and move to the device
    # Check if the data is already a tensor, if not, convert it
    date_tensor = torch.tensor(date, dtype=torch.float32).to(device) if not isinstance(date, torch.Tensor) else date.to(device)
    geo_tensor = torch.tensor(geo, dtype=torch.float32).to(device) if not isinstance(geo, torch.Tensor) else geo.to(device)
    history_tensor = torch.tensor(history, dtype=torch.float32).to(device) if not isinstance(history, torch.Tensor) else history.to(device)
    image_tensor = image.to(device) if isinstance(image, torch.Tensor) else torch.tensor(image, dtype=torch.float32).to(device)
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # No gradient calculation
    with torch.no_grad():
        prediction = model(date_tensor.unsqueeze(0), geo_tensor.unsqueeze(0), history_tensor.unsqueeze(0), image_tensor.unsqueeze(0))
    
    # Convert the prediction to a scalar value
    predicted_count = prediction.item()
    
    return predicted_count

# 示例输入数据
date_example = [2024, 10, 22]  # 示例日期
geo_example = [35.6895, 139.6917]  # 示例经纬度
history_example = [1, 2, 3, 4, 5, 6, 7]  # 示例历史发病数据
image_example = torch.randn(3, 224, 224)  # 随机生成一个示例图像

# 调用预测函数
predicted_illness_count = predict_bird_illness(model, date_example, geo_example, history_example, image_example)
print("Predicted number of infected birds:", predicted_illness_count)
```

