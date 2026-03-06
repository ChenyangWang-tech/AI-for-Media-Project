# Fifth Iteration：
## Modifications：
- I replaced ResNet18 with MobileNetV3[10], a lightweight model optimized for mobile and CPU environments. MobileNetV3 has fewer parameters and faster inference speeds.
- In this iteration, I introduced additional data augmentation techniques specifically tailored for hand gestures, such as RandomPerspective, RandomAffine, and RandomErasing.
- I implemented Label Smoothing (with a smoothing factor of 0.05) to prevent overfitting.
- I used the AdamW optimizer combined with the CosineAnnealingLR learning rate scheduler to enhance training stability.
- Streamlit Interface Optimization: Utilized st.cache_resource and st.cache_data to cache the model and class labels, improving interface responsiveness. Added status displays and real-time FPS monitoring to enhance user experience.
##  Training code display：
```
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# 系统级优化（根据实际CPU核心数调整）
num_workers = 4 if os.cpu_count() >= 8 else 2
torch.set_num_threads(4)  # 限制CPU线程数

# 增强的数据预处理
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, shear=15),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.4, scale=(0.02, 0.1))
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 数据集路径
train_path = 'dataset/train'
val_path = 'dataset/val'

# 加载数据集
train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
val_dataset = datasets.ImageFolder(root=val_path, transform=val_transform)

# 类别平衡采样
class_counts = np.bincount(train_dataset.targets)
class_weights = 1. / torch.Tensor(class_counts)
samples_weights = class_weights[train_dataset.targets]

sampler = WeightedRandomSampler(
    weights=samples_weights,
    num_samples=len(samples_weights),
    replacement=True
)

# 数据加载器配置
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler,
    num_workers=num_workers,
    pin_memory=False,
    persistent_workers=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=False
)

# 优化后的轻量级模型
class BehaviorNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.base_model = models.mobilenet_v3_small(pretrained=True)
        
        # 冻结前70%的层（适用于小数据集）
        total_layers = len(list(self.base_model.features))
        freeze_idx = int(total_layers * 0.7)
        for idx, layer in enumerate(self.base_model.features):
            if idx < freeze_idx:
                for param in layer.parameters():
                    param.requires_grad = False
                    
        # 优化分类头
        self.base_model.classifier = nn.Sequential(
            nn.Linear(576, 512),
            nn.Hardswish(),
            nn.Dropout(0.5),
            nn.LayerNorm(512),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)

# 训练配置
model = BehaviorNet(num_classes=4)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0005,
    weight_decay=0.001
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

# 保存路径
save_dir = 'model_checkpoints'
os.makedirs(save_dir, exist_ok=True)
np.save(os.path.join(save_dir, 'class_names.npy'), train_dataset.classes)

def train_model():
    best_accuracy = 0.0
    early_stop_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(50):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            running_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 记录指标
        train_loss = running_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(accuracy)
        
        # 打印分类报告
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1}/50 ({epoch_time:.1f}s)")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {accuracy:.2f}%")
        
        # 早停机制
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        else:
            early_stop_counter += 1
            if early_stop_counter >= 10:
                print(f"Early stopping at epoch {epoch+1}, Best Acc: {best_accuracy:.2f}%")
                break
        
        scheduler.step()
    
    # 保存训练曲线
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Accuracy', color='green')
    plt.title('Accuracy Curve')
    plt.legend()
    
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()

if __name__ == '__main__':
    print("Training set distribution:", Counter(train_dataset.targets))
    print("Validation set distribution:", Counter(val_dataset.targets))
    train_model()
```
##  applying code display：
```
import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import time

# 页面配置
st.set_page_config(
    page_title="Real-Time Behavior Monitor",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 自定义样式
st.markdown("""
<style>
    .st-camera-frame {
        border-radius: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stAlert {
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 模型定义（必须与训练代码一致）
class BehaviorNet(torch.nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.base_model = models.mobilenet_v3_small(pretrained=False)
        self.base_model.classifier = torch.nn.Sequential(
            nn.Linear(576, 512),
            nn.Hardswish(),
            nn.Dropout(0.5),
            nn.LayerNorm(512),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)

# 加载模型
@st.cache_resource
def load_model():
    model = BehaviorNet(num_classes=4)
    model.load_state_dict(torch.load('model_checkpoints/best_model.pth', map_location='cpu'))
    model.eval()
    return model

# 加载类别标签
@st.cache_data
def get_classes():
    return np.load('model_checkpoints/class_names.npy', allow_pickle=True).tolist()

class BehaviorDetector:
    def __init__(self):
        self.model = load_model()
        self.classes = get_classes()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.history = []
        self.last_pred_time = 0
        
        # 状态配置
        self.status_config = {
            'Normal': {'color': (0, 255, 0), 'text': 'Normal'},
            'Biting_Nails': {'color': (0, 0, 255), 'text': 'Nail Biting!'},
            'Picking_Face': {'color': (0, 0, 255), 'text': 'Face Picking!'},
            'Biting_Oral_Mucosa': {'color': (0, 0, 255), 'text': 'Mucosa Biting!'}
        }
    
    def _preprocess(self, frame):
        """预处理帧数据"""
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return self.transform(img).unsqueeze(0)
    
    def predict(self, frame):
        """执行预测"""
        current_time = time.time()
        if current_time - self.last_pred_time < 0.3:  # 300ms间隔
            return None
        
        self.last_pred_time = current_time
        tensor = self._preprocess(frame)
        
        with torch.no_grad():
            output = self.model(tensor)
            prob = torch.nn.functional.softmax(output, dim=1)
            conf, pred = torch.max(prob, 1)
        
        return {
            'class': self.classes[pred.item()],
            'confidence': conf.item()
        }
    
    def draw_overlay(self, frame, prediction):
        """绘制叠加信息"""
        if prediction['confidence'] < 0.7:
            return frame
        
        status = self.status_config.get(prediction['class'], {})
        text = f"{status.get('text', 'Unknown')} ({prediction['confidence']:.2f})"
        
        # 绘制背景框
        cv2.rectangle(frame, (10, 10), (300, 50), (255, 255, 255), -1)
        
        # 绘制状态文本
        cv2.putText(frame, text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                   status.get('color', (0, 0, 0)), 2)
        
        return frame

def main():
    st.title("Real-Time Behavior Detection")
    detector = BehaviorDetector()
    
    # 侧边栏配置
    with st.sidebar:
        st.header("Settings")
        run_app = st.checkbox("Enable Camera", True)
        conf_thresh = st.slider("Confidence Threshold", 0.5, 0.95, 0.75)
        st.markdown("---")
        st.subheader("Class Definitions")
        for cls in detector.classes:
            st.markdown(f"- **{cls}**: {detector.status_config.get(cls, {}).get('text', '')}")
    
    # 主界面
    camera_placeholder = st.empty()
    status_placeholder = st.empty()
    
    cap = cv2.VideoCapture(0)
    last_alert = ""
    
    try:
        while run_app and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Camera Error")
                break
            
            # 执行预测
            prediction = detector.predict(frame)
            
            # 处理结果显示
            if prediction:
                frame = detector.draw_overlay(frame, prediction)
                if prediction['confidence'] > conf_thresh:
                    status_placeholder.success(f"Status: {prediction['class']} ({prediction['confidence']:.2f})")
                else:
                    status_placeholder.info("Status: Normal (Default)")
            
            # 显示帧率
            fps = 1 / (time.time() - detector.last_pred_time + 1e-6)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            camera_placeholder.image(frame, channels="BGR")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
    finally:
        cap.release()

if __name__ == "__main__":
    main()
```

## Training results：
![d9ddf897244cced0e572c5ba5b9bf30](https://git.arts.ac.uk/24001444/AI-4-Media-Project-Chenyang-Wang/assets/1324/f5750301-cce5-4418-9780-0f2b0b4d5398)
![ea8c72481d19bd98dc77cf3287c4db3](https://git.arts.ac.uk/24001444/AI-4-Media-Project-Chenyang-Wang/assets/1324/92995407-ae51-44e4-af12-c398d86a4bf4)
![e83cbc5687ea2099c8153b248310106](https://git.arts.ac.uk/24001444/AI-4-Media-Project-Chenyang-Wang/assets/1324/cc345285-2bca-4be6-b917-ade1a7009975)
![1339f2472ff054057faa2ab474d2e0c](https://git.arts.ac.uk/24001444/AI-4-Media-Project-Chenyang-Wang/assets/1324/5d0feac7-9d30-4db3-b45b-d15eff0d9701)
![70cf9368693b9d6f8b56741bfa69b89](https://git.arts.ac.uk/24001444/AI-4-Media-Project-Chenyang-Wang/assets/1324/88cfc09f-34fe-498c-b2b1-ac224a88cf0c)
![b6b8f3f6a3a5b6a56d150d96a415fdc](https://git.arts.ac.uk/24001444/AI-4-Media-Project-Chenyang-Wang/assets/1324/9d29b4c7-bcca-45e2-8c75-1d681dd44449)


## Analysis of results：
- The training loss gradually decreased from 0.8662 to 0.2108, indicating that the model fits the training data reasonably well. However, the slow decline in training loss and the relatively high final value suggest that the model may not have fully learned the data's features.
- The validation loss fluctuated between 2.0 and 3.5 without a clear downward trend. The validation loss being significantly higher than the training loss indicates potential overfitting or underfitting.
- The validation accuracy fluctuated between 30% and 55%, with the final best accuracy reaching 55%. For a 4-class classification task, random guessing would yield an accuracy of 25%. The model's accuracy being only slightly better than random chance suggests that it failed to effectively learn distinguishing features between classes.


## Issues：
- Insufficient Data: The dataset is extremely small for a deep learning model. Deep learning models typically require thousands or even tens of thousands of images to learn effective features. If the dataset lacks diversity in backgrounds, lighting, angles, etc., the model may struggle to generalize to real-world scenarios.
- Inability to Capture Subtle Motion Features: The model may not be capable of detecting fine-grained motion characteristics.
- CPU Performance Limitations: The inference speed is constrained by the CPU's performance.
