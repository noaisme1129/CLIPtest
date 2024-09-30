import torch
import open_clip
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader

# 确定使用的计算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化 CLIP 模型
model, _, preprocess_fn = open_clip.create_model_and_transforms('convnext_base_w', pretrained='laion2b_s13b_b82k')
model.to(device).eval()

# 加载 CIFAR-100 数据集
cifar100_data = CIFAR100(root="./data", download=True, train=False, transform=preprocess_fn)
data_loader = DataLoader(cifar100_data, batch_size=16, shuffle=False)

# 准备文本输入
text_labels = [f"a photo of a {label}" for label in cifar100_data.classes]
text_inputs = open_clip.tokenize(text_labels).to(device)

# 计数器初始化
correct_count, total_count = 0, 0

# 推断过程
with torch.no_grad():
    for batch_images, batch_labels in data_loader:
        batch_images = batch_images.to(device)

        # 编码图像与文本
        img_features = model.encode_image(batch_images)
        img_features /= img_features.norm(dim=-1, keepdim=True)

        txt_features = model.encode_text(text_inputs)
        txt_features /= txt_features.norm(dim=-1, keepdim=True)

        # 计算相似度及预测
        similarity_scores = (100.0 * img_features @ txt_features.T).softmax(dim=-1)
        predictions = similarity_scores.argmax(dim=1)

        correct_count += (predictions.cpu() == batch_labels).sum().item()
        total_count += batch_labels.size(0)

# 计算并输出准确率
final_accuracy = (correct_count / total_count) * 100
print(f"Zero-shot accuracy on CIFAR-100: {final_accuracy:.2f}%")