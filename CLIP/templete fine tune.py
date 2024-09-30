import torch
import torch.nn as nn
import torch.optim as optim
import open_clip
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model, _, preprocess_fn = open_clip.create_model_and_transforms('convnext_base_w', pretrained='laion2b_s13b_b82k')
model.to(device)


num_params_to_tune = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"number of fine tune parameter: {num_params_to_tune}")


transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

train_data = CIFAR100(root="./data", download=True, train=True, transform=transform)
test_data = CIFAR100(root="./data", download=True, train=False, transform=transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# diffenet templete
text_labels = [f"a photo of a {label}" for label in train_data.classes]
# text_labels = [f"an image showing a {label}" for label in train_data.classes]
# text_labels = [f"this is a {label}" for label in train_data.classes]
text_inputs = open_clip.tokenize(text_labels).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# training
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_images, batch_labels in train_loader:
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)

        optimizer.zero_grad()

        #
        img_features = model.encode_image(batch_images)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)  # 避免就地操作

        txt_features = model.encode_text(text_inputs)
        txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)  # 避免就地操作

        # 计算相似度及预测
        similarity_scores = (100.0 * img_features @ txt_features.T)
        loss = criterion(similarity_scores, batch_labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")


model.eval()
correct_count, total_count = 0, 0

with torch.no_grad():
    for batch_images, batch_labels in test_loader:
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)

        img_features = model.encode_image(batch_images)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)  # 避免就地操作

        txt_features = model.encode_text(text_inputs)
        txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)  # 避免就地操作

        
        similarity_scores = (100.0 * img_features @ txt_features.T).softmax(dim=-1)
        predictions = similarity_scores.argmax(dim=1)

        correct_count += (predictions.cpu() == batch_labels.cpu()).sum().item()
        total_count += batch_labels.size(0)


final_accuracy = (correct_count / total_count) * 100
print(f"Fine-tuned accuracy on CIFAR-100: {final_accuracy:.2f}%")
