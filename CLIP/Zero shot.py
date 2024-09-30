import torch
import open_clip
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader

# 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# download model
model, _, preprocess_fn = open_clip.create_model_and_transforms('convnext_base_w', pretrained='laion2b_s13b_b82k')
model.to(device).eval()

# dataset
cifar100_data = CIFAR100(root="./data", download=True, train=False, transform=preprocess_fn)
data_loader = DataLoader(cifar100_data, batch_size=16, shuffle=False)


text_labels = [f"a photo of a {label}" for label in cifar100_data.classes]
text_inputs = open_clip.tokenize(text_labels).to(device)

correct_count, total_count = 0, 0

with torch.no_grad():
    for batch_images, batch_labels in data_loader:
        batch_images = batch_images.to(device)

        # encode
        img_features = model.encode_image(batch_images)
        img_features /= img_features.norm(dim=-1, keepdim=True)

        txt_features = model.encode_text(text_inputs)
        txt_features /= txt_features.norm(dim=-1, keepdim=True)

        similarity_scores = (100.0 * img_features @ txt_features.T).softmax(dim=-1)
        predictions = similarity_scores.argmax(dim=1)

        correct_count += (predictions.cpu() == batch_labels).sum().item()
        total_count += batch_labels.size(0)

final_accuracy = (correct_count / total_count) * 100
print(f"Zero-shot accuracy on CIFAR-100: {final_accuracy:.2f}%")
