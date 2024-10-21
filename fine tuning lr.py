import torch
import torch.nn as nn
import torch.optim as optim
import open_clip
from tqdm import tqdm
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import random

all_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained CLIP model
model, _, preprocess_fn = open_clip.create_model_and_transforms('convnext_base_w', pretrained='laion2b_s13b_b82k')
model.to(device).eval()

# Freeze model parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze token_embedding layer parameters
for param in model.token_embedding.parameters():
    param.requires_grad = True

# Calculate the number of parameters to fine-tune
num_params_to_tune = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters to fine-tune: {num_params_to_tune}")

num_samples_per_class = 1
train_data = CIFAR100(root="./data", download=True, train=True, transform=preprocess_fn)

# Sampling
def random_few_shot_sampling(dataset, num_samples_per_class):
    label_indices = {label: [] for label in range(len(dataset.classes))}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label_indices[label].append(idx)
    few_shot_indices = []
    for label in label_indices:
        few_shot_indices.extend(random.sample(label_indices[label], num_samples_per_class))
    return few_shot_indices
few_shot_indices = random_few_shot_sampling(train_data, num_samples_per_class)

random.shuffle(few_shot_indices)
split_idx = int(len(few_shot_indices) * 0.8)  # 80% for training, 20% for validation
train_indices = few_shot_indices[:split_idx]
val_indices = few_shot_indices[split_idx:]

few_shot_train_data = Subset(train_data, train_indices)
few_shot_val_data = Subset(train_data, val_indices)

train_loader = DataLoader(few_shot_train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(few_shot_val_data, batch_size=16, shuffle=False)

test_data = CIFAR100(root="./data", download=True, train=False, transform=preprocess_fn)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

text_labels = [template.format(label) for label in train_data.classes for template in all_templates]
text_inputs = open_clip.tokenize(text_labels).to(device)
selected_indices = range(0, 8000, 80)
selected_indices = list(selected_indices)[:100]
text_inputs = text_inputs[selected_indices]

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.token_embedding.parameters(), lr=5e-5, weight_decay=0.01)
num_epochs = 30
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

best_val_loss = float('inf')

# Training loop with validation
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_images, batch_labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)

        optimizer.zero_grad()
        img_features = model.encode_image(batch_images)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        txt_features = model.encode_text(text_inputs)
        txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)

        # Compute similarity scores and loss
        similarity_scores = 100.0 * img_features @ txt_features.T
        loss = criterion(similarity_scores, batch_labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    scheduler.step()

    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_images, val_labels in tqdm(val_loader, desc="Validation"):
            val_images, val_labels = val_images.to(device), val_labels.to(device)

            img_features = model.encode_image(val_images)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            txt_features = model.encode_text(text_inputs)
            txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)

            similarity_scores = 100.0 * img_features @ txt_features.T
            loss = criterion(similarity_scores, val_labels)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

    # Save the model with the best validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print("Model saved with improved validation loss.")

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))

# Evaluation on test set
model.eval()
correct_count, total_count = 0, 0

with torch.no_grad():
    for batch_images, batch_labels in tqdm(test_loader, desc="Testing"):
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)

        img_features = model.encode_image(batch_images)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)

        txt_features = model.encode_text(text_inputs)
        txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)
        similarity_scores = 100.0 * img_features @ txt_features.T
        predictions = similarity_scores.argmax(dim=1)

        correct_count += (predictions.cpu() == batch_labels.cpu()).sum().item()
        total_count += batch_labels.size(0)

final_accuracy = (correct_count / total_count) * 100
print(f"Fine-tuned CIFAR-100 Accuracy: {final_accuracy:.2f}%")
