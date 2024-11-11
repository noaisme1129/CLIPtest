import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import open_clip
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
import random
import matplotlib.pyplot as plt

# Define learning rates for testing
learning_rates = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

# Prompt templates for text inputs
all_templates = [
    'a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.',
    'a photo of the hard to see {}.', 'a low resolution photo of the {}.',
    'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.',
    'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.',
    'a photo of a hard to see {}.', 'a bright photo of a {}.', 'a photo of a clean {}.',
    'a photo of a dirty {}.', 'a dark photo of the {}.', 'a drawing of a {}.',
    'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.',
    'a close-up photo of a {}.', 'a black and white photo of the {}.',
    'a painting of the {}.', 'a painting of a {}.', 'a pixelated photo of the {}.',
    'a sculpture of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.',
    'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.',
    'a rendering of the {}.', 'a {} in a video game.', 'a photo of one {}.',
    'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.',
    'the origami {}.', 'the {} in a video game.', 'a sketch of a {}.',
    'a doodle of the {}.', 'a origami {}.', 'a low resolution photo of a {}.',
    'the toy {}.', 'a rendition of the {}.', 'a photo of the clean {}.',
    'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.',
    'a photo of a weird {}.', 'a blurry photo of a {}.', 'a cartoon {}.',
    'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.',
    'a pixelated photo of a {}.', 'itap of the {}.', 'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.', 'a plushie {}.', 'a photo of the nice {}.',
    'a photo of the small {}.', 'a photo of the weird {}.', 'the cartoon {}.',
    'art of the {}.', 'a drawing of the {}.', 'a photo of the large {}.',
    'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.',
    'itap of a {}.', 'graffiti of the {}.', 'a toy {}.', 'itap of my {}.',
    'a photo of a cool {}.', 'a photo of a small {}.', 'a tattoo of the {}.'
]

# Define device
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Load model and set it to evaluation mode, unfreezing only the text embeddings
model, _, preprocess_fn = open_clip.create_model_and_transforms('convnext_base_w', pretrained='laion2b_s13b_b82k')
model.to(device).eval()

for param in model.parameters():
    param.requires_grad = False
for param in model.token_embedding.parameters():
    param.requires_grad = True

# Print number of parameters to fine-tune
num_params_to_tune = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of fine-tuning parameters: {num_params_to_tune}")

# Few-shot sampling function
def random_few_shot_sampling(dataset, num_samples_per_class):
    label_counts = {label: 0 for label in range(len(dataset.classes))}
    few_shot_indices = []
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)
    for idx in all_indices:
        _, label = dataset[idx]
        if label_counts[label] < num_samples_per_class:
            few_shot_indices.append(idx)
            label_counts[label] += 1
        if all(count == num_samples_per_class for count in label_counts.values()):
            break
    return few_shot_indices

# Few-shot CIFAR-100 setup
num_samples_per_class = 1
train_data = CIFAR100(root="./data", download=True, train=True, transform=preprocess_fn)

# Few-shot indices and subset for training
few_shot_indices = random_few_shot_sampling(train_data, num_samples_per_class)
few_shot_train_data = Subset(train_data, few_shot_indices)

# Split few-shot training set into training and validation sets (80/20 split)
train_size = int(0.8 * len(few_shot_train_data))
val_size = len(few_shot_train_data) - train_size
train_subset, val_subset = random_split(few_shot_train_data, [train_size, val_size])

# Loaders for training, validation, and test sets
train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)
test_data = CIFAR100(root="./data", download=True, train=False, transform=preprocess_fn)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

# Text inputs for each class
text_labels = [template.format(label) for label in train_data.classes for template in all_templates]
text_inputs = open_clip.tokenize(text_labels).to(device)
selected_indices = range(0, 8000, 80)  # Select a subset of text tokens
text_inputs = text_inputs[selected_indices]

# Define loss criterion
criterion = nn.CrossEntropyLoss()

# Lists to store errors
val_errors = []
test_errors = []

# Training and evaluating across different learning rates
for lr in learning_rates:
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_epochs = 100
    print(f"Training with learning rate: {lr}")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False) as pbar:
            for batch_images, batch_labels in train_loader:
                batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
                optimizer.zero_grad()

                img_features = model.encode_image(batch_images)
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                txt_features = model.encode_text(text_inputs)
                txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)
                similarity_scores = 100.0 * img_features @ txt_features.T
                loss = criterion(similarity_scores, batch_labels)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix({"Loss": loss.item()})
                pbar.update(1)

    # Validation error calculation
    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for batch_images, batch_labels in tqdm(val_loader, desc="Validation", leave=False):
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            img_features = model.encode_image(batch_images)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            similarity_scores = 100.0 * img_features @ txt_features.T
            predictions = similarity_scores.argmax(dim=1)
            correct_val += (predictions.cpu() == batch_labels.cpu()).sum().item()
            total_val += batch_labels.size(0)
    val_error = 1 - correct_val / total_val
    val_errors.append(val_error)

    # Test error calculation
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for batch_images, batch_labels in tqdm(test_loader, desc="Testing", leave=False):
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            img_features = model.encode_image(batch_images)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            similarity_scores = 100.0 * img_features @ txt_features.T
            predictions = similarity_scores.argmax(dim=1)
            correct_test += (predictions.cpu() == batch_labels.cpu()).sum().item()
            total_test += batch_labels.size(0)
    test_error = 1 - correct_test / total_test
    test_errors.append(test_error)

# Plotting validation and test errors
plt.figure(figsize=(10, 6))
plt.plot(learning_rates, val_errors, label="Validation Error", marker='o')
plt.plot(learning_rates, test_errors, label="Test Error", marker='s')
plt.xscale("log")
plt.xlabel("Learning Rate")
plt.ylabel("Error")
plt.title("Validation and Test Errors vs Learning Rate for 1-shot")
plt.legend()
plt.savefig('learning_rate_vs_error.png')
plt.show()
