import torch
import torch.nn as nn
import torch.optim as optim
import open_clip
from tqdm import tqdm
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import random
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, _, preprocess_fn = open_clip.create_model_and_transforms('convnext_base_w', pretrained='laion2b_s13b_b82k')
model.to(device).eval()

for param in model.parameters():
    param.requires_grad = False
for param in model.token_embedding.parameters():
    param.requires_grad = True

num_repeats = 10
shots = [1, 2, 4, 8, 16]
all_accuracies = []
num_epochs = 400
batch_size = 128

train_data = CIFAR100(root="./data", download=True, train=True, transform=preprocess_fn)
test_data = CIFAR100(root="./data", download=True, train=False, transform=preprocess_fn)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

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
text_labels = [template.format(label) for label in train_data.classes for template in all_templates]
text_inputs = open_clip.tokenize(text_labels).to(device)
selected_indices = list(range(0, len(text_labels), 80))[:100]
text_inputs = text_inputs[selected_indices]

criterion = nn.CrossEntropyLoss()


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



all_accuracies = []

for repeat in range(8, 11):
    print(f"\nExperiment Repeat {repeat + 1}/{num_repeats}")
    experiment_accuracies = []

    for shot in shots:
        print(f"\nFine-tuning with {shot} shots per class.")

        model, _, preprocess_fn = open_clip.create_model_and_transforms('convnext_base_w',
                                                                        pretrained='laion2b_s13b_b82k')
        model.to(device).train()

        for param in model.parameters():
            param.requires_grad = False
        for param in model.token_embedding.parameters():
            param.requires_grad = True

        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

        few_shot_indices = random_few_shot_sampling(train_data, shot)
        few_shot_train_data = Subset(train_data, few_shot_indices)
        train_loader = DataLoader(few_shot_train_data, batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in tqdm(range(num_epochs), desc=f"Experiment {repeat + 1}, Shot {shot}"):
            running_loss = 0.0
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

        model.eval()
        correct_count, total_count = 0, 0
        with torch.no_grad():
            for batch_images, batch_labels in test_loader:
                batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
                img_features = model.encode_image(batch_images)
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                txt_features = model.encode_text(text_inputs)
                txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)

                similarity_scores = 100.0 * img_features @ txt_features.T
                predictions = similarity_scores.argmax(dim=1)
                correct_count += (predictions.cpu() == batch_labels.cpu()).sum().item()
                total_count += batch_labels.size(0)

        accuracy = (correct_count / total_count) * 100
        experiment_accuracies.append(accuracy)

    all_accuracies.append(experiment_accuracies)

    plt.figure()
    plt.plot(shots, experiment_accuracies, marker='o', label=f'Experiment {repeat + 1}')
    plt.xlabel("Shots")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Experiment {repeat + 1} Accuracy by Shot")
    plt.grid()
    plt.savefig(f"experiment_{repeat + 1}_accuracy.png")
    plt.close()

print("All experiments completed.")
