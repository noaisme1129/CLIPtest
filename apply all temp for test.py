import torch
import torch.nn as nn
import torch.optim as optim
import open_clip
from tqdm import tqdm
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torchvision import transforms

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
model, _, preprocess_fn = open_clip.create_model_and_transforms('convnext_base_w', pretrained='laion2b_s13b_b82k')
model.to(device).eval()

# dataset
cifar100_data = CIFAR100(root="./data", download=True, train=False, transform=preprocess_fn)
data_loader = DataLoader(cifar100_data, batch_size=16, shuffle=False)

def ensemble_all_templates(classnames, templates):
    with torch.no_grad():
        all_templates_embedding = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = open_clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            all_templates_embedding.append(class_embedding)
        all_templates_embeddings = torch.stack(all_templates_embedding, dim=1).cuda()
    return all_templates_embeddings

ensembled_tempaltes = ensemble_all_templates(cifar100_data.classes, all_templates)
# Save the ensembled templates
torch.save(ensembled_tempaltes, 'ensembled_templates.pt')


correct_count, total_count = 0, 0

with torch.no_grad():
    for batch_images, batch_labels in data_loader:
        batch_images = batch_images.to(device)

        # encode
        img_features = model.encode_image(batch_images)
        img_features /= img_features.norm(dim=-1, keepdim=True)

        similarity_scores = (100.0 * img_features @ ensembled_tempaltes).softmax(dim=-1)
        predictions = similarity_scores.argmax(dim=1)

        correct_count += (predictions.cpu() == batch_labels).sum().item()
        total_count += batch_labels.size(0)

final_accuracy = (correct_count / total_count) * 100
print(f"Zero-shot accuracy on CIFAR-100: {final_accuracy:.2f}%")