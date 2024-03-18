import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import shutil   
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

def extract_features(image_path):
    vgg16 = models.vgg16(pretrained=True)
    vgg16.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        vgg16.to('cuda')
    with torch.no_grad():
        features = vgg16.features(input_batch)
    features = torch.flatten(features, 1)
    return features

def cluster_images(folder_path):
    image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.jpg', '.jpeg', '.png'))]
    all_features = []
    for image_path in image_files:
        features = extract_features(image_path)
        all_features.append(features)

    all_features = torch.stack(all_features)
    all_features = np.vstack(all_features)
    
    pca = PCA(n_components=20) 
    pca_features = pca.fit_transform(all_features)

    kmeans = KMeans(n_clusters=7)  # You can adjust the number of clusters as needed
    kmeans.fit(pca_features)
    labels = kmeans.labels_

    output_folder = f'{folder_path}clustered_images'
    os.makedirs(output_folder, exist_ok=True)
    for i in range(kmeans.n_clusters):
        os.makedirs(os.path.join(output_folder, f'cluster_{i}'), exist_ok=True)

    for i, image_path in enumerate(image_files):
        cluster_label = labels[i]
        cluster_folder = os.path.join(output_folder, f'cluster_{cluster_label}')
        shutil.move(image_path, cluster_folder)