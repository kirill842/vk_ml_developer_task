import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class LogoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.classes = os.listdir(root_dir)
        self.image_paths = []
        self.labels = []

        for i, cls in enumerate(self.classes):
            paths = glob.glob(os.path.join(root_dir, cls, "*.jpg"))
            self.image_paths.extend(paths)
            self.labels.extend([i] * len(paths))

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# Аугментации
train_transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2),
    T.RandomRotation(30),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = (anchor - positive).pow(2).sum(1)
        neg_dist = (anchor - negative).pow(2).sum(1)
        loss = torch.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    losses = []
    for (anc, pos, neg) in tqdm(train_loader):
        anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
        optimizer.zero_grad()
        e_anc = model(anc)
        e_pos = model(pos)
        e_neg = model(neg)
        loss = criterion(e_anc, e_pos, e_neg)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

# Инициализация
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmbeddingNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = TripletLoss()

# Тренировочный цикл (псевдокод)
# dataset = LogoDataset("Logodet3K/Train/", train_transform)
# train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
# for epoch in range(10):
#     loss = train(model, train_loader, optimizer, criterion, device)

def get_prototype(support_images, model, transform):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for img in support_images:
            img = transform(img).unsqueeze(0).to(device)
            emb = model(img).cpu().numpy()
            embeddings.append(emb)
    return np.mean(embeddings, axis=0)

def is_logo(query_image, prototype, model, transform, threshold=0.7):
    model.eval()
    with torch.no_grad():
        query = transform(query_image).unsqueeze(0).to(device)
        emb = model(query).cpu().numpy()
    similarity = np.dot(emb, prototype.T) / (np.linalg.norm(emb)*np.linalg.norm(prototype))
    return similarity > threshold

# Пример использования
support_images = [Image.open("nike1.jpg"), Image.open("nike2.jpg")]
query_image = Image.open("query.jpg")

prototype = get_prototype(support_images, model, train_transform)
result = is_logo(query_image, prototype, model, train_transform, threshold=0.6)

print("Логотип является логотипом искомой организации" if result else "Логотип не является логотипом искомой организации")
