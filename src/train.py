import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from model_base import CaptionGenerator
from dataset import FlickrDataset, collate_fn
from utils import build_vocab

image_paths = [...]  # List of image file paths
captions = [...]  # List of caption indices
vocab, vocab_size = build_vocab(captions)

# Hyperparameters
embed_size = 256
hidden_size = 512
num_layers = 2
batch_size = 32
learning_rate = 0.001
num_epochs = 10


# Create dataset and dataloader
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = FlickrDataset(image_paths, captions, vocab, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Initialize model, loss function, and optimizer
model = CaptionGenerator(vocab_size, embed_size, hidden_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch_idx, (images, captions) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(images, captions[:, :-1])  # Exclude the last token from captions
        targets = captions[:, 1:]  # Exclude the first token from targets
        loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'caption_generator.pth')
