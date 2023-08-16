import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

class FlickrDataset(Dataset):
    def __init__(self, image_folder, captions_file, vocab, transform=None):
        self.image_folder = image_folder
        self.captions_df = pd.read_csv(captions_file)
        self.vocab = vocab
        self.transform = transform
    
    def __len__(self):
        return len(self.captions_df)
    
    def __getitem__(self, index):
        image_name = self.captions_df.iloc[index, 0]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        caption = self.captions_df.iloc[index, 1]
        caption = [self.vocab(token) for token in caption.split()]
        caption = torch.tensor(caption)
        return image, caption

def collate_fn(data):
    images, captions = zip(*data)
    images = torch.stack(images)
    lengths = [len(caption) for caption in captions]
    captions_padded = torch.zeros(len(captions), max(lengths)).long()

    for i, caption in enumerate(captions):
        end = lengths[i]
        captions_padded[i, :end] = caption[:end]

    return images, captions_padded, lengths
