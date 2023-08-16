import torch
import torch.nn as nn
import torchvision.models as models

class CaptionGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(CaptionGenerator, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.cnn = models.resnet50(pretrained=True)  # You can use any CNN model
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])  # Remove the classification layer
        self.lstm = nn.LSTM(embed_size + 2048, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, images, captions):
        features = self.cnn(images).squeeze()
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        lstm_out, _ = self.lstm(embeddings)
        outputs = self.linear(lstm_out)
        return outputs
