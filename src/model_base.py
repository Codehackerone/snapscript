import torch
import torch.nn as nn
import torchvision.models as models

class CaptionGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(CaptionGenerator, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.cnn = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])  # Remove the classification and pool layers
        self.lstm = nn.LSTM(embed_size + 2048, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, images, captions):
        batch_size, caption_length = captions.shape
        
        # Extract CNN features and reshape
        features = self.cnn(images)
        features = features.view(batch_size, -1, 2048)  # Reshape to (batch_size, num_features, 2048)
        features = features.mean(dim=1)  # Average pooling over the spatial dimensions
        
        # Expand features across the caption length
        features_expanded = features.unsqueeze(1).expand(-1, caption_length, -1)
        
        # Embed captions and concatenate with features
        embeddings = self.embed(captions)
        lstm_input = torch.cat((features_expanded, embeddings), dim=2)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(lstm_input)
        
        # Linear layer
        outputs = self.linear(lstm_out)
        return outputs
