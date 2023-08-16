import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel, BertTokenizer

class CaptionGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(CaptionGenerator, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.cnn = models.resnet50(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
        self.transformer = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(2048 + embed_size, vocab_size)
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def forward(self, images, captions):
        batch_size, caption_length = captions.shape
        
        features = self.cnn(images)
        features = features.view(batch_size, -1, 2048)
        features = features.mean(dim=1)
        
        tokens = [self.tokenizer.encode(sentence, add_special_tokens=True) for sentence in captions]
        tokens = torch.tensor(tokens).to(captions.device)
        
        embeddings = self.embed(tokens)
        
        transformer_input = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        
        transformer_output = self.transformer(input_ids=transformer_input)[0]
        
        outputs = self.linear(transformer_output)
        return outputs
