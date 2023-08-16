import torch
from model_base import CaptionGenerator

# Run: python3 -m tests.base   
if __name__ == '__main__':
    # Parameters
    vocab_size = 10000  # Replace with actual vocab size
    embed_size = 256
    hidden_size = 512
    num_layers = 2
    max_seq_length = 20  # Maximum sequence length for captions
    
    # Create a sample CaptionGenerator instance
    model = CaptionGenerator(vocab_size, embed_size, hidden_size, num_layers)
    
    # Generate random input tensors
    batch_size = 8
    image_size = (3, 224, 224)  # Channels x Height x Width
    images = torch.randn(batch_size, *image_size)
    captions = torch.randint(0, vocab_size, (batch_size, max_seq_length))
    
    # Calculate model outputs
    with torch.no_grad():
        outputs = model(images, captions)
    
    print("Input Images Shape:", images.shape)
    print("Input Captions Shape:", captions.shape)
    print("Model Outputs Shape:", outputs.shape)
