from collections import Counter

def build_vocab(captions):
  # Assuming captions is a list of tokenized captions
  all_words = [word for caption in captions for word in caption]
  word_counts = Counter(all_words)

  # Create a vocabulary mapping word to index
  vocab = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}  # Special tokens
  for word, count in word_counts.most_common():
      vocab[word] = len(vocab)

  vocab_size = len(vocab)
  
  return vocab, vocab_size
