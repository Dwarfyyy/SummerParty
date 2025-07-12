import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

class TextDataset(Dataset):
    """
    Кастомный датасет для обучения Transformer на тексте.
    Разбивает текст на блоки длиной max_length с BOS и EOS токенами.
    """
    def __init__(self, text_file: str, tokenizer: Tokenizer, max_length: int = 192):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bos_token_id = 1
        self.eos_token_id = 2
        
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Токенизация текста
        encoded = self.tokenizer.encode(text).ids
        self.data = []
        
        # Разбиение на блоки
        for i in range(0, len(encoded) - max_length, max_length):
            chunk = [self.bos_token_id] + encoded[i:i + max_length - 1] + [self.eos_token_id]
            self.data.append(chunk)
        
        # Последний блок (если остался)
        if len(encoded) % max_length != 0:
            chunk = [self.bos_token_id] + encoded[-max_length + 1:] + [self.eos_token_id]
            self.data.append(chunk)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)