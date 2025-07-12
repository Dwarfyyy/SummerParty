import torch
import pytorch_lightning as pl
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
import os
from model import GeneratorTransformer

def train_tokenizer(text_file: str, vocab_size: int = 30000):
    """
    Обучает BPE токенизатор на тексте.
    """
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[BOS]", "[EOS]"])
    tokenizer.train([text_file], trainer)
    tokenizer.save("results/tokenizer.json")
    return tokenizer

def main():
    # Параметры
    max_length = 192
    num_epochs = 10
    vocab_size = 30000
    
    # Подготовка папки для результатов
    os.makedirs('results', exist_ok=True)
    
    # Обучение токенизатора
    tokenizer = train_tokenizer('text.txt', vocab_size)
    
    # Инициализация модели с явной передачей всех гиперпараметров
    model = GeneratorTransformer(
        vocab_size=vocab_size,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        max_length=max_length,
        dropout=0.1,
        tokenizer=tokenizer,
        text_file='text.txt'
    )
    
    # Обучение
    trainer = pl.Trainer(max_epochs=num_epochs, accelerator='gpu' if torch.cuda.is_available() else 'cpu')
    trainer.fit(model)
    
    # Сохранение чекпоинта
    trainer.save_checkpoint("results/checkpoint.pt")

if __name__ == "__main__":
    main()