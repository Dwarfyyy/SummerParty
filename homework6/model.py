import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from dataset import TextDataset

class PositionalEncoding(nn.Module):
    """
    Реализация позиционного кодирования для Transformer.
    Добавляет информацию о порядке токенов в последовательности.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

class GeneratorTransformer(pl.LightningModule):
    """
    Модель Transformer с декодером для авторегрессивной генерации текста.
    """
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8, 
                 num_layers: int = 6, dim_feedforward: int = 2048, max_length: int = 192,
                 dropout: float = 0.1, tokenizer=None, text_file: str = "text.txt"):
        super().__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_length)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True),
            num_layers
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Токены BOS и EOS
        self.bos_token_id = 1  # Предполагается, что BOS имеет id=1
        self.eos_token_id = 2  # Предполагается, что EOS имеет id=2

        # Для train_dataloader
        self.tokenizer = tokenizer
        self.text_file = text_file
        self.batch_size = 1
        self.losses = []

        # Сохранение гиперпараметров для PyTorch Lightning
        self.save_hyperparameters(ignore=['tokenizer'])  # Игнорируем tokenizer, так как он не сериализуется

        # Перемещение модели на устройство, определённое Lightning
        self.to(self.device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Прямой проход модели.
        Args:
            x: Входные токены (batch_size, seq_len)
            mask: Маска для предотвращения внимания к будущим токенам
        Returns:
            Логиты для каждого токена (batch_size, seq_len, vocab_size)
        """
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(self.device)
        
        output = self.transformer_decoder(x, x, tgt_mask=mask)
        return self.fc_out(output)

    def training_step(self, batch, batch_idx):
        """
        Шаг обучения модели.
        Args:
            batch: Тензор с токенами (batch_size, seq_len)
            batch_idx: Индекс батча
        Returns:
            Значение функции потерь
        """
        x = batch
        y = x[:, 1:].contiguous()
        outputs = self(x[:, :-1], mask=None)
        loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), y.view(-1))
        self.losses.append(loss.item())
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Настройка оптимизатора.
        Returns:
            Оптимизатор Adam
        """
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def train_dataloader(self):
        """
        Создание DataLoader для обучения.
        Returns:
            DataLoader с тренировочными данными
        """
        dataset = TextDataset(self.text_file, self.tokenizer, self.max_length)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def on_train_epoch_end(self):
        """
        Действия после каждой эпохи обучения: сохранение графика потерь.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.losses, label='Обучение')
        plt.xlabel('Итерация')
        plt.ylabel('Потери')
        plt.title('График потерь при обучении')
        plt.legend()
        plt.savefig('results/loss_plot.png')
        plt.close()

    def generate(self, prompt: str, tokenizer, context_len: int = 50, 
                temperature: float = 1.0, max_out_tokens: int = 200, 
                beam_size: int = 1) -> str:
        """
        Генерирует текст авторегрессивно или с использованием beam search.
        Args:
            prompt: Начальный текст
            tokenizer: Токенизатор для кодирования/декодирования
            context_len: Длина контекста
            temperature: Температура для сэмплирования
            max_out_tokens: Максимальное количество генерируемых токенов
            beam_size: Размер луча для beam search (1 для обычной генерации)
        Returns:
            Сгенерированный текст
        """
        self.eval()
        with torch.no_grad():
            input_ids = tokenizer.encode(prompt).ids
            input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
            generated = input_ids.clone()

            if beam_size > 1:
                return self._beam_search(tokenizer, input_ids, context_len, temperature, max_out_tokens, beam_size)
            
            for _ in range(max_out_tokens):
                input_ids = generated[:, -context_len:]
                outputs = self(input_ids)
                next_token_logits = outputs[:, -1, :] / temperature
                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1)
                generated = torch.cat([generated, next_token], dim=1)
                
                if next_token.item() == self.eos_token_id:
                    break
            
            return tokenizer.decode(generated[0].tolist())

    def _beam_search(self, tokenizer, input_ids: torch.Tensor, context_len: int, 
                     temperature: float, max_out_tokens: int, beam_size: int) -> str:
        """
        Реализация beam search для генерации текста.
        Args:
            tokenizer: Токенизатор
            input_ids: Начальные токены
            context_len: Длина контекста
            temperature: Температура для сэмплирования
            max_out_tokens: Максимальное количество токенов
            beam_size: Размер луча
        Returns:
            Сгенерированный текст
        """
        beams = [(input_ids, 0.0)]  # (токены, лог-вероятность)
        for _ in range(max_out_tokens):
            new_beams = []
            for beam_ids, beam_score in beams:
                input_ids = beam_ids[:, -context_len:].to(self.device)
                outputs = self(input_ids)
                logits = outputs[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                top_probs, top_indices = torch.topk(probs, beam_size)
                
                # Извлекаем вероятности и индексы для первого (и единственного) элемента батча
                top_probs = top_probs[0]  # форма [beam_size]
                top_indices = top_indices[0]  # форма [beam_size]
                
                for prob, idx in zip(top_probs, top_indices):
                    new_ids = torch.cat([beam_ids, idx.unsqueeze(0).unsqueeze(0)], dim=1)
                    new_score = beam_score + torch.log(prob).item()
                    new_beams.append((new_ids, new_score))
                
                # Проверяем, не является ли последний токен EOS
                if beam_ids[0, -1].item() == self.eos_token_id:
                    new_beams.append((beam_ids, beam_score))  # Сохраняем луч, если EOS достигнут
            
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            
            # Проверяем, является ли последний токен лучшего луча EOS
            if beams[0][0][0, -1].item() == self.eos_token_id:
                break
        
        return tokenizer.decode(beams[0][0][0].tolist())