import torch
from model import GeneratorTransformer
from tokenizers import Tokenizer

def chat():
    """
    Интерактивный интерфейс для тестирования генерации текста.
    """
    tokenizer = Tokenizer.from_file("results/tokenizer.json")
    model = GeneratorTransformer.load_from_checkpoint("results/checkpoint.pt")
    model.eval()
    
    print("Введите 'quit' для выхода.")
    while True:
        user_input = input("Вы: ")
        if user_input.lower() == 'quit':
            break
        
        # Генерация с обычным сэмплированием
        response = model.generate(user_input, tokenizer, context_len=50, temperature=0.8, max_out_tokens=200)
        print(f"Бот (обычная генерация): {response}")
        
        # Генерация с beam search
        response_beam = model.generate(user_input, tokenizer, context_len=50, temperature=0.8, max_out_tokens=200, beam_size=3)
        print(f"Бот (beam search): {response_beam}")

if __name__ == "__main__":
    chat()