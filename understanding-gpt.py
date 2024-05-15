#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2Model

import numpy as np

# Dataset Preparation
class SimpleDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze()

def main():
    texts = ["Hello, how are you?", "I am fine, thank you.", "What about you?"]
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    dataset = SimpleDataset(texts, tokenizer, max_length=10)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Model Definition
    class SimpleGPT(nn.Module):
        def __init__(self, vocab_size, max_length):
            super(SimpleGPT, self).__init__()
            self.gpt2 = GPT2Model.from_pretrained('gpt2')
            self.linear = nn.Linear(self.gpt2.config.hidden_size, vocab_size)

        def forward(self, input_ids, attention_mask):
            outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
            logits = self.linear(outputs.last_hidden_state)
            return logits

    model = SimpleGPT(vocab_size=len(tokenizer), max_length=10).to('cpu')

    # Training Loop
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    def train(model, dataloader, optimizer, criterion, epochs=5):
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for input_ids, attention_mask in dataloader:
                input_ids, attention_mask = input_ids.to('cpu'), attention_mask.to('cpu')
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                shift_logits = outputs[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

    train(model, dataloader, optimizer, criterion)

    # Text Generation
    def generate_text(model, tokenizer, prompt, max_length=50):
        model.eval()
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cpu')
        generated = input_ids

        for _ in range(max_length):
            outputs = model(input_ids=generated, attention_mask=(generated != tokenizer.pad_token_id))
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            generated = torch.cat((generated, next_token), dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        return generated_text

    prompt = "Once upon a time"
    generated_text = generate_text(model, tokenizer, prompt)
    print(generated_text)

if __name__ == "__main__":
    main()
