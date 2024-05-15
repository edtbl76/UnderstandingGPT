#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2Config


class SimpleDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True,
                                  max_length=self.max_length)
        return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze()


class GPTBlock(nn.Module):
    def __init__(self, config):
        super(GPTBlock, self).__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = nn.MultiheadAttention(config.n_embd, config.n_head, dropout=config.attn_pdrop)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop)
        )

    def forward(self, x, attention_mask=None):
        attn_output, _ = self.attn(x, x, x, attn_mask=attention_mask)
        x = x + attn_output
        x = self.ln_1(x)
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.ln_2(x)
        return x


class SimpleGPT(nn.Module):
    def __init__(self, config):
        super(SimpleGPT, self).__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.config = config

    def forward(self, input_ids, attention_mask=None):
        positions = torch.arange(0, input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.drop(x)

        # Adjusting attention mask for multi-head attention
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).repeat(self.config.n_head, attention_mask.size(1), 1)
            attention_mask = attention_mask.to(dtype=torch.float32)
            attention_mask = (1.0 - attention_mask) * -10000.0

        for block in self.blocks:
            x = block(x.transpose(0, 1), attention_mask)  # Transpose for multi-head attention
            x = x.transpose(0, 1)  # Transpose back to original shape

        x = self.ln_f(x)
        logits = self.head(x)
        return logits


def train(model, dataloader, optimizer, criterion, epochs=5, device='cuda'):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for input_ids, attention_mask in dataloader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")


def generate_text(model, tokenizer, prompt, max_length=50, device='cuda'):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated = input_ids

    for _ in range(max_length):
        outputs = model(generated)
        next_token_logits = outputs[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        generated = torch.cat((generated, next_token), dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break

    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return generated_text


def main():
    texts = ["Hello, how are you?", "I am fine, thank you.", "What about you?"]
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=20,
        n_embd=768,
        n_layer=12,
        n_head=12,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1
    )

    dataset = SimpleDataset(texts, tokenizer, max_length=20)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = SimpleGPT(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    train(model, dataloader, optimizer, criterion, epochs=5, device=device)

    prompt = "Once upon a time"
    generated_text = generate_text(model, tokenizer, prompt, device=device)
    print(generated_text)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
