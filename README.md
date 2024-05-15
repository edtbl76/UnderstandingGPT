# Understanding GPT

This repository contains a simple implementation of a GPT (Generative Pre-trained Transformer) model using PyTorch. The GPT model is designed for natural language processing tasks such as text generation. In this project, we create a simplified version of GPT, train it on a small dataset, and generate text based on a given prompt.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Generating Text](#generating-text)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The GPT model is a type of transformer model used for generating human-like text. It has applications in various NLP tasks, such as text completion, translation, and summarization. This implementation demonstrates the basic structure and functionality of a GPT model.

## Features

- Custom dataset handling for text inputs
- Simplified GPT architecture with transformer blocks
- Training loop with loss calculation and optimization
- Text generation from a trained model

## Installation

To run this code, you need to have Python 3.6 or higher installed. You also need the following libraries:

- torch
- transformers

You can install these libraries using pip:

```bash
pip install torch transformers
```

## Usage

1. Clone this repository:

```bash
git clone https://github.com/yourusername/simple-gpt.git
cd simple-gpt
```

2. Run the script:

```bash
python understanding-gpt.py
```

## Model Architecture

The model consists of the following components:

- **SimpleDataset**: A custom dataset class to handle text inputs and tokenization.
- **GPTBlock**: A single transformer block that includes multi-head self-attention and a feed-forward neural network.
- **SimpleGPT**: The main GPT model class that stacks multiple GPTBlocks and includes token and position embeddings.

## Training the Model

The `train` function handles the training process, including the forward pass, loss calculation, backpropagation, and optimization.

```python
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
```

## Generating Text

The `generate_text` function allows you to generate text from a trained model given a prompt.

```python
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
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to modify and expand this README to suit your project's needs.