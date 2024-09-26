# -*- coding: utf-8 -*-
"""DL_assignment2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1D3Xb-i8ub0UR0fzXUO6b_rhKSysNV9MB

#Importing Libraries
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import math
import numpy as np
import os
from collections import Counter
import tarfile
import shutil
import tempfile
import urllib.request
import math
import random
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir='data/'
url = 'http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz'
filename = 'simple-examples.tgz'
filepath = os.path.join(tempfile.gettempdir(), filename)

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Download the dataset
if not os.path.exists(filepath):
    print("Downloading PTB dataset...")
    urllib.request.urlretrieve(url, filepath)
    print("Download complete.")
else:
    print("PTB dataset already downloaded.")

# Extract the dataset
extracted_path = os.path.join(tempfile.gettempdir(), 'simple-examples')
if not os.path.exists(extracted_path):
    print("Extracting PTB dataset...")
    with tarfile.open(filepath, 'r:gz') as tar:
        tar.extractall(path=tempfile.gettempdir())
    print("Extraction complete.")
else:
    print("PTB dataset already extracted.")

# Move the necessary files to the data_dir
for split in ['train', 'valid', 'test']:
    src = os.path.join(extracted_path, 'data', f'ptb.{split}.txt')
    dst = os.path.join(data_dir, f'ptb.{split}.txt')
    if not os.path.exists(dst):
        shutil.copy(src, dst)
        print(f"Copied {split} data to {dst}")
    else:
        print(f"{split} data already exists at {dst}")

"""#Data Preprocessing"""

def preprocess_data(file_path):
    tokens = []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            line_tokens = line.strip().split()
            line_tokens.append('<eos>')
            tokens.extend(line_tokens)
    print(f"Total tokens: {len(tokens)}")
    print(tokens[:10])
    return tokens

# Paths to Penn Tree Bank dataset files
train_file_path = 'data/ptb.train.txt'
valid_file_path = 'data/ptb.valid.txt'
test_file_path = 'data/ptb.test.txt'

train_tokens = preprocess_data(train_file_path)
valid_tokens = preprocess_data(valid_file_path)
test_tokens = preprocess_data(valid_file_path)

def build_vocab(token_list):
    counter = Counter(token_list)
    vocab = sorted(counter, key=counter.get, reverse=True)

    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    vocab_size = len(word2idx)

    # Check vocab size
    print(f"Vocabulary built. Total words: {vocab_size}")
    return word2idx, idx2word, vocab_size


def encode_tokens(tokens, word2idx):
    encoded_tokens = [word2idx.get(token, word2idx["<unk>"]) for token in tokens]
    return encoded_tokens

word2idx, idx2word, vocab_size = build_vocab(train_tokens)

train_data = encode_tokens(train_tokens, word2idx)
valid_data = encode_tokens(valid_tokens, word2idx)
test_data = encode_tokens(test_tokens, word2idx)

print(train_data[:10])
print(valid_data[:10])
print(test_data[:10])

print("Max index in train_data:", max(train_data))
print("Max index in valid_data:", max(valid_data))
print("Max index in test_data:", max(test_data))
print("Vocab size:", vocab_size)

def batchify(data, batch_size):
    nbatch = len(data) // batch_size
    data = data[:nbatch * batch_size]
    data = torch.tensor(data, dtype=torch.long).to(device)
    data = data.view(batch_size, -1)
    return data

batch_size = 20
train_data = batchify(train_data, batch_size)
valid_data = batchify(valid_data, batch_size)
test_data = batchify(test_data, batch_size)

print(train_data.shape)
print(valid_data.shape)
print(test_data.shape)

"""#Model Architecture"""

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size=200, hidden_size=200, num_layers=2, dropout=0.0):
        super(LSTMModel, self).__init__()
        self.encoder = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()

    def forward(self, input, hidden):
        assert (input >= 0).all() and (input < vocab_size).all(), "Input indices out of bounds!"
        emb = self.dropout(self.encoder(input))
        output, hidden = self.lstm(emb, hidden)
        output = self.dropout(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, batch_size, self.hidden_size),
                weight.new_zeros(self.num_layers, batch_size, self.hidden_size))



class GRUModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.0):
        super(GRUModel, self).__init__()
        self.encoder = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, dropout=dropout)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()

    def forward(self, input, hidden):
        assert (input >= 0).all() and (input < vocab_size).all(), "Input indices out of bounds!"
        emb = self.dropout(self.encoder(input))
        output, hidden = self.gru(emb, hidden)
        output = self.dropout(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return weight.new_zeros(self.num_layers, batch_size, self.hidden_size)

"""#Training and Evaluation Loops"""

seq_len = 20  # Set sequence length to 32
batch_size = 20

def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i, seq_len, batch_size):
    data = [source[j, i * seq_len:(i + 1) * seq_len] for j in range(batch_size)]
    data = torch.stack(data, dim=1)  # Shape: [seq_len, batch_size]
    target = [source[j, i * seq_len + 1:(i + 1) * seq_len + 1] for j in range(batch_size)]
    target = torch.stack(target, dim=1)  # Shape: [seq_len, batch_size]
    return data, target

def train_epoch(model, train_dataset, optimizer, loss_fn, batch_size, seq_len, device):
    print("entered training loop")
    model.train()
    total_loss = 0
    hidden = model.init_hidden(batch_size)  # Initialize hidden state

    # Calculate the number of batches
    num_batches = train_dataset.size(1) // seq_len
    print(f"Number of batches: {num_batches}")

    for i in tqdm(range(num_batches), total=num_batches, desc="Training"):
        inputs, targets = get_batch(train_dataset, i, seq_len, batch_size=batch_size)
        inputs, targets = inputs.to(device), targets.to(device)
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        # Forward pass
        outputs, hidden = model(inputs, hidden)
        assert outputs.size(2) == vocab_size, "Output dimensions do not match vocab size"

        # Reshape outputs to (batch_size * seq_len, vocab_size) for loss calculation
        loss = loss_fn(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Gradient clipping
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    perplexity = calculate_perplexity(avg_loss)
    return avg_loss, perplexity


def evaluate(model, data, loss_fn, batch_size, seq_len, device):
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(batch_size)  # Initialize hidden state

    # Calculate the number of batches
    num_batches = data.size(1) // seq_len

    with torch.no_grad():
        for i in tqdm(range(num_batches), total=num_batches, desc="Evaluating"):
            inputs, targets = get_batch(data, i, seq_len, batch_size=batch_size)
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = repackage_hidden(hidden)

            # Forward pass
            outputs, hidden = model(inputs, hidden)  # Shape: (batch_size, seq_len, vocab_size)
            assert outputs.size(2) == vocab_size, "Output dimensions do not match vocab size"

            # Reshape outputs to (batch_size * seq_len, vocab_size) for loss calculation
            loss = loss_fn(outputs.view(-1, vocab_size), targets.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    perplexity = calculate_perplexity(avg_loss)
    return avg_loss, perplexity

# # Training function
# def train(model, train_data, criterion, optimizer, seq_len):
#     model.train()
#     total_loss = 0.
#     hidden = model.init_hidden(batch_size)

#     for batch_idx in range(train_data.size(1) // seq_len):
#         data, targets = get_batch(train_data, batch_idx, seq_len)
#         data, targets = data.to(device), targets.to(device)
#         hidden = repackage_hidden(hidden)
#         optimizer.zero_grad()
#         output, hidden = model(data, hidden)
#         loss = criterion(output.view(-1, vocab_size), targets.view(-1))
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
#         optimizer.step()
#         total_loss += loss.item()

#     return total_loss / (train_data.size(1) // seq_len)

# # Evaluation function
# def evaluate(model, data_source, criterion, seq_len):
#     model.eval()
#     total_loss = 0.
#     hidden = model.init_hidden(batch_size)

#     with torch.no_grad():
#         for batch_idx in range(data_source.size(1) // seq_len):
#             data, targets = get_batch(data_source, batch_idx, seq_len)
#             data, targets = data.to(device), targets.to(device)
#             hidden = repackage_hidden(hidden)
#             output, hidden = model(data, hidden)
#             loss = criterion(output.view(-1, vocab_size), targets.view(-1))
#             total_loss += loss.item()

#     return total_loss / (data_source.size(1) // seq_len)

def calculate_perplexity(loss):
    """
    Calculates perplexity from loss.
    """
    return math.exp(loss)

def plot_perplexity(train_ppls, valid_ppls, title, save_path):
    """
    Plots training and validation perplexity over epochs.
    """
    epochs = range(1, len(train_ppls) + 1)
    plt.figure(figsize=(10,6))
    plt.plot(epochs, train_ppls, 'b-', label='Training Perplexity')
    plt.plot(epochs, valid_ppls, 'r-', label='Test Perplexity')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

# Set parameters and initialize model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# vocab_size = len(vocab)
# embed_size = 200
# hidden_size = 200
# num_layers = 2
# seq_len = 35  # Set sequence length to 32
# batch_size = 20
# dropout = 0.0  # No dropout
# learning_rate = 4.0
# num_epochs = 30  # Set epochs to 30
# weight_decay = 1e-5  # Add weight decay

def run_experiments():
    """
    Runs the four experimental settings and summarizes the results.
    """
    # Define experimental settings
    experiments = [
        {
            'model_type': 'lstm',
            'dropout': 0.0,
            'lr': 1.6,    # Initial learning rate for SGD
            'epochs': 13,
            'save_path': 'checkpoints/lstm_no_dropout',
            'plot_path': 'plots/lstm_no_dropout_perplexity.png'
        },
        {
            'model_type': 'lstm',
            'dropout': 0.5,
            'lr': 3.4,    # Initial learning rate for SGD
            'epochs': 20,
            'save_path': 'checkpoints/lstm_with_dropout',
            'plot_path': 'plots/lstm_with_dropout_perplexity.png'
        },
        {
            'model_type': 'gru',
            'dropout': 0.0,
            'lr': 1.5,    # Initial learning rate for SGD
            'epochs': 13,
            'save_path': 'checkpoints/gru_no_dropout',
            'plot_path': 'plots/gru_no_dropout_perplexity.png'
        }
        {
            'model_type': 'gru',
            'dropout': 0.3,
            'lr': 1.8,    # Initial learning rate for SGD
            'epochs': 20,
            'save_path': 'checkpoints/gru_with_dropout',
            'plot_path': 'plots/gru_with_dropout_perplexity.png'
        }
    ]

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nUsing device: {device}\n')

    # Initialize a list to store results
    results = []

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    for exp in experiments:

        os.makedirs(os.path.dirname(exp['save_path']), exist_ok=True)
        os.makedirs(os.path.dirname(exp['plot_path']), exist_ok=True)

        print(f"\n--- Starting Experiment: {exp['model_type'].upper()} with Dropout={exp['dropout']} ---")

        # Initialize model based on the experiment
        if exp['model_type'] == 'lstm':
            model = LSTMModel(vocab_size, embed_size=200, hidden_size=200, num_layers=2, dropout=exp['dropout']).to(device)
        elif exp['model_type'] == 'gru':
            model = GRUModel(vocab_size, embed_size=200, hidden_size=200, num_layers=2, dropout=exp['dropout']).to(device)
        else:
            raise ValueError("Model type must be 'lstm' or 'gru'")

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=exp['lr'])  # Using SGD optimizer

        # Define a lambda function to adjust the learning rate
        if exp['dropout'] == 0.0:
            lambda_lr = lambda epoch: exp['lr'] if epoch < 7 else exp['lr'] * (0.5 ** (epoch - 6))
        else:
            lambda_lr = lambda epoch: exp['lr'] if epoch < 8 else exp['lr'] * (0.75 ** (epoch - 7))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

        # Lists to store perplexities per epoch
        train_ppls, valid_ppls, test_ppls = [], [], []
        best_valid_ppl = float('inf')

        # Training and validation loop
        for epoch in range(exp['epochs']):
            print(f'\nEpoch {epoch + 1}/{exp["epochs"]}')

            # Train the model for one epoch
            train_loss, train_ppl = train_epoch(
                model=model,
                train_dataset=train_data,
                optimizer=optimizer,
                loss_fn=criterion,
                batch_size=batch_size,
                seq_len=seq_len,
                device=device
            )
            train_ppls.append(train_ppl)
            print(f'Train Perplexity: {train_ppl:.2f}')

            # Evaluate on validation set
            valid_loss, valid_ppl = evaluate(
                model=model,
                data=valid_data,
                loss_fn=criterion,
                batch_size=batch_size,
                seq_len=seq_len,
                device=device
            )
            valid_ppls.append(valid_ppl)
            print(f'Validation Perplexity: {valid_ppl:.2f}')

            # Test the model on the test set
            test_loss, test_ppl = evaluate(
                model=model,
                data=test_data,
                loss_fn=criterion,
                batch_size=batch_size,
                seq_len=seq_len,
                device=device
            )
            test_ppls.append(test_ppl)
            print(f'Test Perplexity: {test_ppl:.2f}')

            # Save the model if validation perplexity improves
            if valid_ppl < best_valid_ppl:
                best_valid_ppl = valid_ppl
                torch.save(model.state_dict(), exp['save_path'])
                print(f'Best model saved to {exp["save_path"]}')

            # Adjust learning rate using the scheduler
            if scheduler:
                scheduler.step()

        # Plot perplexities for this experiment
        plot_title = f"{exp['model_type'].upper()} with Dropout={exp['dropout']}"
        plot_path = exp['plot_path']
        plot_perplexity(train_ppls, test_ppls, plot_title, plot_path)

        # Store the results
        results.append({
            'Model': exp['model_type'].upper(),
            'Dropout': exp['dropout'],
            'Train Perplexity': min(train_ppls),
            'Validation Perplexity': min(valid_ppls),
            'Test Perplexity': min(test_ppls)
        })

    # Display Results Summary
    print("\n=== Results Summary ===")
    print("| Model | Dropout Probability | Train Perplexity | Validation Perplexity | Test Perplexity |")
    print("|-------|---------------------|------------------|-----------------------|-----------------|")
    for res in results:
        print(f"| {res['Model']} | {res['Dropout']} | {res['Train Perplexity']:.2f} | {res['Validation Perplexity']:.2f} | {res['Test Perplexity']:.2f} |")

"""#Running Experiments"""

if __name__ == "__main__":
    run_experiments()