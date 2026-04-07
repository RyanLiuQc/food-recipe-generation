# lstm
import torch.nn as nn
import torch


class LSTMGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.2):
        super(LSTMGenerator, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 2. LSTM Layer (batch_first=True, as noted in your A3 TODO!)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # 3. Fully Connected Layer
        # INSTEAD of outputting 1 value (binary classification),
        # it outputs 'vocab_size' values (probabilities for the next token)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        # x shape: (batch_size, seq_length)
        embeds = self.embedding(x)

        # lstm_out shape: (batch_size, seq_length, hidden_dim)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # We don't just take the last hidden state like in classification.
        # We pass EVERY time step through the linear layer.
        logits = self.fc(lstm_out)

        return logits, hidden

    def init_hidden(self, batch_size, device):
        # Initializing hidden layers as you did in init_hidden
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device))