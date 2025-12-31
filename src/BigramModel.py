import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    """ Simple bigram model that predicts the next token based on the current token only """

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.reshape(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, tokens, max_new_tokens=5):
        # tokens is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Get predictions
            logits, loss = self(tokens)

            # Get the logits for the last time step
            logits = logits[:, -1, :] # becomes (B, C)

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)

            # For each batch, get the next token by sampling from the distribution
            next_token = torch.multinomial(probs, num_samples=1) # (B, 1)

            # Append sampled token to the running sequence
            tokens = torch.cat((tokens, next_token), dim=1) # (B, T + 1)

        # Tokens is a (1, T) array of token indices, so get first item to reduce to list
        return tokens

    def output_embeddings(self):
        return self.token_embedding_table.weight  # (vocab_size, vocab_size)


class BigramLanguageModelWithPositionalEncoding(nn.Module):
    """ More advanced bigram model that adds positional encoding and token embeddings """

    def __init__(self, vocab_size, block_size, n_embed):
        super().__init__()

        # print(vocab_size, block_size, n_embed)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, tokens, targets=None):
        # B: batch size, T: sequence length
        B, T = tokens.shape

        # For each token in position t, get the token embedding and the position embedding
        # C: embedding dimension
        token_embeddings = self.token_embedding_table(tokens)  # (B, T, C)
        positions = torch.arange(T, device=tokens.device)  # List of positions from 0 to T-1
        position_embeddings = self.position_embedding_table(positions)  # (T, C)
        x = token_embeddings + position_embeddings  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T) if targets is not None else None
            loss = F.cross_entropy(logits, targets) if targets is not None else None
        return logits, loss

    def generate(self, tokens, max_new_tokens=10):
        # tokens is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Get predictions
            logits, loss = self(tokens)

            # Get the logits for the last time step
            logits = logits[:, -1, :] # becomes (B, C)

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)

            # For each batch, get the next token by sampling from the distribution
            next_token = torch.multinomial(probs, num_samples=1) # (B, 1)

            # Append sampled token to the running sequence
            tokens = torch.cat((tokens, next_token), dim=1) # (B, T + 1)

        return tokens
