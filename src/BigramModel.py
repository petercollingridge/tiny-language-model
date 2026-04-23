import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramModel(nn.Module):
    """
    Simple bigram model that predicts the next token based on the current token only.
    Consists of a single matrix of weights that map input nodes to output nodes.
    """

    def __init__(self):
        super().__init__()
        self.model = None

    def build(self, vocab_size):
        self.model = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.model(idx)  # (B, T, C)

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
        return self.model.weight.detach().cpu()  # (vocab_size, vocab_size)

    def output_weights(self):
        return [self.model.weight.detach().cpu()]  # (vocab_size, vocab_size)


class DeeperBigramModel(BigramModel):
    """
    Simple bigram model that predicts the next token based on the current token only.
    Has a single hidden later between the input and output nodes.
    """

    def __init__(self, hidden_size=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = 0

    def build(self, vocab_size):
        self.model = nn.Sequential(
            nn.Linear(vocab_size, self.hidden_size),  # token ids -> vectors
            nn.Linear(self.hidden_size, vocab_size)   # vectors -> logits over vocab
        )
        self.vocab_size = vocab_size

    def forward(self, input_values, targets=None):
        encoded_inputs = F.one_hot(input_values, num_classes=self.vocab_size)
        encoded_inputs = encoded_inputs.to(dtype=self.model[0].weight.dtype)
        logits = self.model(encoded_inputs)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),  # (B*T, vocab)
                targets.reshape(-1)                   # (B*T,)
            )

        return logits, loss

    def output_weights(self):
        """ Return all weights of of the model. """
        return [param.detach().cpu() for param in self.model.parameters()]


class BigramModelWithPositionalEncoding(nn.Module):
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
