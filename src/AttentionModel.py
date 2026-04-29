import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# B: batch - batch size
# T: time - context_len
# C: channel - embed_dim or head_dim

class SingleHeadSelfAttention(nn.Module):
    """ One head of self-attention """

    def __init__(self, embed_dim, head_dim, context_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim

        self.key = nn.Linear(embed_dim, head_dim, bias=False)
        self.query = nn.Linear(embed_dim, head_dim, bias=False)
        self.value = nn.Linear(embed_dim, head_dim, bias=False)
        self.scale = 1.0 / math.sqrt(head_dim)

        # tril: lower triangular mask (T, T) with -inf on future positions
        mask = torch.tril(torch.ones(context_len, context_len)).unsqueeze(0)  # (1, T, T)
        self.register_buffer("mask", mask)

    def forward(self, x):
        """
        x: (B, T, C)
        Returns: same shape as input, with head_dim as the final dimension
        """

        B, T, C = x.shape

        k = self.key(x)    # (B, context_len, head_dim)
        q = self.query(x)  # (B, context_len, head_dim)
        v = self.value(x)  # (B, context_len, head_dim)

        # Compute attention scores
        # (B, context_len, head_dim) * (B, head_dim, context_len) -> (B, context_len, context_len)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask: set -inf to future positions
        mask = self.mask[:, :T, :T]  # (1, T, T)
        attn_logits = attn_logits.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn_logits, dim=-1)  # (B, T, T)

        out = torch.matmul(attn, v)  # (B, T, head_dim)
        return out


class SimpleAttentionModel(nn.Module):
    """ A simple language model with one head of self-attention and no positional embeddings."""
    def __init__(self, vocab_size, context_len, embed_dim, head_dim):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding_table = nn.Embedding(context_len, embed_dim)
        # For now don't use head_dim, just set it equal to embed_dim and skip the projection back to embed_dim after attention
        self.self_attention_head = SingleHeadSelfAttention(embed_dim, embed_dim, context_len)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are (B, T) tensors of token indices
        # B: batch size, T: number of tokens in idx
        B, T = idx.shape

        # Token embeddings
        # x = self.token_embedding_table(idx)  # (B, T, embed_dim)

        # Token and position embeddings
        token_emb = self.token_embedding_table(idx)  # (B, T, embed_dim)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, embed_dim)
        x = token_emb + pos_emb  # (B, T, embed_dim)

        print("idx", idx)
        # print("token_emb", token_emb)
        # print("T", T)
        # print("arange", torch.arange(T, device=idx.device))
        # print("pos_emb", pos_emb)
        print("x", x)

        # Apply self-attention, project back to embed_dim, then add residually.
        attn_out = self.self_attention_head(x)  # (B, T, head_dim)
        x = x + attn_out

        # Project back to vocab size
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, tokens, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        context_size = self.self_attention_head.mask.shape[-1]

        for _ in range(max_new_tokens):
            # Truncate tokens to the last context_size tokens
            tokens_trunc = tokens[:, -context_size:]

            # get the predictions
            logits, loss = self(tokens_trunc)

            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, head_dim)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, head_dim)

            # sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1) # (B, 1)

            # Append sampled index to the running sequence
            tokens = torch.cat((tokens, next_token), dim=1) # (B, T+1)

        # Tokens is a (1, T) array of token indices, so get first item to reduce to list
        return tokens

    
    def output_weights(self):
        """ Return all weights of of the model. """
        return [param.detach().cpu() for param in self.parameters()]


class AttentionModel(SimpleAttentionModel):
    def __init__(self, vocab_size, context_len, embed_dim, head_dim):
        super().__init__(vocab_size, context_len, embed_dim, head_dim)
        self.position_embedding_table = nn.Embedding(context_len, embed_dim)
        self.attention_output_proj = nn.Linear(head_dim, embed_dim)

    def forward(self, idx, targets=None):
        # idx and targets are (B, T) tensors of token indices
        B, T = idx.shape

        # Token embedding (B, T, embed_dim)
        token_emb = self.token_embedding_table(idx)

        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, embed_dim)
        x = token_emb + pos_emb  # (B, T, embed_dim)

        # Apply self-attention, project back to embed_dim, then add residually.
        attn_out = self.self_attention_head(x)  # (B, T, head_dim)
        attn_out = self.attention_output_proj(attn_out)  # (B, T, embed_dim)
        x = x + attn_out

        # Unproject back to vocab size
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
