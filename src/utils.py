import os
import torch


LEARNING_RATE = 1e-3


def get_text(folder, filename):
    with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def get_seqs(text):
    """ Given a block of text, return a list of sequences (lines) """

    seqs = [line.strip() for line in text.splitlines() if line.strip()]
    return seqs


def get_all_seqs_old(seqs):
    """
    Given a list of vectors, return a function that returns all the vectors as tensors,
    shifted by one for the targets
    B: batch size, i.e. number of sequences
    T: time steps, i.e. length of each sequence - 1
    """

    def get_batch():
        x = torch.stack([torch.tensor(seq[:-1], dtype=torch.long) for seq in seqs])  # (B, T)
        y = torch.stack([torch.tensor(seq[1:], dtype=torch.long) for seq in seqs])  # (B, T)
        return x, y
    return get_batch


def get_all_seqs(seqs):
    """
    Given a list of vectors, return a function that returns all the vectors as tensors,
    shifted by one for the targets
    B: batch size, i.e. number of sequences
    T: time steps, i.e. length of each sequence - 1
    """

    seqs = torch.tensor(seqs, dtype=torch.long) # (B, T)

    def get_batch():
        x = seqs[:, :-1]  # (B, T-1) - all sequences, excluding last token
        y = seqs[:, 1:]   # (B, T-1) - all sequences, excluding first token
        return x, y
    return get_batch


def get_random_seqs(seqs, batch_size):
    """
    Given a list of vectors, return a function that returns a random subset of the vectors as
    tensors, shifted by one for the targets.
    B: batch size, i.e. number of sequences
    T: time steps, i.e. length of each sequence - 1
    """

    seqs = torch.tensor(seqs, dtype=torch.long) # (B, T)

    def get_batch():
        indices = torch.randint(0, len(seqs), (batch_size,))
        x = seqs[indices, :-1]  # (B, T-1) - random subset of sequences, excluding last token
        y = seqs[indices, 1:]   # (B, T-1) - random subset of sequences, excluding first token
        return x, y
    return get_batch


def run_model(model, get_batch, steps=10000):
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for step in range(steps):
        # Get a batch of data
        inputs, targets = get_batch()

        # Evaluate the loss
        logits, loss = model(inputs, targets)

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            print(f"Step {step}, loss {loss.item():.4f}")


def generate_text(model, tokeniser, n = 5, max_new_tokens=20):
    """ Given a trained model and tokeniser, generate n sequences of text """

    for _ in range(n):
        # Start with the first token, which should be <BOS>
        first_token = torch.zeros((1, 1), dtype=torch.long)
        tokens = model.generate(first_token, max_new_tokens=max_new_tokens)

        # Tokens is a (1, T) array of token indices, so get first item to reduce to list
        output = tokeniser.decode(tokens[0].tolist())
        print(output)


def write_output(folder, filename, output):
    """ Write output string to a file in the given folder """

    with open(os.path.join(folder, filename), 'w', encoding='utf-8') as f:
        f.write(output)