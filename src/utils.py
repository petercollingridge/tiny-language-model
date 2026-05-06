import os
import torch

from tokeniser import Tokeniser


LEARNING_RATE = 1e-3


def get_text(folder, filename):
    print(folder, filename)
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


def get_random_subseqs(seqs, batch_size=8, context_size=None):
    """
    Given a list of vectors, return a function that returns a random subsequence of a random
    vectors as a tensor, shifted by one for the targets.
    T: time steps, i.e. length of each sequence - 1
    """

    seqs = torch.tensor(seqs, dtype=torch.long) # (B, T)

    def get_batch():
        seq_indices = torch.randint(0, len(seqs), (batch_size,))
        start_positions = torch.randint(0, seqs.size(1) - context_size, (batch_size,))
        x = torch.stack([seqs[i, j:j+context_size] for i, j in zip(seq_indices, start_positions)])
        y = torch.stack([seqs[i, j+1:j+context_size+1] for i, j in zip(seq_indices, start_positions)])
        return x, y
    return get_batch


def get_random_subseq(seqs, context_size):
    """
    Given a list of vectors, return a function that returns a random subsequence of a random
    vector as a tensor, shifted by one for the targets.
    T: time steps, i.e. length of each sequence - 1
    """

    seqs = torch.tensor(seqs, dtype=torch.long) # (num_sequences, sequence_length)

    def get_example():
        seq_index = torch.randint(0, len(seqs), (1,)).item()
        start_position = torch.randint(0, seqs.size(1) - context_size, (1,)).item()
        x = seqs[seq_index, start_position:start_position+context_size]
        y = seqs[seq_index, start_position+1:start_position+context_size+1]
        return x, y
    return get_example


def get_tokeniser(folder, suffix = None, tokeniser=Tokeniser):
    filename = "training_sentences" if suffix is None else f"training_sentences_{suffix}"
    text = get_text(folder, filename + '.txt')
    seqs = get_seqs(text)
    tokeniser = tokeniser(seqs)

    return seqs, tokeniser

def parse_model_data(filepath):
    """
    Given a filepath to a text file of model data, parse the tokens and weights and return as a dictionary.
    """

    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    tokens = []
    weights = []
    flag = None

    for line in text.splitlines():
        if line.startswith("tokens:"):
            flag = "tokens"
            tokens = line.split(':')[1].strip().split('|')
        elif line.startswith("weights:"):
            flag = "weights"
            weights.append([])
        elif line.strip() and flag == "weights":
            weight = [float(x) for x in line.strip().split(",")]
            weights[-1].append(weight)
        else:
            print(line)

    return { 'tokens': tokens, 'weights': weights }


def _get_model_class(model_type):
    try:
        from BigramModel import BigramModel, DeeperBigramModel, BigramModelWithPositionalEncoding
        from AttentionModel import SimpleAttentionModel, AttentionModel
    except ModuleNotFoundError:
        from src.BigramModel import BigramModel, DeeperBigramModel, BigramModelWithPositionalEncoding
        from src.AttentionModel import SimpleAttentionModel, AttentionModel

    model_classes = {
        "BigramModel": BigramModel,
        "DeeperBigramModel": DeeperBigramModel,
        "BigramModelWithPositionalEncoding": BigramModelWithPositionalEncoding,
        "SimpleAttentionModel": SimpleAttentionModel,
        "AttentionModel": AttentionModel,
    }

    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")

    return model_classes[model_type]


def save_checkpoint(filepath, model, tokeniser, model_type, init_kwargs=None, build_kwargs=None, steps=None):
    """
    Save a PyTorch checkpoint with enough metadata to reconstruct the model exactly.
    """

    checkpoint = {
        "steps": steps,
        "model_type": model_type,
        "init_kwargs": init_kwargs or {},
        "build_kwargs": build_kwargs or {},
        "token_vocab": tokeniser.vocab,
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, map_location="cpu"):
    """
    Load a model checkpoint created by save_checkpoint and reconstruct the model.
    """

    checkpoint = torch.load(filepath, map_location=map_location)

    model_class = _get_model_class(checkpoint["model_type"])
    model = model_class(**checkpoint.get("init_kwargs", {}))

    build_kwargs = checkpoint.get("build_kwargs", {})
    if build_kwargs:
        model.build(**build_kwargs)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return {
        "model": model,
        "tokens": checkpoint.get("token_vocab", []),
        "steps": checkpoint.get("steps"),
        "model_type": checkpoint["model_type"],
        "init_kwargs": checkpoint.get("init_kwargs", {}),
        "build_kwargs": build_kwargs,
    }


def run_model(model, get_batch, steps=10000):
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    inputs, targets = get_batch()
    print_steps = int(steps / 10) or 1

    for step in range(steps):
        # Get a batch of data
        inputs, targets = get_batch()

        # Evaluate the loss
        logits, loss = model(inputs, targets)

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % print_steps == 0:
            print(f"Step {step}, loss {loss.item():.4f}")


def generate_text(model, tokeniser, n = 5, max_new_tokens=20):
    """ Given a trained model and tokeniser, generate n sequences of text """

    sequences = []
    for _ in range(n):
        # Start with the first token, which should be <BR>
        # Dimensions are (B, T) where B=1 and T=1, i.e. a batch of one sequence with one token
        # We use a batch of one sequence to be consistent with the model's expected input shape.
        first_token = torch.zeros((1, 1), dtype=torch.long)
        tokens = model.generate(first_token, max_new_tokens=max_new_tokens)

        # Tokens is a (1, T) array of token indices, so get first item to reduce to list
        output = tokeniser.decode(tokens[0].tolist())
        sequences.append(output)

    return sequences


def get_filepath(folder, filename, suffix=None):
    filename = filename if suffix is None else f"{filename}_{suffix}"
    return os.path.join(folder, filename + ".txt")


def write_output(filepath, output):
    """ Write output string to a file in the given folder """

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(output)


def save_model(filepath, steps, model, tokeniser):
    """ Save the model's output embeddings and the tokeniser's vocabulary to a file. """

    output = f"steps: {steps}\n"
    output += "tokens: " + "|".join(tokeniser.vocab) + "\n"

    for weights in model.output_weights():
        output += "weights:\n"
        for row in weights:
            # If row is a 0-dim tensor, convert to scalar string.
            # If row is a 1-d tensor, convert to comma-separated string.
            if row.dim() == 0:
                output += f"{row.item():.3f}"
            else:
                output += ",".join(f"{value.item():.3f}" for value in row)
            output += "\n"

    write_output(filepath, output)
