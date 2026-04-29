from pathlib import Path
import sys

from BigramModel import BigramModel, DeeperBigramModel
from AttentionModel import AttentionModel, SimpleAttentionModel
from utils import generate_text, get_filepath, get_text, get_seqs, get_random_seqs, get_random_subseqs, run_model, write_output, save_model
from tokeniser import Tokeniser, SimpleWordTokeniser


BASE_DIR = Path(__file__).resolve().parent


def get_example_folder(example_name):
    folder_name = example_name.split("_")
    suffix = folder_name[1] if len(folder_name) > 1 else None
    return str((BASE_DIR / folder_name[0]).resolve()), suffix


def get_tokeniser(folder, suffix = None, tokeniser=Tokeniser):
    filename = "training_sentences" if suffix is None else f"training_sentences_{suffix}"
    text = get_text(folder, filename + '.txt')
    seqs = get_seqs(text)
    tokeniser = tokeniser(seqs)

    return seqs, tokeniser


def get_batching_func(tokeniser, seqs):
    # Function to get batches of training data
    encoded_seqs = [tokeniser.encode(seq) for seq in seqs]
    get_batch = get_random_seqs(encoded_seqs, batch_size=8)
    return get_batch


def output_generated_text(filepath, tokeniser, model, n=20, seqs=None):
    """
    Generate n sequences of text using the trained model and save to a file.
    """

    max_new_tokens = min(20, tokeniser.block_size)
    sequences = generate_text(model, tokeniser, n = n, max_new_tokens = max_new_tokens)

    if seqs is not None:
        sequences = [f"{'  ' if seq in seqs else 'x '} {seq}" for seq in sequences]

    output = "\n".join(sequences)   
    write_output(filepath, output)


def run_example(example_name, model, steps=10000):
    """
    Run an example of training a bigram model on two sentences.
    """

    folder, suffix = get_example_folder(example_name)
    seqs, tokeniser = get_tokeniser(folder, suffix, SimpleWordTokeniser)

    model.build(tokeniser.vocab_size)
    get_batch = get_batching_func(tokeniser, seqs)

    run_model(model, get_batch, steps=steps)

    model_filepath = get_filepath(folder, "model_output", suffix)
    save_model(model_filepath, steps, model, tokeniser)

    sentences_filepath = get_filepath(folder, "generated_sentences", suffix)
    output_generated_text(sentences_filepath, tokeniser, model)


def run_example_with_double_tokens(example_name, model, steps=10000):
    """
    Run an example of training a bigram model on two sentences.
    """

    folder, suffix = get_example_folder(example_name)
    seqs, tokeniser = get_tokeniser(folder, suffix, SimpleWordTokeniser)
    print(seqs)


def attention_example(example_name, embed_dim=2, context_size=None, steps=10000):
    folder, suffix = get_example_folder(example_name)
    seqs, tokeniser = get_tokeniser(folder, suffix, SimpleWordTokeniser)

    if context_size is None:
        context_size = tokeniser.block_size

    encoded_seqs = [tokeniser.encode(seq) for seq in seqs]
    get_example = get_random_subseqs(encoded_seqs, batch_size=8, context_size=context_size)
    print(get_example())

    head_dim = embed_dim
    model = SimpleAttentionModel(tokeniser.vocab_size, context_size, embed_dim, head_dim=head_dim)
    run_model(model, get_example, steps=steps)

    model_filepath = get_filepath(folder, "model_output", suffix)
    save_model(model_filepath, steps, model, tokeniser)

    sentences_filepath = get_filepath(folder, "generated_sentences", suffix)
    output_generated_text(sentences_filepath, tokeniser, model, n=30, seqs=seqs)


def test_model(example_name, embed_dim=2, context_size=None):
    """ Create a model and test it without training"""
    folder, suffix = get_example_folder(example_name)
    seqs, tokeniser = get_tokeniser(folder, suffix, SimpleWordTokeniser)

    if context_size is None:
        context_size = tokeniser.block_size

    print(embed_dim, context_size)

    encoded_seqs = [tokeniser.encode(seq) for seq in seqs]
    get_example = get_random_subseqs(encoded_seqs, batch_size=8, context_size=context_size)

    head_dim = embed_dim
    model = SimpleAttentionModel(tokeniser.vocab_size, context_size, embed_dim, head_dim=head_dim)

    print("Model token embedding weights:")
    print(model.token_embedding_table.weight)
    print("Model position embedding weights:")
    print(model.position_embedding_table.weight)

    generate_text(model, tokeniser, 1, max_new_tokens = 4)


def example1(example_name):
    """
    Training a bigram model on two sentences.
    """

    model = BigramModel()
    run_example(example_name, model)


def example2(example_name):
    """
    Training a bigram model with a hidden layer on two sentences.
    """

    model = DeeperBigramModel()
    run_example(example_name, model, steps=10000)


def example3(example_name):
    """
    Same as example 2, but with more words.
    """

    model = DeeperBigramModel(2)
    run_example(example_name, model, steps=10000)


def example4(example_name):
    """
    Introducing attention
    """

    # model = DeeperBigramModel(2)
    # run_example_with_double_tokens(example_name, model, steps=10000)
    # attention_example(example_name, embed_dim=2, context_size=3, steps=10000)
    test_model(example_name, embed_dim=2, context_size=3)


def example5(example_name):
    """
    Introducing attention
    """

    attention_example(example_name, embed_dim=4, steps=10000)


examples = {
    'example1': example1,
    'example2': example2,
    'example3': example3,
    'example4': example4,
    'example5': example5,
}

if __name__ == "__main__":
    # Get value from command line argument to decide which example to run
    if len(sys.argv) > 1:
        example_name = sys.argv[1]
        folder = example_name.split("_")[0]
        if folder in examples:
            examples[folder](example_name)
        else:
            print(f"Unknown example: {example_name}")
    else:
        print("Please provide an example name as a command line argument.")
