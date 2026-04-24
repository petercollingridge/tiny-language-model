from pathlib import Path
import sys

from BigramModel import BigramModel, DeeperBigramModel
from utils import generate_text, get_filepath, get_text, get_seqs, get_random_seqs, run_model, write_output, save_model
from tokeniser import Tokeniser, SimpleWordTokeniser


BASE_DIR = Path(__file__).resolve().parent


def get_example_folder(folder):
    return str((BASE_DIR / folder).resolve())


def get_tokeniser(folder, suffix = None, tokeniser=Tokeniser):
    filename = "training_sentences.txt" if suffix is None else f"training_sentences_{suffix}.txt"
    text = get_text(folder, filename)
    seqs = get_seqs(text)
    tokeniser = tokeniser(seqs)
    print(tokeniser.vocab_size, tokeniser.block_size)

    return seqs, tokeniser


def get_batching_func(tokeniser, seqs):
    # Function to get batches of training data
    encoded_seqs = [tokeniser.encode(seq) for seq in seqs]
    get_batch = get_random_seqs(encoded_seqs, batch_size=8)
    return get_batch


def output_generated_text(filepath, tokeniser, model, n=20):
    """
    Generate n sequences of text using the trained model and save to a file.
    """

    max_new_tokens = min(20, tokeniser.block_size)
    sequences = generate_text(model, tokeniser, n = n, max_new_tokens = max_new_tokens)
    output = "\n".join(sequences)   
    write_output(filepath, output)


def run_example(example_name, model, steps=10000):
    """
    Run an example of training a bigram model on two sentences.
    """

    folder_name, suffix = example_name.split("_")
    folder = get_example_folder(folder_name)
    seqs, tokeniser = get_tokeniser(folder, suffix, SimpleWordTokeniser)
    model.build(tokeniser.vocab_size)
    get_batch = get_batching_func(tokeniser, seqs)

    run_model(model, get_batch, steps=steps)

    model_filepath = get_filepath(folder, "model_output", suffix)
    save_model(model_filepath, steps, model, tokeniser)

    sentences_filepath = get_filepath(folder, "generated_sentences", suffix)
    output_generated_text(sentences_filepath, tokeniser, model)


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


examples = {
    'example1': example1,
    'example2': example2,
    'example3': example3,
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
