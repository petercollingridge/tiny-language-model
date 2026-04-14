import sys

from BigramModel import BigramModel, DeeperBigramModel
from utils import generate_text, get_text, get_seqs, get_random_seqs, run_model, write_output
from tokeniser import Tokeniser, SimpleWordTokeniser


def get_tokeniser(folder, tokeniser=Tokeniser):
    text = get_text(folder, "training_sentences.txt")
    seqs = get_seqs(text)
    tokeniser = tokeniser(seqs)
    print(tokeniser.vocab_size, tokeniser.block_size)

    return seqs, tokeniser


def get_batching_func(tokeniser, seqs):
    # Function to get batches of training data
    encoded_seqs = [tokeniser.encode(seq) for seq in seqs]
    get_batch = get_random_seqs(encoded_seqs, batch_size=8)
    return get_batch


def save_model(folder, model, tokeniser):
    """ Save the model's output embeddings and the tokeniser's vocabulary to a file. """

    output = f"tokens = {tokeniser.vocab}\n"
    output += "weights:\n"
    for weight in model.output_weights():
        output += f"{weight}\n"
    write_output(folder, "model_output.txt", output)


def output_generated_text(folder, tokeniser, model, n=20):
    """
    Generate n sequences of text using the trained model and save to a file.
    """

    max_new_tokens = min(20, tokeniser.block_size)
    sequences = generate_text(model, tokeniser, n = n, max_new_tokens = max_new_tokens)
    output = "\n".join(sequences)   
    write_output(folder, "generated_sentences.txt", output)


def run_example(folder, ModelClass, steps=10000):
    """
    Run an example of training a bigram model on two sentences.
    """

    seqs, tokeniser = get_tokeniser(folder, SimpleWordTokeniser)
    model = ModelClass(tokeniser.vocab_size)
    get_batch = get_batching_func(tokeniser, seqs)

    run_model(model, get_batch, steps=steps)
    save_model(folder, model, tokeniser)
    output_generated_text(folder, tokeniser, model)


def example1(folder):
    """
    Training a bigram model on two sentences.
    """
    run_example(folder, BigramModel)


def example2(folder):
    """
    Training a bigram model with a hidden layer on two sentences.
    """

    run_example(folder, DeeperBigramModel, steps=20000)


def example3(folder):
    """
    Same as example 2, but with more words.
    """

    tokeniser = SimpleWordTokeniser

    text = get_text(folder, "training_sentences.txt")
    seqs = get_seqs(text)
    tokeniser = tokeniser(seqs)

    print(tokeniser.vocab_size, tokeniser.block_size)


examples = {
    'example1': example1,
    'example2': example2,
    'example3': example3,
}

if __name__ == "__main__":
    # Get value from command line argument to decide which example to run
    if len(sys.argv) > 1:
        example_name = sys.argv[1]
        if example_name in examples:
            examples[example_name](example_name)
        else:
            print(f"Unknown example: {example_name}")
