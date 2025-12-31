from BigramModel import BigramLanguageModel, BigramLanguageModelWithPositionalEncoding
from utils import generate_text, get_text, get_seqs, get_random_seqs, run_model
from tokeniser import Tokeniser, SimpleWordTokeniser


def bigram_model(folder, tokeniser=Tokeniser, include_positions=False):
    """ Train a bigram language model on the given text file """

    text = get_text(folder, "sentences.txt")
    seqs = get_seqs(text)
    tokeniser = tokeniser(seqs)

    print(tokeniser.vocab_size, tokeniser.block_size)

    # Function to get batches of training data
    encoded_seqs = [tokeniser.encode(seq) for seq in seqs]
    # get_batch = get_all_seqs(encoded_seqs)
    get_batch = get_random_seqs(encoded_seqs, batch_size=8)

    x, y = get_batch()
    print(x.shape, y.shape)

    # Create model
    if include_positions:
        model = BigramLanguageModelWithPositionalEncoding(tokeniser.vocab_size, tokeniser.block_size, n_embed=8)
    else:
        model = BigramLanguageModel(tokeniser.vocab_size)

    run_model(model, get_batch)

    # Generate some text
    max_new_tokens = min(20, tokeniser.block_size)
    generate_text(model, tokeniser, n = 10, max_new_tokens = max_new_tokens)

    print(model.output_embeddings())


if __name__ == "__main__":
    bigram_model("example1", tokeniser=SimpleWordTokeniser)
