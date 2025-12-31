# Special tokens
BOS = "<BOS>"
EOS = "<EOS>"
PAD = "<PAD>"
BR = "<BR>"


class Tokeniser:
    """ Class that converts characters to integers and back """

    def __init__(self, seqs):
        # Split sequences into tokens (characters)
        text = [self.split_text(seq) for seq in seqs]

        # Set block size to length of longest sequence + 2 (for BOS and EOS)
        self.block_size = max(len(seq) for seq in text) + 2
        self.vocab = self.get_vocab(text)
        self.vocab_size = len(self.vocab)

        print("Vocab size:", self.vocab_size)
        print("Block size:", self.block_size)
        print("Vocab:", self.vocab)


        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def split_text(self, text):
        """ Convert a string to a list of tokens """
        return list(text)

    def join_text(self, text):
        """ Convert a list of tokens to a string """
        return ''.join(text)

    def get_vocab(self, text):
        vocab = set()
        for token in text:
            vocab.update(token)
        return [BOS, PAD, EOS] + sorted(list(vocab))

    def encode(self, text):
        encoded_text = [self.stoi[char] for char in self.split_text(text)]
        if len(encoded_text) < self.block_size + 2:
            encoded_text += [self.stoi[PAD]] * (self.block_size - len(encoded_text) - 1)
        return [self.stoi[BOS]] + encoded_text + [self.stoi[EOS]]

    def decode(self, lst):
        end = lst.index(self.stoi[EOS]) if self.stoi[EOS] in lst else len(lst)
        lst = lst[1: end]  # remove BOS and everything after EOS
        chars = [self.itos[i] for i in lst if i != self.stoi[PAD]]
        return self.join_text(chars)


class WordTokeniser(Tokeniser):
    """ Class that converts words to integers and back """

    def split_text(self, text):
        """ Convert a string to a list of words """
        return text.split()

    def join_text(self, text):
        """ Convert a list of tokens to a string """
        return ' '.join(text)


class SimpleWordTokeniser(WordTokeniser):
    """
    Class that converts words to integers and back.
    Uses a single token to represent sentence breaks and doesn't have padding.
    """

    def get_vocab(self, text):
        vocab = set()
        for token in text:
            vocab.update(token)
        return [BR] + sorted(list(vocab))

    def encode(self, text):
        encoded_text = [self.stoi[char] for char in self.split_text(text)]
        return [self.stoi[BR]] + encoded_text + [self.stoi[BR]]

    def decode(self, lst):
        end = lst.index(self.stoi[BR], 1) if self.stoi[BR] in lst else len(lst)
        lst = lst[1: end]  # remove BR and everything after BR
        chars = [self.itos[i] for i in lst]
        return self.join_text(chars)
