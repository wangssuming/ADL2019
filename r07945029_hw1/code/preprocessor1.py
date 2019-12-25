import logging
import re
import nltk
import string
nltk.download('stopwords')
nltk.download('punkt')

class Preprocessor:
    """

    Args:
        embedding_path (str): Path to the embedding to use.
    """
    def __init__(self, embedding):
        self.embedding = embedding
        self.logging = logging.getLogger(name=__name__)
        self.stopset = set(nltk.corpus.stopwords.words('english'))
        self.symbolset = set(string.punctuation)
        self.pattern = re.compile("(\W|\d)")

    def tokenize(self, sentence):
        """ Tokenize a sentence.
        Args:
            sentence (str): One string.
        Return:
            indices (list of str): List of tokens in a sentence.
        """
        # TODO
        sentence = ' '.join(self.pattern.split(str.lower(sentence)))
        for symbol in self.symbolset:
            sentence = sentence.replace(symbol, '')
        tokens = nltk.tokenize.word_tokenize(sentence)


        return tokens

    def sentence_to_indices(self, sentence):
        """ Convert sentence to its word indices.
        Args:
            sentence (str): One string.
        Return:
            indices (list of int): List of word indices.
        """
        # TODO
        # Hint: You can use `self.embedding
        words = self.tokenize(sentence)
        indices = list()
        sen = list()
        for word in words:
            indices.append(self.embedding.to_index(word))
            sen.append(word)

        return indices, sen
    
    