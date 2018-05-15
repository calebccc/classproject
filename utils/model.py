import collections

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import cmudict
import nltk
from Code.utils.dataset import Dataset


class Model(object):
    # initialised with a choice of language, model and features
    def __init__(self, language, columns):

        self.language = language
        self.columns = columns

        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3

        else:  # spanish
            self.avg_word_length = 6.2

        self.model = RandomForestClassifier()

        # Building the frequency counts from the data
        self.word_count = self.word_freq(language)

        # The possible POS tags, needed to ensure equal vector lengths for each sentence
        tags = ['NN', 'NNP', 'NNS', 'NNPS', 'VB', 'VBG', 'VBN', 'VBP', 'VBZ', 'RB', 'RBR', 'RBS', 'JJ', 'JJR', 'JJS']

        self.vec = CountVectorizer(vocabulary=tags)

        # NLTK dict of syllables in words
        self.syl_dict = cmudict.dict()

    ##########################################################################################

    ### PREPROCESSING FUNTIONS FOR FEATURE EXTRACTION
    # reads the trainset
    def word_freq(self, language):
        data = Dataset(language)

        words_box = []
        for line in data.trainset:
            text = line['sentence'].lower()
            tokens = nltk.word_tokenize(text)
            words_box.extend(tokens)
        word_cnt = collections.Counter(words_box)

        return word_cnt

    # returns a one hot vector of pos tags
    def pos_counts(self, word):
        pos = [' '.join(i[1] for i in nltk.pos_tag(nltk.word_tokenize(word)))]
        X = self.vec.fit_transform(pos)
        pos_counts = X.toarray()[0]
        return pos_counts

    def upper_letters(self, word):
        ret = 0
        for w in word.split(' '):
            if not w:
                continue
            ret += 1 if w[0].isupper() else 0
        return ret

    ############################################################################################

    #### EXTRACTING SET OF FEATURES FROM EACH WORD ###

    def extract_features(self, word):

        feats = []

        # baseline features
        if 'baseline' in self.columns:
            len_chars = len(word) / self.avg_word_length  # relative word length
            len_tokens = len(word.split(' '))  # number of tokens in 'word'(may be sentence)
            feats.extend([len_chars, len_tokens])

        # word starts with a capital letter
        if 'upper' in self.columns:
            feats.append(self.upper_letters(word))

        # word unigram frequency
        if 'freq' in self.columns:
            feats.append(self.word_count[word])

        # Word POS tags one hot encoded 
        if 'pos' in self.columns:
            feats.extend(self.pos_counts(word))

        return feats

    #####################################################################################

    def train(self, trainset):
        X = []
        y = []
        for sent in trainset:
            X.append(self.extract_features(sent['target_word']))
            y.append(sent['gold_label'])

        self.model.fit(X, y)

    def test(self, testset):
        X = []
        for sent in testset:
            X.append(self.extract_features(sent['target_word']))

        return self.model.predict(X)
