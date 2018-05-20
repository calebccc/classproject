import collections

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from sklearn.naive_bayes import MultinomialNB

from assignment.utils.dataset import Dataset


class Model(object):
    # initialised with a choice of language, model and features
    def __init__(self, language, model, columns):

        self.language = language
        self.columns = columns

        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3

        else:  # spanish
            self.avg_word_length = 6.2

        if model == 'lr':
            self.model = LogisticRegression()
        if model == 'rf':
            self.model = RandomForestClassifier()
        if model == 'bayes':
            self.model = MultinomialNB()
        if model == 'svm':
            self.model = SGDClassifier()

        # Building word counts
        self.word_count = self.word_freq(language)

        # The possible POS tags, used to do one hot to get the bag of word
        tags = ['NN', 'NNP', 'NNS', 'NNPS', 'VB', 'VBG', 'VBN', 'VBP', 'VBZ', 'RB', 'RBR', 'RBS', 'JJ', 'JJR', 'JJS']

        self.vec = CountVectorizer(vocabulary=tags)

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
        x = self.vec.fit_transform(pos)
        return x.toarray()[0]

    def upper_letters(self, word):
        ret = 0
        for w in word.split(' '):
            if not w:
                continue
            ret += 1 if w[0].isupper() else 0
        return ret

    def vertify_upper(self, word):
        ret = 0
        for w in word.split(' '):
            if not w:
                continue
            ret += 1 if w.isupper() else 0
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

        # the number of word starts with a upper letter
        if 'upper' in self.columns:
            feats.append(self.upper_letters(word))

        # word frequency
        if 'freq' in self.columns:
            feats.append(self.word_count[word])

        # feats.append(self.vertify_upper(word))

        # one hot bag of word
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
