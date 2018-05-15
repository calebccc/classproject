from Code.utils.dataset import Dataset
from Code.utils.scorer import report_score
from Code.utils.model import Model


def execute_model(language, featureSet):
    data = Dataset(language)

    print("{}: {} training - {} test".format(language, len(data.trainset), len(data.testset)))
    print("Features: {}".format(featureSet))

    system = Model(language, featureSet)

    print("Training...")
    system.train(data.trainset)

    print("Testing...")
    predictions = system.test(data.testset)

    gold_labels = [sent['gold_label'] for sent in data.testset]

    score = report_score(gold_labels, predictions, detailed=True)

    return score


if __name__ == '__main__':
    all_feats = ['baseline', 'upper', 'freq', 'pos']

    languages = ['english', 'spanish']

    execute_model('english', all_feats)
