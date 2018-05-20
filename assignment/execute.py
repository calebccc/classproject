from assignment.utils.dataset import Dataset
from assignment.utils.scorer import report_score
from assignment.utils.model import Model
import matplotlib.pyplot as plt


def execute_model(language, featureSet):
    data = Dataset(language)

    print("{}: {} training - {} test".format(language, len(data.trainset), len(data.testset)))
    print("Features: {}".format(featureSet))

    model = Model(language, 'lr', featureSet)

    print("Training...")
    model.train(data.trainset)

    print("Testing...")
    predictions = model.test(data.testset)

    gold_labels = [sent['gold_label'] for sent in data.testset]

    score = report_score(gold_labels, predictions, detailed=True)

    # print the word that predict wrong  
    # for i in range(len(gold_labels)):
    #     if predictions[i] != gold_labels[i]:
    #         print(data.testset[i]['target_word'])

    return score


def draw_plot(language, all_feats, models):
    data = Dataset(language)
    train_data = data.trainset
    gold_labels = [sent['gold_label'] for sent in data.testset]
    l = len(train_data) / 20
    lr = []
    rf = []
    by = []
    sv = []
    dt = []
    for i in range(20):
        e = int(l * i + l)
        dt.append(e)
        model = Model(language, 'lr', all_feats)
        model.train(train_data[:e])
        lr.append(report_score(gold_labels, model.test(data.testset), detailed=True))

        model = Model(language, 'rf', all_feats)
        model.train(train_data[:e])
        rf.append(report_score(gold_labels, model.test(data.testset), detailed=True))

        model = Model(language, 'bayes', all_feats)
        model.train(train_data[:e])
        by.append(report_score(gold_labels, model.test(data.testset), detailed=True))

        model = Model(language, 'svm', all_feats)
        model.train(train_data[:e])
        sv.append(report_score(gold_labels, model.test(data.testset), detailed=True))

    plt.figure(figsize=(8, 4))
    plt.plot(dt, lr, label='logistic regression', linewidth=1)
    plt.plot(dt, rf, label='random forest', linewidth=1)
    plt.plot(dt, by, label='naive bayes', linewidth=1)
    plt.plot(dt, sv, label='svm', linewidth=1)

    plt.xlabel("number of train data")  # X axis label
    plt.ylabel("f1 score")  # Y axis label
    plt.legend()
    plt.show()  # display


if __name__ == '__main__':
    all_feats = ['baseline', 'upper', 'freq', 'pos']

    languages = ['english', 'spanish']

    models = ['lr', 'rf', 'bayes', 'svm']

    execute_model('spanish', all_feats)

    # draw_plot('english', all_feats, models)
