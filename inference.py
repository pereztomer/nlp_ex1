from tqdm import tqdm
import numpy as np
from itertools import product
from preprocessing import read_test
from collections import OrderedDict
from preprocessing import represent_input_with_features
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def build_history(sentence, k, pp_tag, p_tag, c_tag):
    c_word = sentence[k]
    p_word = sentence[k - 1]
    pp_word = sentence[k - 2]
    n_word = sentence[k + 1]

    h = (
        c_word,
        c_tag,
        p_word,
        p_tag,
        pp_word,
        pp_tag,
        n_word
    )
    return h


def calc_exp_val(pre_trained_weights, h, feature2id):
    features = represent_input_with_features(h, feature2id.feature_to_idx)
    return np.exp(np.sum(pre_trained_weights[features]))


def last_tags_pair(pi_dict):
    max_val = -1
    max_pair = None
    for pair, value in pi_dict.items():
        if value > max_val:
            max_val = value
            max_pair = pair
    return max_pair


def memm_viterbi(sentence, pre_trained_weights, feature2id):
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """
    n = len(sentence)
    tags = list(feature2id.feature_statistics.tags.copy())
    tag_pairs = list(product(tags, repeat=2))

    # Initial pi tables
    pi_dicts = {1: {i: 0 for i in tag_pairs}}
    pi_dicts[1][('*', '*')] = 1

    # initial bp tables
    bp_dicts = {}

    # initial q
    q = OrderedDict()

    for k in range(2, n - 1):
        pi_dicts[k] = OrderedDict()
        bp_dicts[k] = OrderedDict()

        if k == 2:
            s_minus_1 = ['*']
            s_minus_2 = ['*']
            s = tags

        elif k == 3:
            s_minus_1 = tags
            s_minus_2 = ['*']
            s = tags

        else:
            s_minus_1 = tags
            s_minus_2 = tags
            s = tags

        for u, v in product(s_minus_1, s):

            # future elements to place in pi(k, u, v)
            max_probability = -1
            best_tag = None

            width = 5
            if len(s_minus_2) >= 5:
                beam_probs = [pi_dicts[k - 1][t, u] for t in s_minus_2]
                beam_probs_indexes = np.argpartition(beam_probs, kth=(len(beam_probs) - width))[-width:]
                s_minus_2 = np.array(s_minus_2)[beam_probs_indexes]

            # searching for the max and argmax
            for t in s_minus_2:

                h = build_history(sentence=sentence, k=k, pp_tag=t, p_tag=u, c_tag=v)
                if h not in q:
                    softmax_denominator = 0
                    for sm_t in tags:
                        curr_h = build_history(sentence=sentence, k=k, pp_tag=t, p_tag=u, c_tag=sm_t)
                        sm_t_exp_val = calc_exp_val(pre_trained_weights, curr_h, feature2id)
                        softmax_denominator += sm_t_exp_val
                        q[curr_h] = sm_t_exp_val

                    for sm_t in tags:
                        curr_h = build_history(sentence=sentence, k=k, pp_tag=t, p_tag=u, c_tag=sm_t)
                        q[curr_h] /= softmax_denominator

                prob_t_val = pi_dicts[k - 1][t, u] * q[h]
                if prob_t_val > max_probability:
                    max_probability = prob_t_val
                    best_tag = t

            # update k, u, v table
            pi_dicts[k][u, v] = max_probability
            bp_dicts[k][u, v] = best_tag

    tag_n_minus_1, tag_n = last_tags_pair(pi_dict=pi_dicts[n - 2])
    pred = [tag_n, tag_n_minus_1]

    for k in reversed(range(2, n - 3)):
        next_tag = (bp_dicts[k + 2])[pred[-1], pred[-2]]
        pred.append(next_tag)

    return ["*"] + list(reversed(pred))


def score(actual, pred):
    assert len(actual) == len(pred), \
        "Actual sentence and prediction POS do not have same length!"
    hit = len([1 for y, y_t in zip(actual, pred) if y == y_t])

    return len(actual), hit


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    tagged = "test" in test_path or "train1" in test_path or "train2" in test_path
    test = read_test(test_path, tagged=tagged)
    file_name = test_path.split('.')[0]

    output_file = open(predictions_path, "a+")

    total_words, hit_words = 0, 0

    labels = list(feature2id.feature_statistics.tags)
    y_true = []
    y_pred = []

    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]

        pred = memm_viterbi(sentence, pre_trained_weights, feature2id)[1:]
        y_pred = y_pred + pred

        sentence = sentence[2:]
        actual = sen[1][2:-1]
        y_true = y_true + actual

        curr_words, curr_hits = score(actual, pred)
        total_words += curr_words
        hit_words += curr_hits
        if k % 10 == 0:
            print(f'Accuracy for {total_words} words is: {hit_words / total_words:.3f}')
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()

    print(f'Accuracy for {total_words} words is: {hit_words / total_words:.3f}')

    if tagged:
        # confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        display = ConfusionMatrixDisplay(cm, display_labels=labels)
        display.plot()
        plt.savefig(f'{file_name}_full_confusion_matrix.png')

        errors_dict = {}
        for t, p in zip(y_true, y_pred):
            if t != p:
                if t not in errors_dict:
                    errors_dict[t] = 1
                elif t in errors_dict:
                    errors_dict[t] += 1
        top10_keys = list({k: v for k, v in sorted(errors_dict.items(), key=lambda item: item[1])}.keys())[:10]
        top10_idx = [i for i, label in enumerate(labels) if label in top10_keys]
        print(top10_keys)
        top_10_cm = cm[top10_idx][:, top10_idx]

        display = ConfusionMatrixDisplay(top_10_cm, display_labels=top10_keys)
        display.plot()
        plt.savefig(f'{file_name}_top_10_confusion_matrix.png')
