import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from preprocessing import read_test

from itertools import product
from collections import OrderedDict
from preprocessing import represent_input_with_features
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def get_hist(sentence, k, pp_tag, p_tag, c_tag):
    """
    Extracts a history vector for a given index in a sentence
    """
    return (sentence[k], c_tag, sentence[k - 1], p_tag, sentence[k - 2], pp_tag, sentence[k + 1])


def memm_viterbi(sentence, pre_trained_weights, feature2id):
    tags = list(feature2id.feature_statistics.tags.copy())
    tag_pairs = list(product(tags, repeat=2))
    big_pi_table = {1: {i: 0 for i in tag_pairs}}
    big_pi_table[1][('*', '*')] = 1
    backpointer_table = {}
    q = OrderedDict()

    for k in range(2, len(sentence) - 1):
        big_pi_table[k], backpointer_table[k] = OrderedDict(), OrderedDict()

        if k == 2:
            s_1, s_2, s = ["*"], ["*"], tags
        elif k == 3:
            s_1, s_2, s = tags, ["*"], tags
        else:
            s_1, s_2, s = tags, tags, tags

        for u, v in product(s_1, s):
            max_prob, best_tag = -1, None

            # ---- Beam Search ----
            beam_width = 5
            if len(s_2) >= beam_width:
                beam_probs = [big_pi_table[k - 1][t, u] for t in s_2]
                beam_probs_indexes = np.argpartition(beam_probs, kth=(len(beam_probs) - beam_width))[-beam_width:]
                s_2 = np.array(s_2)[beam_probs_indexes]
            # ---------------------

            for t in s_2:
                h = get_hist(sentence, k, t, u, v)
                if h not in q:
                    softmax_denominator = 0
                    for tag in tags:
                        curr_h = get_hist(sentence, k, t, u, tag)
                        tag_value = np.exp(np.sum(
                            pre_trained_weights[represent_input_with_features(curr_h, feature2id.feature_to_idx)]))
                        softmax_denominator += tag_value
                        q[curr_h] = tag_value

                    for tag in tags:
                        curr_h = get_hist(sentence, k, t, u, tag)
                        q[curr_h] /= softmax_denominator

                prob_t_val = big_pi_table[k - 1][t, u] * q[h]
                if prob_t_val > max_prob:
                    max_prob = prob_t_val
                    best_tag = t

            big_pi_table[k][u, v] = max_prob
            backpointer_table[k][u, v] = best_tag

    top_value, best_pair = -1, None
    for p, v in big_pi_table[len(sentence) - 2].items():
        if v > top_value:
            top_value, best_pair = v, p
    tag_n_minus_1, tag_n = best_pair
    pred = [tag_n, tag_n_minus_1]

    for k in reversed(range(2, len(sentence) - 3)):
        next_tag = (backpointer_table[k + 2])[pred[-1], pred[-2]]
        pred.append(next_tag)

    return ["*"] + list(reversed(pred))


def score(actual, pred):
    assert len(actual) == len(pred), \
        "Actual sentence and prediction POS do not have same length!"
    hit = len([1 for y, y_t in zip(actual, pred) if y == y_t])

    return len(actual), hit


# def custom_score_func(y_list, predictions, error_results_dict):
#     assert len(y_list) == len(predictions)
#     total_hits = 0
#     for y, pred in zip(y_list, predictions):
#         if y == pred:
#             total_hits += 1
#         else:
#             if y not in error_results_dict:
#                 error_results_dict[y] = 1
#             else:
#                 error_results_dict[y] += 1
#
#     return len(y_list), total_hits, error_results_dict


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    tagged = "test" in test_path or "train1" in test_path or "train2" in test_path
    test = read_test(test_path, tagged=tagged)
    file_name = test_path.split('.')[0]

    output_file = open(predictions_path, "a+")

    total_words, hit_words = 0, 0

    labels = list(feature2id.feature_statistics.tags)
    y_true = []
    y_pred = []

    error_results_dict = {}
    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]

        pred = memm_viterbi(sentence, pre_trained_weights, feature2id)[1:]
        y_pred = y_pred + pred

        sentence = sentence[2:]
        actual = sen[1][2:-1]
        y_true = y_true + actual

        if tagged:
            # curr_words, curr_hits, error_results_dict = custom_score_func(actual, pred, error_results_dict)
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

    if tagged:  # Only enter if we go over a tagged file
        conf_mat = confusion_matrix(y_true, y_pred, labels=labels)

        display = ConfusionMatrixDisplay(conf_mat, display_labels=labels)
        display.plot()
        plt.savefig(f'{file_name}_full_confusion_matrix.png')

        errors_dict = {}
        for t, p in zip(y_true, y_pred):
            if t != p:
                if t not in errors_dict:
                    errors_dict[t] = 1
                elif t in errors_dict:
                    errors_dict[t] += 1

        sorted_keys = list(reversed({k: v for k, v in sorted(errors_dict.items(), key=lambda item: item[1])}.keys()))
        top10_keys = sorted_keys[:min(len(sorted_keys), 10)]
        top10_idx = [i for i, label in enumerate(labels) if label in top10_keys]
        print(top10_keys)
        top_10_cm = conf_mat[top10_idx][:, top10_idx]

        display = ConfusionMatrixDisplay(top_10_cm, display_labels=top10_keys)
        display.plot()
        plt.savefig(f'{file_name}_top_10_confusion_matrix.png')
