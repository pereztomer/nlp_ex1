import copy
from math import exp

from preprocessing import read_test, represent_input_with_features
from tqdm import tqdm


def calculate_q(history, dict_of_dicts, pre_trained_weights, S):
    """
    A helper function for the memm_viterbi() function, which
    calculates the q parameter in every memm_viterbi() iteration.
    :return: q value (float)
    """
    feature_indexes = represent_input_with_features(history, dict_of_dicts)
    numerator_val = 0
    for feature in feature_indexes:
        numerator_val += pre_trained_weights[feature]
    numerator = exp(numerator_val)
    denominator = 0
    denominator_history = copy.deepcopy(history)
    denominator_vector_multy = 0
    for tag in S:
        denominator_history[1] = tag
        denominator_feature_indexes = represent_input_with_features(denominator_history, dict_of_dicts)
        for feature_denominator in denominator_feature_indexes:
            denominator_vector_multy += pre_trained_weights[feature_denominator]
        denominator += exp(denominator_vector_multy)

    return numerator / denominator


def memm_viterbi(sentence, pre_trained_weights, feature2id):
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """
    # [*, *, 2, ...., 42 ]
    n = len(sentence)
    the_big_pi_table = {i + 1: {} for i in range(n)}
    bp_table = {i + 1: {} for i in range(n)}
    the_big_pi_table[1][('*', '*')] = 1
    S = feature2id.feature_statistics.tags  # all tags in the train set

    S_dict = {num: set() for num in range(0, n + 1)}
    S_dict[0] = S_dict[1] = ['*']

    for key in S_dict:
        if key > 1:
            S_dict[key] = S.copy()

    for index in range(2, n):  # 2, ..., n
        for u in S_dict[index - 1]:
            for v in S_dict[index]:
                # find max for t:
                max_val = 0
                argmax_tag = None
                for t in S_dict[index - 2]:
                    # history  = tuple{c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word}
                    c_word = sentence[index]
                    p_word = sentence[index - 1]
                    pp_word = sentence[index - 2]

                    c_tag = v
                    p_tag = u
                    pp_tag = t
                    if index not in (n, n - 1):
                        history = [c_word, c_tag, p_word, p_tag, pp_word, pp_tag, sentence[index + 1]]
                    else:
                        history = [c_word, c_tag, p_word, p_tag, pp_word, pp_tag, sentence[index]]

                    q = calculate_q(history, feature2id.feature_to_idx, pre_trained_weights, S)

                    current_val = the_big_pi_table[index - 1][(t, u)] * q
                    if current_val > max_val:
                        max_val = current_val
                        argmax_tag = t

                the_big_pi_table[index][(u, v)] = max_val
                bp_table[index][(u, v)] = argmax_tag

    t_assignments = {x: None for x in range(2, n + 1)}
    max_val = 0
    argmax_val = None
    for u in S:
        for v in S:
            current_pi_table_value = the_big_pi_table[n][(u, v)]
            if current_pi_table_value > max_val:
                max_val = current_pi_table_value
                argmax_val = (u, v)
            # triplet_count = feature2id.feature_statistics.tags_triplets_count[str([u, v, '*'])]
            # pairs_count = feature2id.feature_statistics.tags_pairs_count[str([u, v])]
            # q = triplet_count / pairs_count
            # q = calculate_q(history, feature2id.feature_to_idx, pre_trained_weights, S)

            # the_big_pi_table[n] -- This is a dict
            # max(the_big_pi_table[n], key=the_big_pi_table[n].get)[0]
            # bp_table[n][(u, v)]
    t_assignments[n], t_assignments[n - 1] = argmax_val
    # t_assignments[n], t_assignments[n - 1] = max(the_big_pi_table[n], key=the_big_pi_table[n].get)[0], \
    #                                          max(the_big_pi_table[n], key=the_big_pi_table[n].get)[1]
    # if max_val > the_big_pi_table[n][str([u, v])] * q:
    #     max_val = the_big_pi_table[n][str([u, v])] * q
    #     t_assignments[n - 1] = u
    #     t_assignments[n] = v

    for k in range(n - 2, -1, 2):
        t_assignments[k] = bp_table[k + 2][(t_assignments[k + 1], t_assignments[k + 2])]

    t_assignments = [val for val in t_assignments.values()]
    return t_assignments


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)

    output_file = open(predictions_path, "a+")

    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id)[1:]
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()
