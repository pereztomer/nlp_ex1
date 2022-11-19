from preprocessing import read_test
from tqdm import tqdm


def memm_viterbi(sentence, pre_trained_weights, feature2id):
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """
    n = len(sentence)
    the_big_pi_table = [{} for _ in range(n)]
    bp_table = [{} for _ in range(n)]
    the_big_pi_table[0][str(['*', '*'])] = 1
    S = feature2id.feature_statistics.tags
    S_dict = {num: set() for num in range(-1, n)}
    S_dict[-1] = S_dict[0] = {'*'}
    for key in S_dict:
        if key > 0:
            S_dict[key] = S.copy()

    for index in range(1, n + 1):
        for u in S_dict[index - 1]:
            for v in S_dict[index]:
                # find max for w:
                max_val = 0
                argmax_tag = None
                for w in S_dict[index - 2]:
                    triplet_count = feature2id.feature_statistics.tags_triplets_count[str([w, u, v])]
                    pairs_count = feature2id.feature_statistics.tags_pairs_count[str([u, v])]
                    q = triplet_count / pairs_count

                    e = feature2id.feature_statistics.word_tag_counts[(v, sentence[index])] / \
                        feature2id.feature_statistics.tags_counts[v]
                    current_val = the_big_pi_table[index - 1][str([w, u])] * q * e
                    if current_val > max_val:
                        max_val = current_val
                        argmax_tag = w

                the_big_pi_table[index][str([u, v])] = max_val
                bp_table[index][str([u, v])] = argmax_tag

    # calculating y_n_minus_1, y_n
    y_assignments = {x: None for x in range(1, n + 1)}
    max_val = 0
    for u in S:
        for v in S:
            triplet_count = feature2id.feature_statistics.tags_triplets_count[str([u, v, '*'])]
            pairs_count = feature2id.feature_statistics.tags_pairs_count[str([u, v])]
            q = triplet_count / pairs_count
            if max_val > the_big_pi_table[n][str([u, v])] * q:
                max_val = the_big_pi_table[n][str([u, v])] * q
                y_assignments[n - 1] = u
                y_assignments[n] = v

    for k in range(n - 2, -1, 1):
        y_assignments[k] = bp_table[k + 2][str([y_assignments[k + 1], y_assignments[k + 2]])]

    return y_assignments


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
