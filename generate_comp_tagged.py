import pickle
from inference import tag_all_test

weights_1 = f'train1_weights.pkl'
weights_2 = f'train2_weights.pkl'
comp_1 = f'data/comp1.words'
comp_2 = f'data/comp2.words'
comp_1_path = f'comp_m1_206230021_318295029.wtag'
comp_2_path = f'comp_m2_206230021_318295029.wtag'


with open(weights_1, 'rb') as f:
    optimal_params, feature2id = pickle.load(f)

tag_all_test(comp_1, optimal_params[0], feature2id, comp_1_path)

with open(weights_2, 'rb') as f:
    optimal_params, feature2id = pickle.load(f)

tag_all_test(comp_2, optimal_params[0], feature2id, comp_2_path)