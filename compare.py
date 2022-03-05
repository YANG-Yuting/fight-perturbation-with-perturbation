import pickle
import numpy as np

task = 'imdb'
target_model = 'wordLSTM'
with open('data/adversary_training_corpora/%s/AD_dpso_sem_%s.pkl' % (task, target_model), 'rb') as f:
    input_list, test_list, true_label_list, output_list, success, change_list, target_list = pickle.load(f)

# all_num_changes = []
lt4_num = 0.0
for i in range(len(output_list)):
    x_adv = output_list[i]
    x_orig = np.array(input_list[i][0])
    num_changes = np.sum(x_orig != x_adv)
    print(i, num_changes)
    if num_changes < 4:
        lt4_num += 1
        # all_num_changes.append(num_changes)
print(lt4_num / float(len(output_list)))