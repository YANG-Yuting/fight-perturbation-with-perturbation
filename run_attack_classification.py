import os

# for wordCNN target
command = 'python attack_classification.py --dataset_path data/imdb ' \
           '--target_model wordLSTM --batch_size 128 ' \
           '--target_model_path models/wordLSTM/imdb ' \
           '--word_embeddings_path glove.6B/glove.6B.200d.txt ' \
           '--counter_fitting_embeddings_path ./counter-fitted-vectors.txt ' \
           '--counter_fitting_cos_sim_path ./cos_sim_counter_fitting.npy ' \
           '--USE_cache_path ./tf_cache'

# command = 'python fastca.py --dataset_path data/imdb ' \
#            '--target_model wordCNN --batch_size 128 ' \
#            '--target_model_path ./imdb ' \
#            '--word_embeddings_path ./glove.6B/glove.6B.300d.txt ' \
#            '--counter_fitting_embeddings_path ./counter-fitted-vectors.txt ' \
#            '--counter_fitting_cos_sim_path ./cos_sim_counter_fitting.npy ' \
#            '--USE_cache_path ./tf_cache'
#
# command = 'python fastca.py --dataset_path data/imdb ' \
#            '--word_embeddings_path ./glove.6B/glove.6B.300d.txt ' \
#            '--target_model bdlstm --batch_size 128 ' \
#            '--target_model_path data/Imdb/bdlstm_models'

# command = 'python fastca.py --dataset_path data/imdb ' \
#            '--word_embeddings_path ./glove.6B/glove.6B.300d.txt ' \
#            '--target_model bert --batch_size 128 ' \
#            '--target_model_path models/bert/imdb'

# for BERT target
#command = 'python attack_classification.py --dataset_path data/yelp ' \
#         '--target_model bert ' \
#          '--target_model_path /scratch/jindi/adversary/BERT/results/yelp ' \
#          '--max_seq_length 256 --batch_size 32 ' \
#          '--counter_fitting_embeddings_path /data/medg/misc/jindi/nlp/embeddings/counter-fitted-vectors.txt ' \
#          '--counter_fitting_cos_sim_path /scratch/jindi/adversary/cos_sim_counter_fitting.npy ' \
#          '--USE_cache_path /scratch/jindi/tf_cache'

os.system(command)
