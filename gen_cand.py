import OpenHowNet
hownet_dict = OpenHowNet.HowNetDict()
word_candidate = {}
word_pos = {}
word_sem = {}
word = 'love'
tree = hownet_dict.get_sememes_by_word(word, merge = False, structured = True, lang = "en")
w1_sememes = hownet_dict.get_sememes_by_word(word, structured = False, lang = "en", merge = False)
new_w1_sememes = [t['sememes'] for t in w1_sememes]
w1_pos_list = [x['word']['en_grammar'] for x in tree]

print(hownet_dict.get(word))
hownet_dict.initialize_sememe_similarity_calculation()
print(hownet_dict.get_nearest_words_via_sememes(word, K=5))
# print(w1_sememes)
# print(new_w1_sememes)
# print(w1_pos_list)
# 1. 词干还原
# 2. 获得词性和词义