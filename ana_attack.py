import pickle
import argparse
from numpy import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
from operator import itemgetter

# args
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default='imdb', help="task name: mr/imdb")
parser.add_argument("--max_seq_length", default=128, type=int, help="max sequence length for BERT target model")
parser.add_argument("--target_model", type=str, default='bert', help="For mr/imdb: wordLSTM or bert")
parser.add_argument("--word_embeddings_path", type=str, default='./glove.6B/glove.6B.200d.txt', help="path to the word embeddings for the target model")
args = parser.parse_args()
args.target_model_path = ('models/%s/%s' % (args.target_model, args.task))

seq_len_list = {'imdb': 256, 'mr': 128, 'snli':128}
args.max_seq_length = seq_len_list[args.task]
test_size = {'imdb': 25000, 'mr':1067, 'snli': 10000}
sizes = {'imdb': 50000, 'mr': 20000, 'snli': 50000}
MAX_VOCAB_SIZE = sizes[args.task]
# with open('data/adversary_training_corpora/%s/dataset_%d.pkl' % (args.task, MAX_VOCAB_SIZE), 'rb') as f:
#     dataset = pickle.load(f)
# test_texts = dataset.test_seqs2
# lens = []
# for text in test_texts:
#     lens.append(len(text))
# lens.sort(reverse=True)
# print(lens[0], lens[1], lens[2])
# print(np.min(lens), np.max(lens))
# exit(0)

# print(dataset.test_seqs2[539], len(dataset.test_seqs2[23]))
# print(' '.join(dataset.inv_full_dict[i] for i in dataset.test_seqs2[539]))
# exit(0)


def arrange_result():
    lines = open('FindAdresult/last_result/%s_%s_MPDP' % (args.task, args.target_model), 'r').read().splitlines()
    i = 0
    data = {}
    temp = []
    while i < len(lines):
        if 'text id' in lines[i]:
            idx = int(lines[i].replace('text id:', ''))
            print(idx)
            i += 1
            continue
        else:
            temp.append(lines[i])
            i += 1
            if i >= len(lines):
                data[idx] = temp
                temp = []
                continue
            if 'text id' in lines[i]:
                data[idx] = temp
                temp = []
    keys = sorted(data.keys())
    with open('FindAdresult/last_result/%s_%s_MPDP_arrange' % (args.task, args.target_model), 'w') as f:
        for k in keys:
            f.write('text id:%d' % k + '\n')
            for d in data[k]:
                f.write(d + '\n')


def ana_TH():
    # mr/imdb
    with open('data/adversary_training_corpora/%s/AD_dpso_sem_%s_ifvalid_predictTrue1000.pkl' % (args.task, args.target_model), 'rb') as f:
        wrong_clas_id, too_long_id, attack_list = pickle.load(f)
    with open('data/adversary_training_corpora/%s/AD_dpso_sem_%s_fail_predictTrue1000.pkl' % (args.task, args.target_model), 'rb') as f:
        failed_list, failed_input_list, failed_time = pickle.load(f)
    with open('data/adversary_training_corpora/%s/AD_dpso_sem_%s_modtoomuch_predictTrue1000.pkl' % (args.task, args.target_model), 'rb') as f:
        modift_too_much_id, modift_too_much_changes, modift_too_much_input_list, modift_too_much_output_list, modift_too_much_time = pickle.load(f)
    with open('data/adversary_training_corpora/%s/AD_dpso_sem_%s_success_predictTrue1000.pkl' % (args.task, args.target_model), 'rb') as f:
        input_list, true_label_list, output_list, success, change_list, target_list, num_change_list, success_time = pickle.load(f)

    # snli
    # with open('/pub/data/huangpei/SememePSO/SNLI/dataset/AD_dpso_sem_%s_ifvalid.pkl' % (args.target_model), 'rb') as f:
    #     wrong_clas_id, too_long_id, attack_list = pickle.load(f)
    # with open('/pub/data/huangpei/SememePSO/SNLI/dataset/AD_dpso_sem_%s_fail.pkl' % (args.target_model), 'rb') as f:
    #     failed_list, failed_input_list, failed_time = pickle.load(f)
    # with open('/pub/data/huangpei/SememePSO/SNLI/dataset/AD_dpso_sem_%s_modtoomuch.pkl' % (args.target_model), 'rb') as f:
    #     modift_too_much_id, modift_too_much_changes, modift_too_much_input_list, modift_too_much_output_list, modift_too_much_time = pickle.load(f)
    # with open('/pub/data/huangpei/SememePSO/SNLI/dataset/AD_dpso_sem_%s_success.pkl' % (args.target_model), 'rb') as f:
    #     input_list, true_label_list, output_list, success, change_list, target_list, num_change_list, success_time = pickle.load(f)

    #
    # print(modift_too_much_id.index(42))
    print(attack_list[-1])
    exit(0)

    print(failed_list.index(34))
    print(num_change_list[success.index(310)])
    exit(0)

    print('-'*60)
    print('Task: %s, Test size: %d, Model: %s' % (args.task, len(wrong_clas_id)+len(too_long_id)+len(attack_list), args.target_model))
    # print('Task: %s, Test size: %d, Model: %s' % (args.task, 1000, args.target_model))
    print('Num of wrong classified: %d (%f)' % (len(wrong_clas_id), 1-float(len(wrong_clas_id))/float(len(wrong_clas_id)+len(too_long_id)+len(attack_list))))
    print('Num of too long input (longer than %d): %d' % (args.max_seq_length, len(too_long_id)))
    print('Num of attack: ', len(attack_list))

    print('Num of attack failed:', len(failed_list))
    print('---Among them, avg attack time:', mean(failed_time))
    print('Num of change ratio more than 0.25:', len(modift_too_much_id))
    print('---Among them, avg attack time:', mean(modift_too_much_time))
    print('Num of attack success: %d (%f)' % (len(success), float(len(success)/float(len(attack_list)))))
    print('---Among them, avg change ratio:', mean(change_list))
    print('---Among them, avg change num:', mean(num_change_list))
    print('---Among them, avg attack time:', mean(success_time))
    print('-'*60)

    return

def ana_MIT():
    with open('adv_results/adv_hownettop5_%s_%s_ifvalid.pkl' % (args.task, args.target_model), 'rb') as f:
        wrong_clas_id = pickle.load(f)
    with open('adv_results/adv_hownettop5_%s_%s_fail.pkl' % (args.task, args.target_model), 'rb') as f:
        failed_list, failed_input_list, failed_time = pickle.load(f)
    with open('adv_results/adv_hownettop5_%s_%s_success.pkl' % (args.task, args.target_model), 'rb') as f:
        input_list, true_label_list, output_list, success, change_list, num_change_list, success_time = pickle.load(f)

    # 过滤成功攻击中比例超过0.25的

    new_success = np.array(success)[np.array(change_list) < 0.25]
    new_change_list = np.array(change_list)[np.array(change_list) < 0.25]
    new_num_change_list = np.array(num_change_list)[np.array(change_list) < 0.25]
    new_success_time = np.array(success_time)[np.array(change_list) < 0.25]
    test_size = len(wrong_clas_id)+len(failed_list)+len(success)

    print('-'*60)
    print('Task: %s, Test size: %d, Model: %s' % (args.task, test_size , args.target_model))
    # print('Task: %s, Test size: %d, Model: %s' % (args.task, 1000, args.target_model))
    print('Num of wrong classified: %d (%f)' % (len(wrong_clas_id), 1- float(len(wrong_clas_id))/test_size))
    print('Num of attack failed:', len(failed_list))
    print('---Among them, avg attack time:', mean(failed_time))
    print('Num of attack success: %d (%f)' % (len(new_success), float(len(new_success))/test_size))
    print('---Among them, avg change ratio:', mean(new_change_list))
    print('---Among them, avg change num:', mean(new_num_change_list))
    print('---Among them, avg attack time:', mean(new_success_time))
    print('-'*60)


def ana_HP():
    Infilename= ('FindAdresult/%s_%s_onlyDPFGWS' % (args.task, args.target_model))
    f=open(Infilename)
    lines = f.read().splitlines()

    changes = {}

    f = open('FindAdresult/%s_%s_onlyDPFGWS_ana.txt' % (args.task, args.target_model), 'w')
    i = 0
    attacks = 0
    attack_list = []
    found=0
    while i < len(lines):
        line = lines[i]
        if 'text id' in line:
            idx = int(line.replace('text id:', ''))
            # print('text id:', idx)
            i += 1
            attacks += 1
            attack_list.append(idx)
            # if attacks > 1000:
            #     # print('get 1000 predict true!')
            #     break
            # print(attacks)
            # attack_list.sort()
            # print(len(attack_list), attack_list)
            continue

        if 'r=' in line:
            i += 1
            if 'Certified' in lines[i]:
                i += 1
            if 'Apply' in lines[i]:
                i += 1
            if 'Found' in lines[i]:
                i += 1
                # print(lines[i])
                found_info = lines[i].replace('  ',' ').replace('  ',' ').split(' ')
                # print(found_info)
                ori_length = int(found_info[2])
                change_num = int(found_info[4])
                change_ratio = float(found_info[5])
                if change_ratio <= 0.25:
                    changes[idx] = [ori_length, change_num, change_ratio]
                    print(idx, ori_length, change_num, change_ratio)
                    print(idx, ori_length, change_num, change_ratio, file=f)
                i += 1
            try:
                if 'Failed' in lines[i]:
                    i += 1
            except:
                list(set(attack_list)).sort()
                return attack_list
        if 'Certified' in line:
            i += 1
        if 'Apply' in lines[i]:
            i += 1
            continue
        if 'Found' in lines[i]:
            i += 1
            found += 1
            # print(lines[i])
            found_info = lines[i].replace('  ', ' ').replace('  ', ' ').split(' ')
            # print(found_info)
            ori_length = int(found_info[2])
            change_num = int(found_info[4])
            change_ratio = float(found_info[5])
            if change_ratio <= 0.25:
                changes[idx] = [ori_length, change_num, change_ratio]
                # print(idx, ori_length, change_num, change_ratio)
                # print(idx, ori_length, change_num, change_ratio, file=f)
            i += 1
            continue
        if 'Failed' in lines[i]:
            i += 1
            continue
    # attack_list.sort()
    # print(attack_list)
    print('-----------------------')
    print(args.task, args.target_model, 'FGWS')
    print('Found with ratio < 0.25:', len(changes))
    print('Found:',found)
    print('Found with ratio < 0.25 / 200:' , float(len(changes))/200.0)
    return attack_list


def ana_imdb_bert_certi():
    f = open('FindAdresult/last_result/imdb_bert_onlyDP', 'r')
    lines = f.read().splitlines()

    all_ids = []
    changes = {}
    fail_ids = []
    all_nums = 0
    lt_256_nums = 0
    for line in lines:
        if 'text id' in line:
            all_nums += 1
            idx = int(line.replace('text id:', ''))
            all_ids.append(idx)
            # print(idx, len(test_texts[idx-1]))
            continue

        if 'Changed' in line:
            found_info = line.replace('  ', ' ').replace('  ', ' ').split(' ')
            ori_length = int(found_info[2])
            change_num = int(found_info[4])
            if ori_length <= 256:
                changes[idx] = change_num
                lt_256_nums += 1

        if 'Failed' in line:
            fail_ids.append(idx)
            if len(test_texts[idx]) <= 256:
                changes[idx] = 10000
                lt_256_nums += 1

        # if len(changes) >200:
        #     break

    print('All data:', all_nums)
    print('Less than 256:', lt_256_nums)

    certified_num = {'1': 0, '2': 0, '3': 0, '4': 0}
    for c in changes.values():
        if c > 1:
            certified_num['1'] += 1
        if c > 2:
            certified_num['2'] += 1
        if c > 3:
            certified_num['3'] += 1
        if c > 4:
            certified_num['4'] += 1
    print(certified_num)
    certified_percent = {'1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0}
    found_percent = {'1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0}

    for cn in certified_num.keys():
        certified_percent[cn] = certified_num[cn] / float(len(changes))
        found_percent[cn] = 1 - certified_percent[cn]

    print(found_percent)
    print(certified_percent)

    return None


def get_HP_attacks():
    Infilename= ('FindAdresult/last_result/%s_%s_onlyDP' % (args.task, args.target_model))
    f=open(Infilename)
    lines = f.read().splitlines()

    i = 0
    attack_list = []
    for i in range(len(lines)):
        line = lines[i]
        if 'text id' in line:
            idx = int(line.replace('text id:', ''))
            # print('text id:', idx)
            attack_list.append(idx)

    attack_list = list(set(attack_list))
    attack_list.sort()
    return attack_list
def get_HP_failed():
    Infilename= ('FindAdresult/last_result/%s_%s_onlyDP2.3' % (args.task, args.target_model))
    # Infilename= ('FindAdresult/last_result/%s_%s_MPDP' % (args.task, args.target_model))
    # Infilename= ('/pub/data/huangpei/SememePSO/SNLI/FindAdresult/%s_%s_MPDP' % (args.task, args.target_model))

    f=open(Infilename)
    lines = f.read().splitlines()
    failed_list = []
    for i in range(len(lines)):
        line = lines[i]
        if 'text id' in line:
            idx = int(line.replace('text id:', ''))
        if 'Failed' in line:
            failed_list.append(idx)

    failed_list = list(set(failed_list))
    failed_list.sort()
    return failed_list

def ana_HP_certi():
    # Infilename=sys.argv[1]
    Infilename= ('FindAdresult/last_result/%s_%s_MPDP_arrange' % (args.task, args.target_model)) # /pub/data/huangpei/SememePSO/SNLI/
    f=open(Infilename)
    lines = f.read().splitlines()

    changes = {}

    i = 0
    not_certified_ids = []
    all_ids = []
    while len(all_ids) < 1001:
        line = lines[i]
        if 'text id' in line:
            idx = int(line.replace('text id:', ''))
            all_ids.append(idx)
            print(len(all_ids), len(not_certified_ids), len(all_ids) - len(not_certified_ids))
            i += 1
            continue

        if 'r=' in line:
            i += 1
            if 'Certified' in lines[i]:
                i += 1
            if 'Apply' in lines[i]:
                i += 1
            if 'Found' in lines[i]:
                i += 1
                found_info = lines[i].replace('  ',' ').replace('  ',' ').split(' ')
                ori_length = int(found_info[2])
                change_num = int(found_info[4])
                change_ratio = float(found_info[5])
                if change_ratio <= 0.25:
                    changes[idx] = [ori_length, change_num, change_ratio]
                i += 1
                not_certified_ids.append(idx)
            if 'Failed' in lines[i]:
                i += 1
                not_certified_ids.append(idx)

        if 'Certified' in line:
            i += 1


def huangdiudiu(attack_list_Se,success_Se,change_list_Se,num_change_list_Se,success_time_Se):
    dic_list=list()
    for i in range(len(attack_list_Se)):
        SS={
            'text_id':attack_list_Se[i],
            'success':None,
            'cratio':None,
            'cnum':None,
            'time':None

        }
        for j in range(len(success_Se)):
            if success_Se[j]==attack_list_Se[i]:
                SS['success']=True
                SS['cratio']=change_list_Se[j]
                SS['cnum']=num_change_list_Se[j]
                SS['time']=success_time_Se[j]
        dic_list.append(SS)
    return dic_list

def removeDump(dic_list):

    dic_list1=list()

    for dic in dic_list:
        flag=True
        for dic1 in dic_list1:
            if dic['text_id'] ==dic1['text_id']:
                flag=False
        if  flag==True:
            dic_list1.append(dic)
    return dic_list1


def compare_imdb():
    # ------- SemPSO -------
    # imdb
    print('------- SemPSO -------')
    with open('data/adversary_training_corpora/%s/AD_dpso_sem_%s_ifvalid_1500.pkl' % (args.task, args.target_model),'rb') as f:
        wrong_clas_id, too_long_id, attack_list_Se = pickle.load(f)
    with open('data/adversary_training_corpora/%s/AD_dpso_sem_%s_fail_1500.pkl' % (args.task, args.target_model), 'rb') as f:
        failed_list_Se, _, _ = pickle.load(f)
    with open('data/adversary_training_corpora/%s/AD_dpso_sem_%s_success_1500.pkl' % (args.task, args.target_model),'rb') as f:
        input_list_Se, _, output_list_Se, success_Se, change_list_Se, _, num_change_list_Se, success_time_Se = pickle.load(f)
    # 取出前1000个中,攻击成功的
    attack_list_Se.sort()
    attack_list_Se = attack_list_Se[:1000]  # 前1000个攻击id
    new_success_Se = []
    new_num_change_list_Se = []
    new_change_list_Se = []
    new_output_list_Se = []
    new_failed_list_Se = []
    for idx in range(len(attack_list_Se)):
        if attack_list_Se[idx] in success_Se:
            new_success_Se.append(attack_list_Se[idx])
            iii = success_Se.index(attack_list_Se[idx])
            new_num_change_list_Se.append(num_change_list_Se[iii])
            new_change_list_Se.append(change_list_Se[iii])
            new_output_list_Se.append(output_list_Se[iii])
        elif attack_list_Se[idx] in failed_list_Se:
            new_failed_list_Se.append(attack_list_Se[idx])

    print(len(attack_list_Se), len(new_success_Se), np.mean(new_num_change_list_Se), np.mean(new_change_list_Se))


    new_num_statis = dict.fromkeys(range(np.min(new_num_change_list_Se), np.max(new_num_change_list_Se)+1), 0)
    for c_num in new_num_change_list_Se:
        for i in range(np.min(new_num_change_list_Se), np.max(new_num_change_list_Se)+1):
            if c_num > i:
                new_num_statis[i] += 1
    fail_num = len(new_failed_list_Se)
    for k in new_num_statis.keys():
        new_num_statis[k] += fail_num
        new_num_statis[k] = float(new_num_statis[k]) /float(len(new_success_Se) + fail_num)
    print(len(new_success_Se)+fail_num)
    print(new_num_statis)


    # -------TextFooler-------
    print('-------TextFooler-------')
    # imdb
    with open('adv_results/adv_hownettop5_%s_%s_fail_2000.pkl' % (args.task, args.target_model), 'rb') as f:
        failed_list_Te, failed_input_list, failed_time = pickle.load(f)
    with open('adv_results/adv_hownettop5_%s_%s_success_2000.pkl' % (args.task, args.target_model), 'rb') as f:
        input_list_Te, _, output_list_Te, success_Te, change_list_Te, num_change_list_Te, success_time_Te = pickle.load(f)
    attack_list_Te = failed_list_Te + success_Te
    attack_list_Te.sort()
    attack_list_Te = attack_list_Te[:1000]  # 前1000个攻击id
    new_success_Te = []
    new_num_change_list_Te = []
    new_change_list_Te = []
    new_output_list_Te = []
    new_failed_list_Te = []
    for idx in range(len(attack_list_Te)):
        if attack_list_Te[idx] in success_Te:
            iii = success_Te.index(attack_list_Te[idx])
            if change_list_Te[iii] <= 0.25:  # 在前1000个攻击中，保留替换比例小于0.25的
                new_success_Te.append(attack_list_Te[idx])
                new_num_change_list_Te.append(num_change_list_Te[iii])
                new_change_list_Te.append(change_list_Te[iii])
                new_output_list_Te.append(output_list_Te[iii])
        elif attack_list_Te[idx] in failed_list_Te:
            new_failed_list_Te.append(attack_list_Te[idx])
    print(len(attack_list_Te), len(new_success_Te), np.mean(new_num_change_list_Te), np.mean(new_change_list_Te))

    new_num_statis = dict.fromkeys(range(np.min(new_num_change_list_Te), np.max(new_num_change_list_Te)+1), 0)
    for c_num in new_num_change_list_Te:
        for i in range(np.min(new_num_change_list_Te), np.max(new_num_change_list_Te)+1):
            if c_num > i:
                new_num_statis[i] += 1
    fail_num = len(new_failed_list_Te)
    for k in  new_num_statis.keys():
        new_num_statis[k] += fail_num
        new_num_statis[k] = float(new_num_statis[k]) /float(len(new_success_Te) + fail_num)
    print(len(new_success_Te) + fail_num)
    print(new_num_statis)

    # idx = 24 #21
    # ori_text = ' '.join(dataset.inv_full_dict[i] for i in test_texts[idx])
    # print(ori_text)
    # Se_attack = output_list_Se[success_Se.index(idx)]
    # Se_attack = ' '.join(dataset.inv_full_dict[i] for i in Se_attack)
    # print(Se_attack)
    # Te_attack = output_list_Te[success_Te.index(idx)]
    # print(Te_attack)
    # exit(0)

    # ------- PDP -------
    print('------- PDP -------')
    lines = open('FindAdresult/%s_%s_onlyDP_ana.txt' % (args.task, args.target_model), 'r').read().splitlines()
    success_DP = [int(line.split(' ')[0]) for line in lines]
    change_list_DP = [float(line.split(' ')[3]) for line in lines]
    num_change_list_DP = [int(line.split(' ')[2]) for line in lines]
    failed_list_DP = get_HP_failed()

    # 按照id排序
    attack_list_DP = get_HP_attacks()
    attack_list_DP = attack_list_DP[:1000]  # 前1000个攻击id
    # 取出前1000个中，攻击成功的
    new_success_DP = []
    new_num_change_list_DP = []
    new_change_list_DP = []
    new_failed_list_DP = []
    for idx in range(len(attack_list_DP)):
        if attack_list_DP[idx] in success_DP:
            new_success_DP.append(attack_list_DP[idx])
            iii = success_DP.index(attack_list_DP[idx])
            new_num_change_list_DP.append(num_change_list_DP[iii])
            new_change_list_DP.append(change_list_DP[iii])
        elif attack_list_DP[idx] in failed_list_DP:
            new_failed_list_DP.append(attack_list_DP[idx])

    print(len(attack_list_DP), len(new_success_DP), np.mean(new_num_change_list_DP), np.mean(new_change_list_DP))

    # c_num_statis = dict.fromkeys(range(np.min(new_num_change_list_DP), np.max(new_num_change_list_DP)+1), 0)
    # print(c_num_statis)
    # for c_num in new_num_change_list_DP:
    #     if c_num in c_num_statis.keys():
    #         c_num_statis[c_num] += 1
    # print(c_num_statis)

    new_num_statis = dict.fromkeys(range(np.min(new_num_change_list_DP), np.max(new_num_change_list_DP)+1), 0)
    for c_num in new_num_change_list_DP:
        for i in range(np.min(new_num_change_list_DP), np.max(new_num_change_list_DP)+1):
            if c_num > i:
                new_num_statis[i] += 1

    fail_num = len(new_failed_list_DP)
    for k in new_num_statis.keys():
        new_num_statis[k] += fail_num
        new_num_statis[k] = float(new_num_statis[k]) /float(len(new_success_DP) + fail_num)
    print(len(new_success_DP) + fail_num)
    print(new_num_statis)
    exit(0)




    # 组合三种攻击结果
    # ids = [idx for idx in attack_list_DP if idx in attack_list_Se and idx in attack_list_Te]
    ids = [idx for idx in new_success_Se if idx in new_success_Te and idx in new_success_DP]  # and idx in new_success_DP
    print('Overlap successful ids in three attack methods:', len(ids))
    # list_re = list(set(new_success_Se).difference(set(new_success_DP)))
    # print('Success in Se but fail in DP:', list_re)
    # for idx in list_re:
    #     index_Se= new_success_Se.index(idx)
    #     print(idx, new_num_change_list_Se[index_Se])
    # print(all_num_change)
    # list_re = list(set(new_success_Te).difference(set(new_success_DP)))
    # print(list_re)
    # exit(0)
    all_num_change = {}
    # f1 = open('FindAdresult/num_changes_pdpVSse_%s_%s.txt' % (args.task, args.target_model), 'w')
    # f2 = open('FindAdresult/num_changes_pdpVSte_%s_%s.txt' % (args.task, args.target_model), 'w')
    for idx in ids:
        index_Se = new_success_Se.index(idx)
        index_Te = new_success_Te.index(idx)
        index_DP = new_success_DP.index(idx)
        all_num_change[idx] = [new_num_change_list_Se[index_Se], new_num_change_list_Te[index_Te], new_num_change_list_DP[index_DP]]
        # print(new_num_change_list_DP[index_DP], new_num_change_list_Se[index_Se], file=f1)
        # print(new_num_change_list_DP[index_DP], new_num_change_list_Te[index_Te], file=f2)
    # ids = [idx for idx in success_MDP if idx in success_DP]
    # f1 = open('FindAdresult/num_changes_mdpVSdp_%s_%s.txt' % (args.task, args.target_model), 'w')
    # for idx in ids:
    #     index_mdp = success_MDP.index(idx)
    #     index_DP = success_DP.index(idx)
    #     print(num_change_list_MDP[index_mdp], num_change_list_DP[index_DP], file=f1)
    # print(all_num_change)

    print('SemPSO better than others:')
    better_num = 0
    fail_ids = []
    for idx in all_num_change.keys():
        if all_num_change[idx][0] < all_num_change[idx][1] and all_num_change[idx][0] < all_num_change[idx][2]:
            better_num += 1
            # print(idx,all_num_change[idx])
            # fail_ids.append(idx)
    print(better_num, float(better_num) /float(len(all_num_change)))

    print('TextFooler better than others:')
    better_num = 0
    for idx in all_num_change.keys():
        if all_num_change[idx][1] < all_num_change[idx][0] and all_num_change[idx][1] < all_num_change[idx][2]:
            better_num += 1
            fail_ids.append(idx)
    print(better_num, float(better_num) /float(len(all_num_change)))
    print('PDP better than others:')
    better_num = 0
    better_ids = []
    for idx in all_num_change.keys():
        if all_num_change[idx][2] < all_num_change[idx][0] and all_num_change[idx][2] < all_num_change[idx][1]:
            better_num += 1
            better_ids.append(idx)
            # fail_ids.append(idx)
            # print(idx, all_num_change[idx])
    print(better_num, float(better_num) /float(len(all_num_change)))


    # temp = {}
    # for idx in better_ids:
    #     index_Se = new_success_Se.index(idx)
    #     index_Te = new_success_Te.index(idx)
    #     index_MDP = new_success_DP.index(idx)
    #     temp[idx] = [new_num_change_list_Se[index_Se], new_num_change_list_Te[index_Te], new_num_change_list_DP[index_MDP]]
    # print(better_ids)
    # print(temp)
    # print(fin_te_list)

    # print(np.array(new_change_list_Se)[fail_ids])
    # print(np.array(new_change_list_Te)[fail_ids])
    # print(np.array(new_num_change_list_DP)[fail_ids])

def compare_mr():
    # with open('data/adversary_training_corpora/mr/dataset_20000.pkl', 'rb') as f:
    #     dataset = pickle.load(f)
    # inv_full_dict = dataset.inv_full_dict
    # inv_dict = dataset.inv_dict


    # ------- SemPSO -------
    print('------- SemPSO -------')
    # mr
    # with open('data/adversary_training_corpora/%s/AD_dpso_sem_%s_ifvalid.pkl' % (args.task, args.target_model), 'rb') as f:
    #     wrong_clas_id, too_long_id, attack_list = pickle.load(f)
    # with open('data/adversary_training_corpora/%s/AD_dpso_sem_%s_fail.pkl' % (args.task, args.target_model),'rb') as f:
    #     failed_list_Se, _, _ = pickle.load(f)
    # with open('data/adversary_training_corpora/%s/AD_dpso_sem_%s_success.pkl' % (args.task, args.target_model), 'rb') as f:
    #     input_list_Se, _, output_list_Se, success_Se, change_list_Se, _, num_change_list_Se, success_time_Se = pickle.load(f)
    wrong_clas_id, too_long_id, attack_list, failed_list_Se, output_list_Se, success_Se, change_list_Se, \
    num_change_list_Se, new_success_time_Se = combing_PSO()

    print('Attack num: %d, Attack success: %d, Mean of Change num: %f, Mean of Change ratio: %f ' %
          (len(attack_list), len(success_Se), np.mean(num_change_list_Se), np.mean(change_list_Se)))
    # print(' '.join([inv_dict[i] for i in input_list_Se[success_Se.index(20)][0]]))
    # print(' '.join([inv_dict[i] for i in output_list_Se[success_Se.index(20)]]))

    new_num_statis = dict.fromkeys(range(np.min(num_change_list_Se), np.max(num_change_list_Se)+1), 0)
    for c_num in num_change_list_Se:
        for i in range(np.min(num_change_list_Se), np.max(num_change_list_Se)+1):
            if c_num > i:
                new_num_statis[i] += 1
    fail_num = len(failed_list_Se)
    for k in new_num_statis.keys():
        new_num_statis[k] += fail_num
        new_num_statis[k] = float(new_num_statis[k]) /float(len(success_Se) + fail_num)
    print('Fail: %d' % fail_num)
    print('Percent of safe region:', new_num_statis)


    # -------TextFooler-------
    print('-------TextFooler-------')
    # mr
    with open('adv_results/adv_hownettop5_%s_%s_fail_2.3_new.pkl' % (args.task, args.target_model), 'rb') as f:
        failed_list_Te, failed_input_list, failed_time = pickle.load(f)
    with open('adv_results/adv_hownettop5_%s_%s_success_2.3_new.pkl' % (args.task, args.target_model), 'rb') as f:
        input_list_Te, _, _, success_Te, change_list_Te, num_change_list_Te, success_time_Te = pickle.load(f)
    new_success_Te = []
    new_num_change_list_Te = []
    new_change_list_Te = []
    for idx in range(len(success_Te)):
        if change_list_Te[idx] <= 0.25:
            new_success_Te.append(success_Te[idx])
            new_num_change_list_Te.append(num_change_list_Te[idx])
            new_change_list_Te.append(change_list_Te[idx])
    print('Attack num: %d, Attack success: %d, Mean of Change num: %f, Mean of Change ratio: %f ' %
          (len(success_Te) + len(failed_list_Te), len(new_success_Te),np.mean(new_num_change_list_Te), np.mean(new_change_list_Te)))


    new_num_statis = dict.fromkeys(range(np.min(new_num_change_list_Te), np.max(new_num_change_list_Te)+1), 0)
    for c_num in new_num_change_list_Te:
        for i in range(np.min(new_num_change_list_Te), np.max(new_num_change_list_Te)+1):
            if c_num > i:
                new_num_statis[i] += 1
    fail_num = len(failed_list_Te)
    for k in new_num_statis.keys():
        new_num_statis[k] += fail_num
        new_num_statis[k] = float(new_num_statis[k]) /float(len(new_success_Te) + fail_num)
    print('Fail: %d' % fail_num)
    print('Percent of safe region:', new_num_statis)

    # ------- PDP -------
    print('------- PDP -------')
    # lines = open('FindAdresult/%s_%s_MPDP2.4_ana.txt' % (args.task, args.target_model), 'r').read().splitlines()
    lines = open('FindAdresult/%s_%s_onlyDP2.3_ana.txt' % (args.task, args.target_model), 'r').read().splitlines()
    success_DP = [int(line.split(' ')[0]) for line in lines]
    change_list_DP = [float(line.split(' ')[3]) for line in lines]
    num_change_list_DP = [int(line.split(' ')[2]) for line in lines]
    failed_list_DP = get_HP_failed()

    print('Attack num: %d, Attack success: %d, Mean of Change num: %f, Mean of Change ratio: %f ' %
          (len(success_DP)+len(failed_list_DP), len(success_DP), np.mean(num_change_list_DP), np.mean(change_list_DP)))

    new_num_statis = dict.fromkeys(range(np.min(num_change_list_DP), np.max(num_change_list_DP)+1), 0)
    for c_num in num_change_list_DP:
        for i in range(np.min(num_change_list_DP), np.max(num_change_list_DP)+1):
            if c_num > i:
                new_num_statis[i] += 1
    fail_num = len(failed_list_DP)
    for k in new_num_statis.keys():
        new_num_statis[k] += fail_num
        new_num_statis[k] = float(new_num_statis[k]) /float(len(success_DP) + fail_num)
    print('Fail: %d' % fail_num)
    print('Percent of safe region:', new_num_statis)

    print('------- Comparsion result -------')
    # 组合三种攻击结果
    # ids = [idx for idx in attack_list_DP if idx in attack_list_Se and idx in attack_list_Te]
    ids = [idx for idx in success_Se if idx in new_success_Te and idx in success_DP]  # and idx in new_success_DP
    # print('Overlap successful ids in three attack methods:', len(ids))
    # list_re = list(set(success_Se).difference(set(success_DP)))
    # print(list_re)
    # list_re = list(set(new_success_Te).difference(set(success_DP)))
    # print(list_re)
    # exit(0)
    all_num_change = {}
    # f1 = open('FindAdresult/num_changes_pdpVSse_%s_%s.txt' % (args.task, args.target_model), 'w')
    # f2 = open('FindAdresult/num_changes_pdpVSte_%s_%s.txt' % (args.task, args.target_model), 'w')
    for idx in ids:
        index_Se = success_Se.index(idx)
        index_Te = new_success_Te.index(idx)
        index_DP = success_DP.index(idx)
        all_num_change[idx] = [num_change_list_Se[index_Se], new_num_change_list_Te[index_Te], num_change_list_DP[index_DP]]
        # print(num_change_list_DP[index_DP], num_change_list_Se[index_Se], file=f1)
        # print(num_change_list_DP[index_DP], num_change_list_Te[index_Te], file=f2)

    # ids = [idx for idx in success_MDP if idx in success_DP]
    # f1 = open('FindAdresult/num_changes_mdpVSdp_%s_%s.txt' % (args.task, args.target_model), 'w')
    # for idx in ids:
    #     index_mdp = success_MDP.index(idx)
    #     index_DP = success_DP.index(idx)
    #     print(num_change_list_MDP[index_mdp], num_change_list_DP[index_DP], file=f1)

    print('SemPSO')
    better_num = 0
    better_ids = []
    for idx in all_num_change.keys():
        if all_num_change[idx][0] < all_num_change[idx][1] and all_num_change[idx][0] < all_num_change[idx][2]:
            better_num += 1
            better_ids.append(idx)
    print(better_num, float(better_num) /float(len(all_num_change)))

    print('TextFooler')
    better_num = 0
    for idx in all_num_change.keys():
        if all_num_change[idx][1] < all_num_change[idx][0] and all_num_change[idx][1] < all_num_change[idx][2]:
            better_num += 1
    print(better_num, float(better_num) /float(len(all_num_change)))

    print('PDP')
    better_num = 0
    for idx in all_num_change.keys():
        if all_num_change[idx][2] < all_num_change[idx][0] and all_num_change[idx][2] < all_num_change[idx][1]:
            better_num += 1
    print(better_num, float(better_num) /float(len(all_num_change)))

    temp = {}
    for idx in better_ids:
        index_Se = success_Se.index(idx)
        index_Te = success_Te.index(idx)
        index_MDP = success_DP.index(idx)
        temp[idx] = [num_change_list_Se[index_Se], num_change_list_Te[index_Te], num_change_list_DP[index_MDP]]
    print(better_ids)
    print(temp)



def compare_snli():
    # ------- SemPSO -------
    with open('/pub/data/huangpei/SememePSO/SNLI/dataset/AD_dpso_sem_%s_success_predictTrue1000.pkl' % (args.target_model), 'rb') as f:
        input_list, true_label_list, output_list, success_Se, change_list_Se, target_list, num_change_list_Se, success_time = pickle.load(f)
    with open('/pub/data/huangpei/SememePSO/SNLI/dataset/AD_dpso_sem_%s_fail_predictTrue1000.pkl' % (args.target_model), 'rb') as f:
        failed_list_Se, _, _ = pickle.load(f)
    print(len(success_Se), np.mean(num_change_list_Se), np.mean(change_list_Se))
    new_num_statis = dict.fromkeys(range(np.min(num_change_list_Se), np.max(num_change_list_Se)+1), 0)
    for c_num in num_change_list_Se:
        for i in range(np.min(num_change_list_Se), np.max(num_change_list_Se)+1):
            if c_num > i:
                new_num_statis[i] += 1
    fail_num = len(failed_list_Se)
    for k in new_num_statis.keys():
        new_num_statis[k] += fail_num
        new_num_statis[k] = float(new_num_statis[k]) /float(len(success_Se) + fail_num)
    print(len(success_Se) + fail_num)
    print(new_num_statis)


    # ------- TextFooler -------
    with open('/pub/data/huangpei/SememePSO/SNLI/dataset/adv_hownettop5_%s_%s_success_predictTrue1000.pkl' % (args.task, args.target_model), 'rb') as f:
        input_list_Te, _, _, success_Te, change_list_Te, num_change_list_Te, success_time_Te = pickle.load(f)
    with open('/pub/data/huangpei/SememePSO/SNLI/dataset/adv_hownettop5_%s_%s_fail_predictTrue1000.pkl' % (args.task, args.target_model), 'rb') as f:
        failed_list_Te, failed_input_list, failed_time = pickle.load(f)
    # 过滤成功攻击中比例超过0.25的
    index = []
    for i in range(len(change_list_Te)):
        if change_list_Te[i]<=0.25:
            index.append(i)

    success_Te = np.array(success_Te)[index].tolist()
    change_list_Te = np.array(change_list_Te)[index].tolist()
    num_change_list_Te = np.array(num_change_list_Te)[index].tolist()
    success_time_Te = np.array(success_time_Te)[index].tolist()
    print(len(success_Te), np.mean(num_change_list_Te), np.mean(change_list_Te))

    new_num_statis = dict.fromkeys(range(np.min(num_change_list_Te), np.max(num_change_list_Te)+1), 0)
    for c_num in num_change_list_Te:
        for i in range(np.min(num_change_list_Te), np.max(num_change_list_Te)+1):
            if c_num > i:
                new_num_statis[i] += 1
    fail_num = len(failed_list_Te)
    for k in new_num_statis.keys():
        new_num_statis[k] += fail_num
        new_num_statis[k] = float(new_num_statis[k]) /float(len(success_Te) + fail_num)
    print(len(success_Te) + fail_num)
    print(new_num_statis)

    # HP
    lines = open('FindAdresult/%s_%s_MPDP_ana.txt' % (args.task, args.target_model), 'r').read().splitlines()
    success_DP = [int(line.split(' ')[0]) for line in lines]
    change_list_DP = [float(line.split(' ')[3]) for line in lines]
    num_change_list_DP = [int(line.split(' ')[2]) for line in lines]
    failed_list_DP = get_HP_failed()
    print(len(success_DP), np.mean(num_change_list_DP), np.mean(change_list_DP))

    new_num_statis = dict.fromkeys(range(np.min(num_change_list_DP), np.max(num_change_list_DP)+1), 0)
    for c_num in num_change_list_DP:
        for i in range(np.min(num_change_list_DP), np.max(num_change_list_DP)+1):
            if c_num > i:
                new_num_statis[i] += 1
    fail_num = len(failed_list_DP)
    for k in new_num_statis.keys():
        new_num_statis[k] += fail_num
        new_num_statis[k] = float(new_num_statis[k]) /float(len(success_DP) + fail_num)
    print(len(success_DP) + fail_num)
    print(new_num_statis)
    exit(0)

    # 组合
    ids = [idx for idx in success_Se if idx in success_Te and idx in success_DP]
    all_num_change = {}
    for idx in ids:
        index_Se = success_Se.index(idx)
        index_Te = success_Te.index(idx)
        index_DP = success_DP.index(idx)
        all_num_change[idx] = [num_change_list_Se[index_Se], num_change_list_Te[index_Te], num_change_list_DP[index_DP]]

    better_num = 0
    # fail_ids = []
    better_ids = []
    for idx in all_num_change.keys():
        if all_num_change[idx][0] < all_num_change[idx][1] and all_num_change[idx][0] < all_num_change[idx][2]:
            better_num += 1
            # fail_ids.append(idx)
    print(better_num, float(better_num) /float(len(all_num_change)))
    better_num = 0
    for idx in all_num_change.keys():
        if all_num_change[idx][1] < all_num_change[idx][0] and all_num_change[idx][1] < all_num_change[idx][2]:
            better_num += 1
            # fail_ids.append(idx)
    print(better_num, float(better_num) /float(len(all_num_change)))
    better_num = 0
    for idx in all_num_change.keys():
        if all_num_change[idx][2] < all_num_change[idx][0] and all_num_change[idx][2] < all_num_change[idx][1]:
            better_num += 1
            better_ids.append(idx)
    print(better_num, float(better_num) /float(len(all_num_change)))


def combing_PSO():
    new_wrong_clas_id=[]
    new_too_long_id=[]
    new_attack_list=[]
    new_failed_list_Se=[]
    new_output_list_Se=[]
    new_success_Se=[]
    new_change_list_Se=[]
    new_num_change_list_Se=[]
    new_success_time_Se=[]
    for i in range(200,1200,200):
        with open('data/adversary_training_corpora/mr/AD_dpso_sem_%s_ifvalid_%d_2.3.pkl' % (args.target_model, i), 'rb') as f:
            wrong_clas_id, too_long_id, attack_list = pickle.load(f)
        with open('data/adversary_training_corpora/mr/AD_dpso_sem_%s_fail_%d_2.3.pkl' % (args.target_model, i), 'rb') as f:
            failed_list_Se, _, _ = pickle.load(f)
        with open('data/adversary_training_corpora/mr/AD_dpso_sem_%s_success_%d_2.3.pkl' % (args.target_model, i), 'rb') as f:
            _, _, output_list_Se, success_Se, change_list_Se, _, num_change_list_Se, success_time_Se = pickle.load(f)
        new_wrong_clas_id.extend(wrong_clas_id)
        new_too_long_id.extend(too_long_id)
        new_attack_list.extend(attack_list)
        new_failed_list_Se.extend(failed_list_Se)
        new_output_list_Se.extend(output_list_Se)
        new_success_Se.extend(success_Se)
        new_change_list_Se.extend(change_list_Se)
        new_num_change_list_Se.extend(num_change_list_Se)
        new_success_time_Se.extend(success_time_Se)
    return new_wrong_clas_id, new_too_long_id, new_attack_list,new_failed_list_Se,new_output_list_Se,new_success_Se,\
           new_change_list_Se,new_num_change_list_Se,new_success_time_Se

# 将获得的对抗样本写入文件
def write_adv():
    task = 'mr'
    target_model = 'wordLSTM'
    # 待写入对抗样本
    with open('adv_results/adv_hownettop5_%s_%s_fail_2.3_new.pkl' % (task, target_model), 'rb') as f:
        failed_list, _, _ = pickle.load(f)
    with open('adv_results/adv_hownettop5_%s_%s_success_2.3_new.pkl' % (task, target_model), 'rb') as f:
        _, true_label_list, output_list, success, change_list, _, _ = pickle.load(f)
    test_size = len(failed_list)+len(success)
    # 过滤成功攻击中比例超过0.25的
    new_true_label_list = np.array(true_label_list)[np.array(change_list) <= 0.25]
    new_output_list = np.array(output_list)[np.array(change_list) <= 0.25]
    print('Test size: %d Attack success: %d' % (test_size, len(new_true_label_list)))
    filename = ('data/adversary_training_corpora/%s/test_adv.txt' % task)
    with open(filename, 'w') as w:
        for i in range(len(new_true_label_list)):
            w.write(str(new_true_label_list[i]) + ' ' + new_output_list[i] + '\n')


if __name__ == '__main__':
    # ana_TH()
    # ana_MIT()
    ana_HP()
    # ana_imdb_bert_certi()
    # ana_HP_certi()

    # combing_PSO()
    # compare_imdb()
    # compare_mr()
    # compare_snli()
    # list_re = list(set(list2).difference(set(list1)))
    # list_re.sort()
    # compare_snli()

    # arrange_result()
    #
    # print(list1)
    # print(list2)
    # print(list_re, len(list_re))

    # write_adv()
