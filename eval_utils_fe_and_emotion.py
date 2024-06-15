# This file contains the evaluation functions
import re
import jieba
import editdistance
import difflib
from fuzzywuzzy import fuzz
from collections import Counter
import numpy as np


clause_list = ['情感句', '原因句', 'emotion clause', 'cause clause']
emotion_category_list = ['happiness', 'sadness', 'anger', 'disgust', 'surprise', 'fear']


def extract_spans_extraction(task, seq):
    extractions = []
    if task in ['ecpe']:
        ecpe_extractions, ee_extractions, ce_extractions = [], [], []
        all_pt = seq.split(';')
        # each_ecpe_ex, each_ee_ex, each_ce_ex = [], [], []
        c_clas = []
        for pt in all_pt:
            pt = pt[1:-1]
            try:
                all_contence = pt.split(',')
                e_clause = all_contence[0]
                c_clauses = all_contence[1:-1] #','.join(all_contence[2:])
                e_category = all_contence[-1]
                c_clas += c_clauses
            except ValueError:
                pt.split(',')
            ecpe_extractions.append(all_contence)
            ee_extractions.append(e_clause)
            ce_extractions = c_clas
        return [ecpe_extractions, ee_extractions, ce_extractions]
    elif task in ['ee', 'ce', 'ece']:
        # print('pt    seq:', seq)
        all_pt = seq.split(';')
        for pt in all_pt:
            # print('pt:', pt)
            extractions.append(pt)
        return extractions


def extract_spans_annotation(task, seq):
    if task in ['ee', 'ce', 'ece']:
        extracted_spans = extract_sa(seq)
    elif task in ['ecpe']:
        extracted_ecpes, ex_ees, ex_ces = extract_sa_ecpe_ee_ce(seq)
        return [extracted_ecpes, ex_ees, ex_ces]
    return extracted_spans


def extract_sa(seq):
    ecps_clauses = re.findall('\[.*?\]', seq)
    ecps = []
    for ecp in ecps_clauses:
        ecps.append(ecp)
    return ecps


def extract_sa_ecpe_ee_ce(seq):
    ecps_clauses = re.findall('\[.*?\]', seq)
    ecps = []
    ees = []
    ces = []
    for ecp in ecps_clauses:
        ecps.append(ecp)
        ecp = ecp[1:-1].split(',')
        if len(ecp) == 4:
            ees.append(ecp[1])
            ces.append(ecp[1])
        elif len(ecp) == 5:
            ees.append(ecp[1])
            ces.append(ecp[3])
        elif len(ecp) > 5:
            ees.append(ecp[1])
            i = 3
            while i < len(ecp):
                ces.append(ecp[i])
                i +=2
    return ecps, ees, ces


def extract_pairs(seq):
    aps = re.findall('\[.*?\]', seq)
    aps = [ap[1:-1] for ap in aps]
    pairs = []
    for ap in aps:
        # the original sentence might have 
        try:
            at, ots = ap.split('|')
        except ValueError:
            at, ots = '', ''
        
        if ',' in ots:     # multiple ots 
            for ot in ots.split(', '):
                pairs.append((at, ot))
        else:
            pairs.append((at, ots))    
    return pairs        


def recover_terms_with_editdistance(original_term, sent):
    words = original_term.split(' ')
    if sent != (emotion_category_list or clause_list):
        sents = ''.join(sent)
        sent = sents.split(',')

    new_words = []
    for word in words:
        edit_dis = []
        for token in sent:
            edit_dis.append(editdistance.eval(word, token))
        smallest_idx = edit_dis.index(min(edit_dis))
        new_words.append(sent[smallest_idx])
    new_term = ' '.join(new_words)
    return new_term


def recover_terms_with_fuzz(original_term, sent):
    words = original_term.split(' ')
    if sent != (emotion_category_list or clause_list):
        sents = ''.join(sent)
        sent = sents.split(',')
    new_words = []
    for word in words:
        fuzz_dis = []
        for token in sent:
            fuzz_dis.append(fuzz.ratio(word, token))
        smallest_idx = fuzz_dis.index(min(fuzz_dis))
        new_words.append(sent[smallest_idx])
    new_term = ' '.join(new_words)
    return new_term


def recover_terms_with_difflib(original_term, sent):
    words = original_term.split(' ')
    if sent != (emotion_category_list or clause_list):
        sents = ''.join(sent)
        sent = sents.split(',')
    new_words = []
    for word in words:
        diff_dis = []
        for token in sent:
            diff_dis.append(difflib.SequenceMatcher(None, token, word).ratio())
        biggest_idx = diff_dis.index(max(diff_dis))
        new_words.append(sent[biggest_idx])
    new_term = ' '.join(new_words)
    return new_term


def recover_terms_with_cos_sim(original_term, sent):
    def cos_sim(str1, str2):  # str1，str2是分词后的标签列表
        co_str1 = (Counter(str1))
        co_str2 = (Counter(str2))
        p_str1 = []
        p_str2 = []
        for temp in set(str1 + str2):
            p_str1.append(co_str1[temp])
            p_str2.append(co_str2[temp])
        p_str1 = np.array(p_str1)
        p_str2 = np.array(p_str2)
        return p_str1.dot(p_str2) / (np.sqrt(p_str1.dot(p_str1)) * np.sqrt(p_str2.dot(p_str2)))

    words = original_term.split(' ')
    if sent != (emotion_category_list or clause_list):
        sents = ''.join(sent)
        sent = sents.split(',')
    new_words = []
    for word in words:
        diff_dis = []
        for token in sent:
            diff_dis.append(cos_sim(jieba.lcut(word), jieba.lcut(token)))
        biggest_idx = diff_dis.index(max(diff_dis))
        new_words.append(sent[biggest_idx])
    new_term = ' '.join(new_words)
    return new_term


def fix_ann_preds_ecpe(all_pairs, sents):
    ecpe_all_new_pairs, ee_all_new_pairs, ce_all_new_pairs = [], [], []
    for i, pairs in enumerate(all_pairs):
        new_pairs, new_ee, new_ce = [], [], []
        if pairs[0] == []:
            ecpe_all_new_pairs.append(pairs[0])
            ee_all_new_pairs.append(pairs[0])
            ce_all_new_pairs.append(pairs[0])
        else:
            for pair in pairs[0]:
                pair = pair[1:-1].split(',')
                print('pair:', pair)
                #one emotion or many emotion
                nub_emo_l = []
                for j, item_i in enumerate(pair):
                    if (item_i not in emotion_category_list) and (' ' not in item_i):
                        for cha in item_i:
                            if not '\u0e00' <= cha <= '\u9fa5':
                                pair[j] = recover_terms_with_editdistance(item_i, emotion_category_list)
                            else:
                                break
                        # pair[j] = recover_terms_with_editdistance(item_i, emotion_category_list)
                        if pair[j] in emotion_category_list:
                            nub_emo_l.append(pair[j])
                    elif item_i in emotion_category_list:
                        nub_emo_l.append(item_i)
                if len(nub_emo_l) == 1:
                    if len(pair) == 5:
                        if pair[0] not in emotion_category_list:
                            new_emo_category = recover_terms_with_editdistance(pair[0], emotion_category_list)
                        else:
                            new_emo_category = pair[0]

                        if pair[1] not in sents[i]:
                            new_emo_clause = recover_terms_with_editdistance(pair[1], sents[i])
                            new_ee.append((new_emo_clause))
                        else:
                            new_emo_clause = pair[1]
                            new_ee.append((new_emo_clause))

                        # if pair[2] not in clause_list:
                        #     new_emo_label = recover_terms_with_editdistance(pair[2], clause_list)
                        # else:
                        #     new_emo_label = pair[2]
                        for chr in pair[2]:
                            if not '\u0e00' <= chr <= '\u9fa5':
                                new_emo_label = 'emotion clause'
                            else:
                                new_emo_label = '情感句'
                        if pair[3] not in sents[i]:
                            new_cau_clause = recover_terms_with_editdistance(pair[3], sents[i])
                            new_ce.append((new_cau_clause))

                        else:
                            new_cau_clause = pair[3]
                            new_ce.append((new_cau_clause))

                        # if pair[4] not in clause_list:
                        #     new_cau_label = recover_terms_with_editdistance(pair[4], clause_list)
                        # else:
                        #     new_cau_label = pair[4]
                        for chr in pair[4]:
                            if not '\u0e00' <= chr <= '\u9fa5':
                                new_cau_label = 'cause clause'
                            else:
                                new_cau_label = '原因句'

                        new_pairs.append(('['+new_emo_category+','+new_emo_clause+','+new_emo_label+','+new_cau_clause+','+new_cau_label+']'))
                    elif len(pair) == 4:
                        if pair[0] not in emotion_category_list:
                            new_emo_category = recover_terms_with_editdistance(pair[0], emotion_category_list)
                        else:
                            new_emo_category = pair[0]

                        if pair[1] not in sents[i]:
                            new_emo_clause = recover_terms_with_editdistance(pair[1], sents[i])
                            new_ee.append((new_emo_clause))
                            new_ce.append((new_emo_clause))
                        else:
                            new_emo_clause = pair[1]
                            new_ee.append((new_emo_clause))
                            new_ce.append((new_emo_clause))
                        # if pair[2] not in clause_list:
                        #     new_emo_label = recover_terms_with_editdistance(pair[2], clause_list)
                        # else:
                        #     new_emo_label = pair[2]
                        for chr in pair[2]:
                            if not '\u0e00' <= chr <= '\u9fa5':
                                new_emo_label = 'emotion clause'
                            else:
                                new_emo_label = '情感句'

                        # if pair[3] not in clause_list:
                        #     new_cau_label = recover_terms_with_editdistance(pair[3], clause_list)
                        # else:
                        #     new_cau_label = pair[3]
                        for chr in pair[3]:
                            if not '\u0e00' <= chr <= '\u9fa5':
                                new_cau_label = 'cause clause'
                            else:
                                new_cau_label = '原因句'

                        new_pairs.append(('['+new_emo_category+','+new_emo_clause+','+new_emo_label+','+new_cau_label+']'))
                    elif len(pair) > 5:
                        if pair[0] not in emotion_category_list:
                            new_emo_category = recover_terms_with_editdistance(pair[0], emotion_category_list)
                        else:
                            new_emo_category = pair[0]

                        if pair[1] not in sents[i]:
                            new_emo_clause = recover_terms_with_editdistance(pair[1], sents[i])
                            new_ee.append((new_emo_clause))
                        else:
                            new_emo_clause = pair[1]
                            new_ee.append((new_emo_clause))

                        # if pair[2] not in clause_list:
                        #     new_emo_label = recover_terms_with_editdistance(pair[2], clause_list)
                        #     new_emo_label = 'emotion clause'
                        # else:
                        #     new_emo_label = pair[2]
                        #     new_emo_label = 'emotion clause'
                        for chr in pair[2]:
                            if not '\u0e00' <= chr <= '\u9fa5':
                                new_emo_label = 'emotion clause'
                            else:
                                new_emo_label = '情感句'
                        c_l = 3
                        all_caus = ''
                        while c_l < len(pair):
                            if pair[c_l] in clause_list:
                                c_l -= 1

                            if (pair[c_l] not in sents[i]) and (pair[c_l] not in clause_list):
                                new_cau_clause = recover_terms_with_editdistance(pair[c_l], sents[i])
                                new_ce.append((new_cau_clause))
                            else:
                                new_cau_clause = pair[c_l]
                                new_ce.append(new_cau_clause)

                            for chr in pair[c_l+1]:
                                if not '\u0e00' <= chr <= '\u9fa5':
                                    new_cau_label = 'cause clause'
                                    break
                                else:
                                    new_cau_label = '原因句'
                            # if pair[c_l+1] not in clause_list:
                            #     new_cau_label = recover_terms_with_editdistance(pair[c_l+1], clause_list)
                            # else:
                            #     new_cau_label = pair[c_l+1]
                            # new_cau_label = 'cause label'
                            c_l += 2
                            all_caus += ','+new_cau_clause+','+new_cau_label

                        new_pairs.append(('['+new_emo_category+','+new_emo_clause+','+new_emo_label+all_caus+']'))
                        # print('new_pair:', new_pairs)
                elif len(nub_emo_l) > 1:
                    new_emo_category = ','.join(nub_emo_l)
                    # print('new_emo_category:', new_emo_category)
                    ##[sadness,是他对自己生存期的忧虑,情感句,老吴可能等不到妻子随迁入户深圳,原因句]
                    if len(nub_emo_l) + 4 == len(pair): ## emo_category, emo_clause, emo_label, cau_clause, cau_label
                        if pair[len(nub_emo_l)] not in sents[i]:
                            new_emo_clause = recover_terms_with_editdistance(pair[len(nub_emo_l)], sents[i])
                            new_ee.append((new_emo_clause))
                        else:
                            new_emo_clause = pair[len(nub_emo_l)]
                            new_ee.append((new_emo_clause))

                        # if pair[len(nub_emo_l)+1] not in clause_list:
                        #     new_emo_label = recover_terms_with_editdistance(pair[len(nub_emo_l)+1], clause_list)
                        # else:
                        #     new_emo_label = pair[len(nub_emo_l)+1]

                        for chr in pair[len(nub_emo_l)+1]:
                            if not '\u0e00' <= chr <= '\u9fa5':
                                new_emo_label = 'emotion clause'
                            else:
                                new_emo_label = '情感句'

                        if pair[len(nub_emo_l)+2] not in sents[i]:
                            new_cau_clause = recover_terms_with_editdistance(pair[len(nub_emo_l)+2], sents[i])
                            new_ce.append((new_cau_clause))

                        else:
                            new_cau_clause = pair[len(nub_emo_l)+2]
                            new_ce.append((new_cau_clause))

                        # if pair[len(nub_emo_l)+3] not in clause_list:
                        #     new_cau_label = recover_terms_with_editdistance(pair[len(nub_emo_l)+3], clause_list)
                        # else:
                        #     new_cau_label = pair[len(nub_emo_l)+3]
                        for chr in pair[len(nub_emo_l)+3]:
                            if not '\u0e00' <= chr <= '\u9fa5':
                                new_cau_label = 'cause clause'
                            else:
                                new_cau_label = '原因句'
                        new_pairs.append(('['+new_emo_category+','+new_emo_clause+','+new_emo_label+','+new_cau_clause+','+new_cau_label+']'))
                    ##[happiness, 当初获得入户指标的那份欣喜, 情感句, 原因句]
                    elif len(nub_emo_l) + 3 == len(pair):
                        # print(pair[len(nub_emo_l)])
                        if pair[len(nub_emo_l)] not in sents[i]:
                            new_emo_clause = recover_terms_with_editdistance(pair[len(nub_emo_l)], sents[i])
                            new_ee.append((new_emo_clause))
                            new_ce.append((new_emo_clause))
                        else:
                            new_emo_clause = pair[len(nub_emo_l)]
                            new_ee.append((new_emo_clause))
                            new_ce.append((new_emo_clause))

                        # if pair[len(nub_emo_l)+1] not in clause_list:
                        #     new_emo_label = recover_terms_with_editdistance(pair[len(nub_emo_l)+1], clause_list)
                        # else:
                        #     new_emo_label = pair[len(nub_emo_l)+1]
                        for chr in pair[len(nub_emo_l)+1]:
                            if not '\u0e00' <= chr <= '\u9fa5':
                                new_emo_label = 'emotion clause'
                            else:
                                new_emo_label = '情感句'
                        for chr in pair[len(nub_emo_l)+2]:
                            if not '\u0e00' <= chr <= '\u9fa5':
                                new_cau_label = 'cause clause'
                            else:
                                new_cau_label = '原因句'

                        # if pair[len(nub_emo_l)+2] not in clause_list:
                        #     new_cau_label = recover_terms_with_editdistance(pair[len(nub_emo_l)+2], clause_list)
                        # else:
                        #     new_cau_label = pair[len(nub_emo_l)+2]
                        new_pairs.append(('['+new_emo_category+','+new_emo_clause+','+new_emo_label+','+new_cau_label+']'))
                    ##[sadness,anger,无奈才选择跳楼轻生,情感句,该女子是由于对方拖欠工程款,原因句,家中又急需用钱,原因句,生活压力大,原因句]
                    elif (len(nub_emo_l)+4 > 6) and (len(pair) > 6):
                        if pair[len(nub_emo_l)] not in sents[i]:
                            new_emo_clause = recover_terms_with_editdistance(pair[len(nub_emo_l)], sents[i])
                            new_ee.append((new_emo_clause))
                        else:
                            new_emo_clause = pair[len(nub_emo_l)]
                            new_ee.append((new_emo_clause))

                        # if pair[len(nub_emo_l)+1] not in clause_list:
                        #     new_emo_label = recover_terms_with_editdistance(pair[len(nub_emo_l)+1], clause_list)
                        # else:
                        #     new_emo_label = pair[len(nub_emo_l)+1]
                        for chr in pair[len(nub_emo_l)+1]:
                            if not '\u0e00' <= chr <= '\u9fa5':
                                new_emo_label = 'emotion clause'
                            else:
                                new_emo_label = '情感句'
                        c_l = len(nub_emo_l) + 1
                        all_caus = ''
                        while c_l < len(pair):
                            if pair[c_l] in clause_list:
                                c_l -= 1
                            if pair[c_l] not in sents[i]:
                                new_cau_clause = recover_terms_with_editdistance(pair[c_l], sents[i])
                                new_ce.append((new_cau_clause))
                            else:
                                new_cau_clause = pair[c_l]
                                new_ce.append(new_cau_clause)
                            # if pair[c_l + 1] not in clause_list:
                            #     new_cau_label = recover_terms_with_editdistance(pair[c_l + 1], clause_list)
                            # else:
                            #     new_cau_label = pair[c_l + 1]
                            for chr in pair[c_l + 1]:
                                if not '\u0e00' <= chr <= '\u9fa5':
                                    new_cau_label = 'cause clause'
                                else:
                                    new_cau_label = '原因句'
                            c_l += 2
                            all_caus += ',' + new_cau_clause + ',' + new_cau_label
                        new_pairs.append(('[' + new_emo_category + ',' + new_emo_clause + ',' + new_emo_label + all_caus + ']'))
                # print(pair, '>>>>>', word_and_sentiment)
                # print(all_target_pairs[i])
            print('new_pairs:', new_pairs)
            ecpe_all_new_pairs.append(new_pairs)
            ee_all_new_pairs.append(new_ee)
            ce_all_new_pairs.append(new_ce)
    return [ecpe_all_new_pairs, ee_all_new_pairs, ce_all_new_pairs]


def fix_ext_preds_ecpe(all_pairs, sents):
    ecpe_all_new_pairs, ee_all_new_pairs, ce_all_new_pairs = [], [], []
    for i, pairs in enumerate(all_pairs):
        new_pairs, new_ee, new_ce = [], [], []
        if pairs[0] == []:
            ecpe_all_new_pairs.append(pairs[0])
            ee_all_new_pairs.append(pairs[0])
            ce_all_new_pairs.append(pairs[0])
        else:
            # all_emo_clauses, all_cau_clauses = [], []
            for pair in pairs[0]:
                # pair = pair[1:-1].split(',')
                # AT not in the original sentence
                if len(pair) == 3:
                    if pair[0] not in sents[i]:
                        new_emo_clause = recover_terms_with_cos_sim(pair[0], sents[i])
                        # new_emo_clause = recover_terms_with_fuzz(pair[0], sents[i])
                        # new_emo_clause = recover_terms_with_difflib(pair[0], sents[i])
                        # new_emo_clause = recover_terms_with_editdistance(pair[0], sents[i])

                        new_ee.append((new_emo_clause))
                    else:
                        new_emo_clause = pair[0]
                        new_ee.append((new_emo_clause))

                    if pair[1] not in sents[i]:
                        new_cau_clause = recover_terms_with_editdistance(pair[1], sents[i])
                        new_ce.append((new_cau_clause))
                    else:
                        new_cau_clause = pair[1]
                        new_ce.append((new_cau_clause))

                    if pair[2] not in emotion_category_list:
                        # new_emo_cat = recover_terms_with_cos_sim(pair[2], emotion_category_list)
                        # new_emo_cat = recover_terms_with_fuzz(pair[2], emotion_category_list)
                        new_emo_cat = recover_terms_with_difflib(pair[2], emotion_category_list)
                        # new_emo_cat = recover_terms_with_editdistance(pair[2], emotion_category_list)
                    else:
                        new_emo_cat = pair[2]


                    new_pairs.append(([new_emo_clause, new_cau_clause, new_emo_cat]))

                elif len(pair) > 3:
                    if pair[0] not in sents[i]:
                        new_emo_clause = recover_terms_with_editdistance(pair[0], sents[i])
                        new_ee.append((new_emo_clause))
                    else:
                        new_emo_clause = pair[0]
                        new_ee.append((new_emo_clause))

                    if pair[-1] not in emotion_category_list:
                        new_emo_cat = recover_terms_with_difflib(pair[1], emotion_category_list)
                    else:
                        new_emo_cat = pair[1]

                    c_l = 1
                    all_caus = []
                    while c_l < len(pair)-1:
                        if pair[c_l] not in sents[i]:
                            new_cau_clause = recover_terms_with_editdistance(pair[c_l], sents[i])
                            new_ce.append((new_cau_clause))
                        else:
                            new_cau_clause = pair[c_l]
                            new_ce.append(new_cau_clause)
                        c_l += 1

                        all_caus.append(new_cau_clause)
                    new_pairs.append([new_emo_clause]+all_caus+[new_emo_cat])
                # print(pair, '>>>>>', word_and_sentiment)
                # print(all_target_pairs[i])
                # new_ee = all_emo_clauses
                # new_ce = all_cau_clauses
            ecpe_all_new_pairs.append(new_pairs)
            ee_all_new_pairs.append(new_ee)
            ce_all_new_pairs.append(new_ce)
    return [ecpe_all_new_pairs, ee_all_new_pairs, ce_all_new_pairs]


def fix_ann_preds_ee(all_pairs, sents):
    all_new_pairs = []
    for i, pairs in enumerate(all_pairs):
        new_pairs = []
        if pairs == []:
            all_new_pairs.append(pairs)
        else:
            for pair in pairs:
                pair = pair[1:-1].split(',')
                # AT not in the original sentence
                if len(pair) == 2:
                    if pair[0] not in sents[i]:
                        new_emo_clause = recover_terms_with_editdistance(pair[0], sents[i])
                    else:
                        new_emo_clause = pair[0]

                    # if pair[1] not in clause_list:
                    #     new_emo_label = recover_terms_with_editdistance(pair[1], clause_list)
                    # else:
                    #     new_emo_label = pair[1]
                    for chr in pair[1]:
                        if not '\u0e00' <= chr <= '\u9fa5':
                            new_emo_label = 'emotion clause'
                        else:
                            new_emo_label = '情感句'

                    new_pairs.append(('['+new_emo_clause+','+new_emo_label+']'))
                # print(pair, '>>>>>', word_and_sentiment)
                # print(all_target_pairs[i])
            all_new_pairs.append(new_pairs)

    return all_new_pairs


def fix_ext_preds_ee(all_pairs, sents):
    all_new_pairs = []
    for i, pairs in enumerate(all_pairs):
        new_pairs = []
        if pairs == []:
            all_new_pairs.append(pairs)
        else:
            for pair in pairs:
                pair = pair[1:-1].split(',')
                # AT not in the original sentence
                if len(pair) == 1:
                    if pair[0] not in sents[i]:
                        new_emo_clause = recover_terms_with_editdistance(pair[0], sents[i])
                    else:
                        new_emo_clause = pair[0]

                    new_pairs.append(('['+new_emo_clause+']'))
                # print(pair, '>>>>>', word_and_sentiment)
                # print(all_target_pairs[i])
            all_new_pairs.append(new_pairs)

    return all_new_pairs


def fix_ann_preds_ce(all_pairs, sents):
    all_new_pairs = []
    for i, pairs in enumerate(all_pairs):
        new_pairs = []
        if pairs == []:
            all_new_pairs.append(pairs)
        else:
            for pair in pairs:
                pair = pair[1:-1].split(',')
                # AT not in the original sentence
                if len(pair) == 2:
                    if pair[0] not in sents[i]:
                        new_cau_clause = recover_terms_with_editdistance(pair[0], sents[i])
                    else:
                        new_cau_clause = pair[0]

                    # if pair[1] not in clause_list:
                    #     new_cau_label = recover_terms_with_editdistance(pair[1], clause_list)
                    # else:
                    #     new_cau_label = pair[1]

                    for chr in pair[1]:
                        if not '\u0e00' <= chr <= '\u9fa5':
                            new_cau_label = 'cause clause'
                        else:
                            new_cau_label = '原因句'

                    new_pairs.append(('['+new_cau_clause+','+new_cau_label+']'))

                # print(pair, '>>>>>', word_and_sentiment)
                # print(all_target_pairs[i])
            all_new_pairs.append(new_pairs)

    return all_new_pairs


def fix_ext_preds_ce(all_pairs, sents):
    all_new_pairs = []
    for i, pairs in enumerate(all_pairs):
        new_pairs = []
        if pairs == []:
            all_new_pairs.append(pairs)
        else:
            for pair in pairs:
                pair = pair[1:-1].split(',')
                # AT not in the original sentence
                if len(pair) == 1:
                    if pair[0] not in sents[i]:
                        new_cau_clause = recover_terms_with_editdistance(pair[0], sents[i])
                    else:
                        new_cau_clause = pair[0]

                    new_pairs.append(('['+new_cau_clause+']'))

                # print(pair, '>>>>>', word_and_sentiment)
                # print(all_target_pairs[i])
            all_new_pairs.append(new_pairs)

    return all_new_pairs


def fix_ann_preds_ece(all_pairs, sents):
    all_new_pairs = []
    for i, pairs in enumerate(all_pairs):
        new_pairs = []
        if pairs == []:
            all_new_pairs.append(pairs)
        else:
            for pair in pairs:
                pair = pair[1:-1].split(',')
                # AT not in the original sentence
                if len(pair) == 2:
                    if pair[0] not in emotion_category_list:
                        new_emo_cat = recover_terms_with_editdistance(pair[0], emotion_category_list)
                    else:
                        new_emo_cat = pair[0]

                    if pair[1] not in sents[i]:
                        new_cau_clause = recover_terms_with_editdistance(pair[1], sents[i])
                    else:
                        new_cau_clause = pair[1]

                    new_pairs.append(('['+new_emo_cat+','+new_cau_clause+']'))

                # print(pair, '>>>>>', word_and_sentiment)
                # print(all_target_pairs[i])
            all_new_pairs.append(new_pairs)

    return all_new_pairs


def fix_ext_preds_ece(all_pairs, sents):
    all_new_pairs = []
    for i, pairs in enumerate(all_pairs):
        new_pairs = []
        if pairs == []:
            all_new_pairs.append(pairs)
        else:
            for pair in pairs:
                pair = pair[1:-1].split(',')
                # AT not in the original sentence
                if len(pair) == 1:
                    if pair[0] not in sents[i]:
                        new_cau_clause = recover_terms_with_editdistance(pair[0], sents[i])
                    else:
                        new_cau_clause = pair[0]

                    new_pairs.append(('['+new_cau_clause+']'))

                # print(pair, '>>>>>', word_and_sentiment)
                # print(all_target_pairs[i])
            all_new_pairs.append(new_pairs)

    return all_new_pairs


def fix_pred_with_editdistance(all_predictions, sents, task, io_format):
    if task == 'ecpe':
        if io_format == 'annotation':
            fixed_preds = fix_ann_preds_ecpe(all_predictions, sents)
        elif io_format == 'extraction':
            fixed_preds = fix_ext_preds_ecpe(all_predictions, sents)
    elif task == 'ee':
        if io_format == 'annotation':
            fixed_preds = fix_ann_preds_ee(all_predictions, sents)
        elif io_format == 'extraction':
            fixed_preds = fix_ext_preds_ee(all_predictions, sents)
    elif task == 'ce':
        if io_format == 'annotation':
            fixed_preds = fix_ann_preds_ce(all_predictions, sents)
        elif io_format == 'extraction':
            fixed_preds = fix_ext_preds_ce(all_predictions, sents)
    elif task == 'ece':
        if io_format == 'annotation':
            fixed_preds = fix_ann_preds_ece(all_predictions, sents)
        elif io_format == 'extraction':
            fixed_preds = fix_ext_preds_ece(all_predictions, sents)

    else:
        print("*** Unimplemented Error ***")
        fixed_preds = all_predictions

    return fixed_preds


def compute_f1_scores(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold pairs/triplets
    The input needs to be already processed
    """
    # number of true postive, gold standard, predicted aspect terms
    n_tp, n_gold, n_pred = 0, 0, 0
    # print('pred_pt:', pred_pt)
    # print('gold_pt:', gold_pt)

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1

    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}

    return scores


def compute_scores(pred_seqs, gold_seqs, sents, io_format, task):
    """
    compute metrics for multiple tasks
    """
    assert len(pred_seqs) == len(gold_seqs) 
    num_samples = len(gold_seqs)

    all_labels, all_predictions = [], []

    for i in range(num_samples):
        if task in ['ecpe']:
            if io_format == 'annotation':
                gold_list = extract_spans_annotation(task, gold_seqs[i])
                pred_list = extract_spans_annotation(task, pred_seqs[i])
            elif io_format == 'extraction':
                gold_list = extract_spans_extraction(task, gold_seqs[i])
                pred_list = extract_spans_extraction(task, pred_seqs[i])
            all_labels.append(gold_list)
            all_predictions.append(pred_list)
        elif task in ['ee', 'ce', 'ece']:
            if io_format == 'annotation':
                gold_list = extract_spans_annotation(task, gold_seqs[i])
                pred_list = extract_spans_annotation(task, pred_seqs[i])

            elif io_format == 'extraction':
                gold_list = extract_spans_extraction(task, gold_seqs[i])
                pred_list = extract_spans_extraction(task, pred_seqs[i])

            all_labels.append(gold_list)
            all_predictions.append(pred_list)

    if task in ['ee', 'ce', 'ece']:
        # print('all_predictions:', all_predictions)
        print("\nResults of raw output")
        raw_scores = compute_f1_scores(all_predictions, all_labels)
        print(raw_scores)

        # fix the issues due to generation
        all_predictions_fixed = fix_pred_with_editdistance(all_predictions, sents, task, io_format)
        # print('all_predictions_fixed:', all_predictions_fixed)

        print("\nResults of fixed output")
        fixed_scores = compute_f1_scores(all_predictions_fixed, all_labels)
        print(fixed_scores)

        log_file_path = f"results_log/{task}-{io_format}.txt"
        with open(log_file_path, "a+", encoding='utf-8') as f:
            f.write(str(all_predictions) + '\n' + str(all_predictions_fixed))

        return raw_scores, fixed_scores, all_labels, all_predictions, all_predictions_fixed
    elif task in ['ecpe']:
        print("\nResults of raw output")
        ecpe_all_p, ee_all_p, ce_all_p = [], [], []
        ecpe_all_l, ee_all_l, ce_all_l = [], [], []
        # print('all_labels:', all_labels)
        for i, tri_re in enumerate(all_predictions):
            ecpe_all_p.append(tri_re[0])
            ee_all_p.append(tri_re[1])
            ce_all_p.append(tri_re[2])
        for i, tri_label in enumerate(all_labels):
            ecpe_all_l.append(tri_label[0])
            ee_all_l.append(tri_label[1])
            ce_all_l.append(tri_label[2])
        raw_scores_ecpe = compute_f1_scores(ecpe_all_p, ecpe_all_l)
        raw_scores_ee = compute_f1_scores(ee_all_p, ee_all_l)
        raw_scores_ce = compute_f1_scores(ce_all_p, ce_all_l)
        print('raw_scores_ecpe：', raw_scores_ecpe)
        print('raw_scores_ee:', raw_scores_ee)
        print('raw_scores_ce:', raw_scores_ce)
        # raw_scores = compute_f1_scores(all_predictions, all_labels)
        # print(raw_scores)
        # print('all_predictions:', all_predictions)

        # fix the issues due to generation
        all_predictions_fixed = fix_pred_with_editdistance(all_predictions, sents, task, io_format)
        # print('all_predictions_fixed:', all_predictions_fixed)
        print("\nResults of fixed output")
        fixed_scores_ecpe = compute_f1_scores(all_predictions_fixed[0], ecpe_all_l)
        print('fixed_scores_ecpe:', fixed_scores_ecpe)
        fixed_scores_ee = compute_f1_scores(all_predictions_fixed[1], ee_all_l)
        print('fixed_scores_ee:', fixed_scores_ee)
        fixed_scores_ce = compute_f1_scores(all_predictions_fixed[2], ce_all_l)
        print('fixed_scores_ce:', fixed_scores_ce)

        log_file_path = f"results_log/{task}-{io_format}.txt"
        with open(log_file_path, "a+", encoding='utf-8') as f:
            f.write(str(all_predictions) + '\n' + str(all_predictions_fixed))

        return raw_scores_ecpe, raw_scores_ee, raw_scores_ce, fixed_scores_ecpe, fixed_scores_ee, fixed_scores_ce, \
               all_labels, all_predictions, all_predictions_fixed


# pred_seqs = [
#     '[surprise,to Alice "s great surprise,cause clause,the Duchess "s voice died away,cause clause,even in the middle of her favourite word moral," and the arm that was linked into hers began to tremble. Alice looked up,cause clause]']
#
# gold_seqs = [
#     '[surprise,to Alice "s great surprise,emotion clause,the Duchess "s voice died away,cause clause,even in the middle of her favourite word moral," and the arm that was linked into hers began to tremble. Alice looked up,cause clause]']
#
# sents = ['to Alice "s great surprise,the Duchess "s voice died away,even in the middle of her favourite word moral," and the arm that was linked into hers began to tremble. Alice looked up']
#
# in_format = 'annotation'
# task = 'ecpe'
# raw_scores, fixed_scores, all_labels, all_preds, all_preds_fixed = compute_scores(pred_seqs, gold_seqs, sents, in_format, task)
#




# pred_seqs = [
#              # '[happiness,当初获得入户指标的那份欣喜,情感句,原因句];[sadness,是他对自己生存期的忧虑,情感句,老吴可能等不到妻子随迁入户深圳,原因句]',
#              # '[happiness,激动地对中新网记者说,情感句,国家公安部国家工商总局国家科学技术委员会科技部卫生部国家发展改革委员会等部委均接受并采纳过的我的建议,原因句]',
#              # '[sadness,小男孩胸口蔫瘦得让人心疼,情感句,原因句]',
#              '[sadness,无奈才选择跳楼轻生,情感句,该女子是由于对方拖欠工程款,原因句,家中又急需用钱,原因句,生活压力大,原因句]']
#
# gold_seqs = [
#              # '[happiness,当初获得入户指标的那份欣喜,情感句,原因句];[sadness,是他对自己生存期的忧虑,情感句,老吴可能等不到妻子随迁入户深圳,原因句]',
#              # '[happiness,激动地对中新网记者说,情感句,国家公安部国家工商总局国家科学技术委员会科技部卫生部国家发展改革委员会等部委均接受并采纳过的我的建议,原因句]',
#              # '[sadness,小男孩胸口蔫瘦得让人心疼,情感句,原因句]',
#              '[sadness,无奈才选择跳楼轻生,情感句,该女子是由于对方拖欠工程款,原因句,家中又急需用钱,原因句,生活压力大,原因句]']
#
# sents = [#'2013年6月,在深圳打拼10年的吴树梁终于拿到大红的深圳市户口,儿子吴同也随之迁入深圳,但妻子丁维清却必须等候吴树梁入户满两年才能随迁,半年后,当初获得入户指标的那份欣喜,因为老吴患上肺癌晚期的噩耗而荡然无存,取而代之的,是他对自己生存期的忧虑,医生的判决是36个月,这意味着,老吴可能等不到妻子随迁入户深圳',
# #         '当我看到建议被采纳,部委领导写给我的回信时,我知道我正在为这个国家的发展尽着一份力量,27日,河北省邢台钢铁有限公司的普通工人白金跃,拿着历年来国家各部委反馈给他的感谢信,激动地对中新网记者说,27年来,国家公安部国家工商总局国家科学技术委员会科技部卫生部国家发展改革委员会等部委均接受并采纳过的我的建议',
# #         '2002年6月3日上午,当值的曾友蔚接报,狮山镇小塘走马营村一树林里有一名年仅2岁多的小男孩躺在草地上,无人认领,曾友蔚立即赶到现场处置,只见林中的草地上,小男孩被包在一条毛巾里,很孱弱,不哭也不闹,小眼珠子静静地望着眼前的警察叔叔,曾友蔚打开包裹着的围巾,小男孩胸口蔫瘦得让人心疼,贴身处有一张写着出生年月的纸条和一封利是,曾友蔚意识到,这可能是个因病被弃的孩子',
#         '为尽快将女子救下,指挥员立即制订了救援方案,第一组在楼下铺设救生气垫,并对周围无关人员进行疏散,另一组队员快速爬上6楼,在楼内对女子进行劝说,劝说过程中,消防官兵了解到,该女子是由于对方拖欠工程款,家中又急需用钱,生活压力大,无奈才选择跳楼轻生']
#
# in_format = 'annotation'
# task = 'ecpe'
# raw_scores, fixed_scores, all_labels, all_preds, all_preds_fixed = compute_scores(pred_seqs, gold_seqs, sents, in_format, task)
# #




# pred_seqs = [
#              '[当初获得入户指标的那份欣喜,happiness,当初获得入户指标的那份欣喜];[是他对自己生存期的忧虑,sadness,老吴可能等不到妻子随迁入户深圳]',
#              '[激动地对中新网记者说,happiness,国家公安部国家工商总局国家科学技术委员会科技部卫生部国家发展改革委员会等部委均接受并采纳过的我的建议]',
#              '[小男孩胸口蔫瘦得让人心疼,sadness,小男孩胸口蔫瘦得让人心疼]',
#              '[无奈才选择跳楼轻生,sadness,该女子是由于对方拖欠工程款,家中又急需用钱,生活压力大]']
#
# gold_seqs = [
#              '[当初获得入户指标的那份欣喜,happiness,当初获得入户指标的那份欣喜];[是他对自己生存期的忧虑,sadness,老吴可能等不到妻子随迁入户深圳]',
#              '[激动地对中新网记者说,happiness,国家公安部国家工商总局国家科学技术委员会科技部卫生部国家发展改革委员会等部委均接受并采纳过的我的建议]',
#              '[小男孩胸口蔫瘦得让人心疼,sadness,小男孩胸口蔫瘦得让人心疼]',
#              '[无奈才选择跳楼轻生,sadness,该女子是由于对方拖欠工程款,家中又急需用钱,生活压力大]']

# pred_seqs = [
#              '[当初获得入户指标的那份欣慰,获得入户指标的欣喜,happy];[对自己生存的忧虑,老吴可能等不到妻子随迁入户深圳,sad]',
#              '[激动地对中新网记者说,国家公安部国家工商总局国家科学技术委员会科技部卫生部国家发展改革委员会等部委均接受并采纳过的我的建议,happiness]',
#              '[小男孩胸口蔫瘦得让人心疼,小男孩胸口蔫瘦得让人心疼,sadness]',
#              '[无奈才选择跳楼轻生,该女子是由于对方拖欠工程款,家中又急需用钱,生活压力大,sadness]',
#              '[并 希望 通过 本报 向 侯马市 公安局 向 侯马 志愿者 团队 向 所有 关心 帮助 他们 全家 寻找 小鑫鑫 的 好心人 表示感谢,并 希望 通过 本报 向 侯马市 公安局 向 侯马 志愿者 团队 向 所有 关心 帮助 他们 全家 寻找 小鑫鑫 的 好心人 表示感谢,happiness']
# gold_seqs = [
#              '[当初获得入户指标的那份欣喜,当初获得入户指标的那份欣喜,happiness];[是他对自己生存期的忧虑,老吴可能等不到妻子随迁入户深圳,sadness]',
#              '[激动地对中新网记者说,国家公安部国家工商总局国家科学技术委员会科技部卫生部国家发展改革委员会等部委均接受并采纳过的我的建议,happiness]',
#              '[小男孩胸口蔫瘦得让人心疼,小男孩胸口蔫瘦得让人心疼,sadness]',
#              '[无奈才选择跳楼轻生,该女子是由于对方拖欠工程款,家中又急需用钱,生活压力大,sadness]',
#              '[并 希望 通过 本报 向 侯马市 公安局 向 侯马 志愿者 团队 向 所有 关心 帮助 他们 全家 寻找 小鑫鑫 的 好心人 表示感谢,并 希望 通过 本报 向 侯马市 公安局 向 侯马 志愿者 团队 向 所有 关心 帮助 他们 全家 寻找 小鑫鑫 的 好心人 表示感谢,happiness']
#
# sents = ['2013年6月,在深圳打拼10年的吴树梁终于拿到大红的深圳市户口,儿子吴同也随之迁入深圳,但妻子丁维清却必须等候吴树梁入户满两年才能随迁,半年后,当初获得入户指标的那份欣喜,因为老吴患上肺癌晚期的噩耗而荡然无存,取而代之的,是他对自己生存期的忧虑,医生的判决是36个月,这意味着,老吴可能等不到妻子随迁入户深圳',
#         '当我看到建议被采纳,部委领导写给我的回信时,我知道我正在为这个国家的发展尽着一份力量,27日,河北省邢台钢铁有限公司的普通工人白金跃,拿着历年来国家各部委反馈给他的感谢信,激动地对中新网记者说,27年来,国家公安部国家工商总局国家科学技术委员会科技部卫生部国家发展改革委员会等部委均接受并采纳过的我的建议',
#         '2002年6月3日上午,当值的曾友蔚接报,狮山镇小塘走马营村一树林里有一名年仅2岁多的小男孩躺在草地上,无人认领,曾友蔚立即赶到现场处置,只见林中的草地上,小男孩被包在一条毛巾里,很孱弱,不哭也不闹,小眼珠子静静地望着眼前的警察叔叔,曾友蔚打开包裹着的围巾,小男孩胸口蔫瘦得让人心疼,贴身处有一张写着出生年月的纸条和一封利是,曾友蔚意识到,这可能是个因病被弃的孩子',
#         '为尽快将女子救下,指挥员立即制订了救援方案,第一组在楼下铺设救生气垫,并对周围无关人员进行疏散,另一组队员快速爬上6楼,在楼内对女子进行劝说,劝说过程中,消防官兵了解到,该女子是由于对方拖欠工程款,家中又急需用钱,生活压力大,无奈才选择跳楼轻生',
#         '2 月 13 日晚,记者 在 李世铭家 见到 被 拐骗 45 天 回家 的 李沐羿鑫,李世铭 说,孩子 回来 后 不敢 与 陌生人 说话,比 以前 胆子 小 了 许多,他们 会 用 全部 的 爱 抚慰 孩子 幼小 的 心灵,让 孩子 更 健康 更 阳光 地 成长,并 希望 通过 本报 向 侯马市 公安局 向 侯马 志愿者 团队 向 所有 关心 帮助 他们 全家 寻找 小鑫鑫 的 好心人 表示感谢']
#
# in_format = 'extraction'
# task = 'ecpe'
# raw_scores, fixed_scores, all_labels, all_preds, all_preds_fixed = compute_scores(pred_seqs, gold_seqs, sents, in_format, task)
