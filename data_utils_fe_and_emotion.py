# This file contains all data loading and transformation functions
import time
from os.path import join
from torch.utils.data import Dataset

senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
TRAIN_FILE = 'fold%s_train.txt'
VALID_FILE = 'fold%s_val.txt'
TEST_FILE = 'fold%s_test.txt'


def read_sina_line_examples_from_file(data_path):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    doc_id_list, doc_len_list, doc_sents_list, doc_ecps_list, doc_emo_list, doc_cau_list, doc_emocat_list = [], [], [], [], [], [], []
    inputFile = open(data_path, 'r', encoding='utf-8')
    while True:
        line = inputFile.readline()
        if line == '':
            break
        line = line.strip().split()
        doc_id = line[0]
        doc_id_list.append(doc_id)
        doc_len = int(line[1])
        doc_len_list.append(doc_len)

        ecps = eval('[' + inputFile.readline().strip() + ']')
        doc_ecps_list.append(ecps)
        emo_l, cau_l = [], []
        for pi in ecps:
            if pi[0] not in emo_l:
                emo_l.append(pi[0])
            if pi[1] not in cau_l:
                cau_l.append(pi[1])
        doc_emo_list.append(emo_l)
        doc_cau_list.append(cau_l)

        contents_l, doc_emocat_l = [], []
        for i in range(doc_len):
            clause_line = inputFile.readline().strip().split(',')
            emo_cat = clause_line[1]
            if emo_cat != 'null':
                doc_emocat_l.append(emo_cat)
            content = clause_line[-1].replace(' ', '')
            contents_l.append(content)
        doc_emocat_list.append(doc_emocat_l)
        doc_sents_list.append(contents_l)
    return doc_id_list, doc_len_list, doc_sents_list, doc_ecps_list, doc_emo_list, doc_cau_list, doc_emocat_list


def read_ntcir_line_examples_from_file(data_path):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    doc_id_list, doc_len_list, doc_sents_list, doc_ecps_list, doc_emo_list, doc_cau_list, doc_emocat_list = [], [], [], [], [], [], []
    inputFile = open(data_path, 'r')
    while True:
        line = inputFile.readline()
        if line == '':
            break
        line = line.strip().split()
        doc_id = line[0]
        doc_id_list.append(doc_id)
        doc_len = int(line[1])
        doc_len_list.append(doc_len)

        ecps = eval('[' + inputFile.readline().strip() + ']')
        doc_ecps_list.append(ecps)
        emo_l, cau_l = [], []
        for pi in ecps:
            if pi[0] not in emo_l:
                emo_l.append(pi[0])
            if pi[1] not in cau_l:
                cau_l.append(pi[1])
        doc_emo_list.append(emo_l)
        doc_cau_list.append(cau_l)

        contents_l, doc_emocat_l = [], []
        for i in range(doc_len):
            clause_line = inputFile.readline().strip().split(',')
            emo_cat = clause_line[1]
            if emo_cat != 'null':
                doc_emocat_l.append(emo_cat)
            content = clause_line[-1].replace("'", '"')
            contents_l.append(content)
        doc_emocat_list.append(doc_emocat_l)
        doc_sents_list.append(contents_l)
    return doc_id_list, doc_len_list, doc_sents_list, doc_ecps_list, doc_emo_list, doc_cau_list, doc_emocat_list


def get_sina_annotated_ecpe_targets(sents, labels, doc_emocat):
    annotated_targets = []
    num_sents = len(sents)
    for i in range(num_sents):
        tuples = labels[i]
        emocats = doc_emocat[i]
        # tup: ([7, 9])\(1, 2), (1, 3)
        emo_all_l, cau_all_l = [], []
        for tup in tuples:
            emo_id, cau_id = tup[0], tup[1]
            if emo_id not in emo_all_l:
                emo_all_l.append(emo_id)
            if cau_id not in cau_all_l:
                cau_all_l.append(cau_id)
        for emo_i in range(len(emo_all_l)):
            for j, tup in enumerate(tuples):
                cur_emo_id, cur_cau_id = tup[0], tup[1]
                if cur_emo_id == emo_all_l[emo_i]: ## 当前是第几个情感
                    cur_emocat = emocats[emo_i]
                    same_emo_cau_l = []
                    if cur_emo_id == cur_cau_id: #当前情感句=原因句
                        same_emo_cau_l.append(cur_emo_id)
                    else: #当前emotion有几个原因
                        same_emo_cau_l.append(cur_cau_id)
                        k = j+1
                        while k < len(tuples):
                            nemo, ncau = tuples[k][0], tuples[k][1]
                            if nemo == cur_emo_id:
                                same_emo_cau_l.append(ncau)
                            k += 1
                    if same_emo_cau_l == cau_all_l or set(same_emo_cau_l).issubset(set(cau_all_l)):
                        if len(same_emo_cau_l) == 1:
                            if cur_emo_id == same_emo_cau_l[0]:
                                sents[i][cur_emo_id - 1] = f"[{cur_emocat},{sents[i][cur_emo_id - 1]},{'情感句'},{'原因句'}]"
                            else:
                                sents[i][cur_emo_id - 1] = \
                                    f"[{cur_emocat},{sents[i][cur_emo_id - 1]},{'情感句'},{sents[i][same_emo_cau_l[0] - 1]},{'原因句'}]"
                        else:
                            all_cause = ''
                            for same_e_cau_all_id in same_emo_cau_l:
                                all_cause += f",{sents[i][same_e_cau_all_id - 1]},{'原因句'}"
                            sents[i][cur_emo_id - 1] = f"[{cur_emocat},{sents[i][emo_id - 1]},{'情感句'}{all_cause}]"

                        break
                    emo_i += 1
        annotated_targets.append(sents[i])

    return annotated_targets


def get_sina_annotated_ee_targets(sents, labels):
    annotated_targets = []
    num_sents = len(sents)
    for i in range(num_sents):
        tuples = labels[i]
        # tup: ([7, 9])
        for tup in tuples:
            emo_id = tup
            sents[i][emo_id - 1] = f"[{sents[i][emo_id - 1]},{'情感句'}]"
            # print(sents[i])
        annotated_targets.append(sents[i])

    return annotated_targets


def get_sina_annotated_ce_targets(sents, labels):
    annotated_targets = []
    num_sents = len(sents)
    for i in range(num_sents):
        tuples = labels[i]
        # tup: ([7, 9])
        for tup in tuples:
            cau_id = tup
            sents[i][cau_id - 1] = f"[{sents[i][cau_id - 1]},{'原因句'}]"
            # print(sents[i])
        annotated_targets.append(sents[i])

    return annotated_targets


def get_sina_annotated_alle_targets(sents, labels):
    annotated_targets = []
    num_sents = len(sents)
    for i in range(num_sents):
        tuples = labels[i]
        # tup: ([7, 9])
        for tup in tuples:
            cau_id = tup
            sents[i][cau_id - 1] = f"[{sents[i][cau_id - 1]},{'原因句'}]"
            # print(sents[i])
        annotated_targets.append(sents[i])

    return annotated_targets


def get_sina_annotated_ece_targets(sents, emocat_labels, cau_labels):
    annotated_targets = []
    num_sents = len(sents)
    del_list = []
    for i in range(num_sents):
        emo_cats = emocat_labels[i]
        # tup: ([7, 9])
        if len(emo_cats) == 1:
            tuples = cau_labels[i]
            for tup in tuples:
                cau_id = tup
                sents[i][cau_id - 1] = f"[{emo_cats[0]},{sents[i][cau_id - 1]}]"
        else:
            del_list.append(i)
        annotated_targets.append(sents[i])
    for del_id in reversed(del_list):
        annotated_targets.pop(del_id)
    return annotated_targets, del_list


def get_ntcir_annotated_ecpe_targets(sents, labels, doc_emocat):
    annotated_targets = []
    num_sents = len(sents)
    for i in range(num_sents):
        tuples = labels[i]
        emocats = doc_emocat[i]
        # tup: ([7, 9])\(1, 2), (1, 3)
        emo_all_l, cau_all_l = [], []
        for tup in tuples:
            emo_id, cau_id = tup[0], tup[1]
            if emo_id not in emo_all_l:
                emo_all_l.append(emo_id)
            if cau_id not in cau_all_l:
                cau_all_l.append(cau_id)
        for emo_i in range(len(emo_all_l)):
            for j, tup in enumerate(tuples):
                cur_emo_id, cur_cau_id = tup[0], tup[1]
                if cur_emo_id == emo_all_l[emo_i]: ## 当前是第几个情感
                    cur_emocat = emocats[emo_i]
                    same_emo_cau_l = []
                    if cur_emo_id == cur_cau_id: #当前情感句=原因句
                        same_emo_cau_l.append(cur_emo_id)
                    else: #当前emotion有几个原因
                        same_emo_cau_l.append(cur_cau_id)
                        k = j+1
                        while k < len(tuples):
                            nemo, ncau = tuples[k][0], tuples[k][1]
                            if nemo == cur_emo_id:
                                same_emo_cau_l.append(ncau)
                            k += 1
                    if same_emo_cau_l == cau_all_l or set(same_emo_cau_l).issubset(set(cau_all_l)):
                        if len(same_emo_cau_l) == 1:
                            if cur_emo_id == same_emo_cau_l[0]:
                                sents[i][cur_emo_id - 1] = f"[{cur_emocat},{sents[i][cur_emo_id - 1]},{'emotion clause'},{'cause clause'}]"
                            else:
                                sents[i][cur_emo_id - 1] = \
                                    f"[{cur_emocat},{sents[i][cur_emo_id - 1]},{'emotion clause'},{sents[i][same_emo_cau_l[0] - 1]},{'cause clause'}]"
                        else:
                            all_cause = ''
                            for same_e_cau_all_id in same_emo_cau_l:
                                all_cause += f",{sents[i][same_e_cau_all_id - 1]},{'cause clause'}"
                            sents[i][cur_emo_id - 1] = f"[{cur_emocat},{sents[i][emo_id - 1]},{'cause clause'}{all_cause}]"

                        break
                    emo_i += 1
        annotated_targets.append(sents[i])

    return annotated_targets


def get_ntcir_annotated_ee_targets(sents, labels):
    annotated_targets = []
    num_sents = len(sents)
    for i in range(num_sents):
        tuples = labels[i]
        # tup: ([7, 9])
        for tup in tuples:
            emo_id = tup
            sents[i][emo_id - 1] = f"[{sents[i][emo_id - 1]},{'emotion clause'}]"
            # print(sents[i])
        annotated_targets.append(sents[i])

    return annotated_targets


def get_ntcir_annotated_ce_targets(sents, labels):
    annotated_targets = []
    num_sents = len(sents)
    for i in range(num_sents):
        tuples = labels[i]
        # tup: ([7, 9])
        for tup in tuples:
            cau_id = tup
            sents[i][cau_id - 1] = f"[{sents[i][cau_id - 1]},{'cause clause'}]"
            # print(sents[i])
        annotated_targets.append(sents[i])

    return annotated_targets


def get_ntcir_annotated_ece_targets(sents, emocat_labels, cau_labels):
    annotated_targets = []
    num_sents = len(sents)
    del_list = []
    for i in range(num_sents):
        emo_cats = emocat_labels[i]
        # tup: ([7, 9])
        if len(emo_cats) == 1:
            tuples = cau_labels[i]
            for tup in tuples:
                cau_id = tup
                sents[i][cau_id - 1] = f"[{emo_cats[0]},{sents[i][cau_id - 1]}]"
        else:
            del_list.append(i)
        annotated_targets.append(sents[i])
    for del_id in reversed(del_list):
        annotated_targets.pop(del_id)
    return annotated_targets, del_list


def get_sina_extraction_ecpe_targets(sents, labels, doc_emocat):
    targets = []
    for i, label in enumerate(labels):
        if label == []:
            targets.append('None')
        else:
            ##统计emotion
            emo_all_l = []
            for ei in label:
                if ei[0] not in emo_all_l:
                    emo_all_l.append(ei[0])

            label_strs = []
            for e_i, emoi in enumerate(emo_all_l):
                e_clause = sents[i][emoi - 1]  ##当前emo子句
                e_cat = doc_emocat[i][e_i]

                all_c_str = ''
                for tri in label:

                    if tri[0] == emoi: ##是第一个emo
                        all_c_str += ','+sents[i][tri[1]-1]
                label_strs.append('['+e_clause+all_c_str+','+e_cat+']')
            targets.append(';'.join(label_strs))
    return targets


def get_sina_extraction_ee_targets(sents, labels):
    targets = []
    for i, label in enumerate(labels):
        if label == []:
            targets.append('None')
        else:
            all_tri = []
            for tri in label:
                e = sents[i][tri - 1]
                all_tri.append((e))
            label_strs = ['[' + l + ']' for l in all_tri]
            targets.append(';'.join(label_strs))
    return targets


def get_sina_extraction_ce_targets(sents, labels):
    targets = []
    for i, label in enumerate(labels):
        if label == []:
            targets.append('None')
        else:
            all_tri = []
            for tri in label:
                c = sents[i][tri - 1]
                all_tri.append((c))
            label_strs = ['[' + l + ']' for l in all_tri]
            targets.append(';'.join(label_strs))
    return targets


def get_sina_extraction_ece_targets(sents, emocat_labels, cau_labels):
    targets = []
    del_list = []
    for i, label in enumerate(cau_labels):
        emo_cats = emocat_labels[i]
        if label == []:
                targets.append('None')
        else:
            if len(emo_cats) == 1:
                all_tri = []
                for tri in label:
                    c = sents[i][tri - 1]
                    all_tri.append((c))
                label_strs = ['[' + l + ']' for l in all_tri]
            else:
                del_list.append(i)
            targets.append(';'.join(label_strs))
    for del_id in reversed(del_list):
        targets.pop(del_id)
    return targets, del_list


def get_ntcir_extraction_ecpe_targets(sents, labels, doc_emocat):
    targets = []
    for i, label in enumerate(labels):
        if label == []:
            targets.append('None')
        else:
            all_tri = []
            for tri in label:
                e = sents[i][tri[0]-1]
                c = sents[i][tri[1]-1]
                all_tri.append((e, c))
            label_strs = ['('+', '.join(l)+')' for l in all_tri]
            targets.append('; '.join(label_strs))
    return targets


def get_ntcir_extraction_ee_targets(sents, labels):
    targets = []
    for i, label in enumerate(labels):
        if label == []:
            targets.append('None')
        else:
            all_tri = []
            for tri in label:
                e = sents[i][tri - 1]
                all_tri.append((e))
            label_strs = ['(' + l + ')' for l in all_tri]
            targets.append('; '.join(label_strs))
    return targets


def get_ntcir_extraction_ce_targets(sents, labels):
    targets = []
    for i, label in enumerate(labels):
        if label == []:
            targets.append('None')
        else:
            all_tri = []
            for tri in label:
                c = sents[i][tri - 1]
                all_tri.append((c))
            label_strs = ['(' + l + ')' for l in all_tri]
            targets.append('; '.join(label_strs))
    return targets


def get_ntcir_extraction_ece_targets(sents, emocat_labels, cau_labels):
    targets = []
    del_list = []
    for i, label in enumerate(cau_labels):
        emo_cats = emocat_labels[i]
        if label == []:
                targets.append('None')
        else:
            if len(emo_cats) == 1:
                all_tri = []
                for tri in label:
                    c = sents[i][tri - 1]
                    all_tri.append((c))
                label_strs = ['[' + l + ']' for l in all_tri]
            else:
                del_list.append(i)
            targets.append(';'.join(label_strs))
    for del_id in reversed(del_list):
        targets.pop(del_id)
    return targets, del_list


def get_sina_transformed_io(data_path, paradigm, task):
    """
    The main function to transform the Input & Output according to
    the specified paradigm and task
    """
    doc_ids, doc_lens, doc_sents, doc_ecps, doc_emos, doc_caus, doc_emocat = read_sina_line_examples_from_file(data_path)

    # the input is just the raw sentence
    inputs = [s.copy() for s in doc_sents]

    # Get target according to the paradigm
    # annotate the sents (with label info) as targets
    if paradigm == 'annotation':
        if task == 'ecpe':
            targets = get_sina_annotated_ecpe_targets(doc_sents, doc_ecps, doc_emocat)
        elif task == 'ee':
            targets = get_sina_annotated_ee_targets(doc_sents, doc_emos)
        elif task == 'ce':
            targets = get_sina_annotated_ce_targets(doc_sents, doc_caus)
        elif task == 'ece':
            targets, del_list = get_sina_annotated_ece_targets(doc_sents, doc_emocat, doc_caus)
            for del_id in reversed(del_list):
                inputs.pop(del_id)
        else:
            raise NotImplementedError
    # directly treat label infor as the target
    elif paradigm == 'extraction':
        if task == 'ecpe':
            targets = get_sina_extraction_ecpe_targets(doc_sents, doc_ecps, doc_emocat)
        elif task == 'ee':
            targets = get_sina_extraction_ee_targets(doc_sents, doc_emos)
        elif task == 'ce':
            targets = get_sina_extraction_ce_targets(doc_sents, doc_caus)
        elif task == 'ece':
            targets, del_list = get_sina_extraction_ece_targets(doc_sents, doc_emocat, doc_caus)
            for del_id in reversed(del_list):
                inputs.pop(del_id)
        else:
            raise NotImplementedError
    else:
        print('Unsupported paradigm!')
        raise NotImplementedError
    return inputs, targets


def get_ntcir_transformed_io(data_path, paradigm, task):
    """
    The main function to transform the Input & Output according to
    the specified paradigm and task
    """
    doc_ids, doc_lens, doc_sents, doc_ecps, doc_emos, doc_caus, doc_emocat = read_ntcir_line_examples_from_file(data_path)

    # the input is just the raw sentence
    inputs = [s.copy() for s in doc_sents]

    # Get target according to the paradigm
    # annotate the sents (with label info) as targets
    if paradigm == 'annotation':
        if task == 'ecpe':
            targets = get_ntcir_annotated_ecpe_targets(doc_sents, doc_ecps, doc_emocat)
        elif task == 'ee':
            targets = get_ntcir_annotated_ee_targets(doc_sents, doc_emos)
        elif task == 'ce':
            targets = get_ntcir_annotated_ce_targets(doc_sents, doc_caus)
        elif task == 'ece':
            targets, del_list = get_ntcir_annotated_ece_targets(doc_sents, doc_emocat, doc_caus)
            for del_id in reversed(del_list):
                inputs.pop(del_id)
        else:
            raise NotImplementedError
    # directly treat label infor as the target
    elif paradigm == 'extraction':
        if task == 'ecpe':
            targets = get_ntcir_extraction_ecpe_targets(doc_sents, doc_ecps, doc_emocat)
        elif task == 'ee':
            targets = get_ntcir_extraction_ee_targets(doc_sents, doc_emos)
        elif task == 'ce':
            targets = get_ntcir_extraction_ce_targets(doc_sents, doc_caus)
        elif task == 'ece':
            targets, del_list = get_ntcir_extraction_ece_targets(doc_sents, doc_emocat, doc_caus)
            for del_id in reversed(del_list):
                inputs.pop(del_id)
        else:
            raise NotImplementedError
    else:
        print('Unsupported paradigm!')
        raise NotImplementedError
    return inputs, targets


def write_results_to_log(log_file_path, best_test_result, args, dev_results, test_results, global_steps):
    """
    Record dev and test results to log file
    """
    local_time = time.asctime(time.localtime(time.time()))
    exp_settings = "Exp setting: {0} on {1} under {2} | {3:.4f} | ".format(
        args.task, args.dataset, args.paradigm, best_test_result
    )
    train_settings = "Train setting: bs={0}, lr={1}, num_epochs={2}".format(
        args.train_batch_size, args.learning_rate, args.num_train_epochs
    )
    results_str = "\n* Results *:  Dev  /  Test  \n"

    metric_names = ['f1', 'precision', 'recall']
    for gstep in global_steps:
        results_str += f"Step-{gstep}:\n"
        for name in metric_names:
            name_step = f'{name}_{gstep}'
            results_str += f"{name:<8}: {dev_results[name_step]:.4f} / {test_results[name_step]:.4f}"
            results_str += ' '*5
        results_str += '\n'

    log_str = f"{local_time}\n{exp_settings}\n{train_settings}\n{results_str}\n\n"

    with open(log_file_path, "a+") as f:
        f.write(log_str)


class SINADataset(Dataset):
    def __init__(self, tokenizer, fold_id, data_type, task, paradigm, data_dir, max_len=512):
        self.data_type = data_type
        self.task = task
        self.paradigm = paradigm
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.inputs = []
        self.targets = []
        if self.data_type == 'train':
            self.data_path = join('data/' + self.task + '/' + data_dir, TRAIN_FILE % fold_id)
        elif self.data_type == 'test':
            self.data_path = join('data/' + self.task + '/' + data_dir, TEST_FILE % fold_id)
        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        source_ids = self.inputs[item]["input_ids"].squeeze()
        target_ids = self.targets[item]["input_ids"].squeeze()

        src_mask = self.inputs[item]["attention_mask"].squeeze()      # might need to squeeze
        target_mask = self.targets[item]["attention_mask"].squeeze()  # might need to squeeze
        return {"source_ids": source_ids, "source_mask": src_mask,
                "target_ids": target_ids, "target_mask": target_mask}

    def _build_examples(self):

        inputs, targets = get_sina_transformed_io(self.data_path, self.paradigm, self.task)
        print('samples numbers:', len(inputs))

        for i in range(len(inputs)):
            input = '，'.join(inputs[i])
            if self.paradigm == 'annotation':
                target = '，'.join(targets[i])
            else:
                target = targets[i]
            tokenized_input = self.tokenizer.batch_encode_plus(
              [input], max_length=self.max_len, pad_to_max_length=True, truncation=True,
              return_tensors="pt"
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
              [target], max_length=self.max_len, pad_to_max_length=True, truncation=True,
              return_tensors="pt"
            )
            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)


class NTCIRDataset(Dataset):
    def __init__(self, tokenizer, fold_id, data_type, task, paradigm, data_dir, max_len=512):
        self.data_type = data_type
        self.task = task
        self.paradigm = paradigm
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.inputs = []
        self.targets = []
        if self.data_type == 'train':
            self.data_path = join('data/' + self.task + '/' + data_dir, TRAIN_FILE % fold_id)
        elif self.data_type == 'val':
            self.data_path = join('data/' + self.task + '/' + data_dir, VALID_FILE % fold_id)
        elif self.data_type == 'test':
            self.data_path = join('data/' + self.task + '/' + data_dir, TEST_FILE % fold_id)
        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        source_ids = self.inputs[item]["input_ids"].squeeze()
        target_ids = self.targets[item]["input_ids"].squeeze()

        src_mask = self.inputs[item]["attention_mask"].squeeze()      # might need to squeeze
        target_mask = self.targets[item]["attention_mask"].squeeze()  # might need to squeeze
        return {"source_ids": source_ids, "source_mask": src_mask,
                "target_ids": target_ids, "target_mask": target_mask}

    def _build_examples(self):

        inputs, targets = get_ntcir_transformed_io(self.data_path, self.paradigm, self.task)
        print('samples numbers:', len(inputs))

        for i in range(len(inputs)):
            input = '，'.join(inputs[i])
            if self.paradigm == 'annotation':
                # if self.task != 'tasd':
                #     target = ' '.join(targets[i])
                # else:
                #     target = targets[i]
                target = ','.join(targets[i])
            else:
                target = targets[i]

            tokenized_input = self.tokenizer.batch_encode_plus(
              [input], max_length=self.max_len, pad_to_max_length=True, truncation=True,
              return_tensors="pt",
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
              [target], max_length=self.max_len, pad_to_max_length=True, truncation=True,
              return_tensors="pt"
            )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)
