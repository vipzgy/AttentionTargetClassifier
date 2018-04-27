# -*- encoding: utf-8 -*-
import re
import codecs
import pickle
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = string.lower()
    string = re.sub(r"，", ",", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)

    string = re.sub(r"^-user-$", "<user>", string)
    string = re.sub(r"^-url-$", "<url>", string)
    string = re.sub(r"^-lqt-$", "\'", string)
    string = re.sub(r"^-rqt-$", "\'", string)
    # 或者
    # string = re.sub(r"^-lqt-$", "\"", string)
    # string = re.sub(r"^-rqt-$", "\"", string)
    string = re.sub(r"^-lrb-$", "\(", string)
    string = re.sub(r"^-rrb-$", "\)", string)
    string = re.sub(r"^lol$", "<lol>", string)
    string = re.sub(r"^<3$", "<heart>", string)
    string = re.sub(r"^#.*", "<hashtag>", string)
    string = re.sub(r"^[0-9]*$", "<number>", string)
    string = re.sub(r"^\:\)$", "<smile>", string)
    string = re.sub(r"^\;\)$", "<smile>", string)
    string = re.sub(r"^\:\-\)$", "<smile>", string)
    string = re.sub(r"^\;\-\)$", "<smile>", string)
    string = re.sub(r"^\;\'\)$", "<smile>", string)
    string = re.sub(r"^\(\:$", "<smile>", string)
    string = re.sub(r"^\)\:$", "<sadface>", string)
    string = re.sub(r"^\)\;$", "<sadface>", string)
    string = re.sub(r"^\:\($", "<sadface>", string)
    return string.strip()


def read_sentence_line(path):
    """
    读取一行是一句话的语料,比如情感分类
    Args:
        path: str, 语料文件路径
    Return:
        data: list中tuple
    """
    data = []
    with open(path, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            line = line.strip()
            if len(line) == 0 or line == '':
                print("an empty sentence, please check")
            else:
                data.append(line)
    return data


def read_word_line(path, is_train=False):
    """
    读取一个单词是一行的语料,比如NER
    如果读取的是train集，需要建立好词频的表，标签
    Args:
        path: str, 语料文件路径
        is_train: boolean, 判断是否是train集
    Return:
        data: list
        sentence_len: dict
        feature_dict: dict
        label_dict: dict
    """
    data = []
    sentence_len = Counter()
    feature_dict = Counter()
    label_dict = Counter()
    with open(path, 'r', encoding='utf-8') as input_file:
        s = []
        start = -1
        end = -1
        label = None
        count = 0
        target_flag = False
        for line in input_file:
            line = line.strip()
            if len(line) == 0 or line == '':
                sentence_len[len(s)] += 1
                if target_flag:
                    end = count - 2
                data.append((s, start, end, label))
                s = []
                count = 0
                if is_train:
                    label_dict[label] += 1
            else:
                strings = line.split(' ')
                word = clean_str(strings[0])
                words = word.split(' ')
                for w in words:
                    s.append(w)
                    count += 1
                    if is_train:
                        feature_dict[w] += 1
                if strings[-1][0] == 'o':
                    if target_flag:
                        end = count - 2
                        target_flag = False
                elif strings[-1][0] == 'b':
                    start = count - 1
                    label = strings[1][2:]
                    target_flag = True
        if len(s) != 0:
            if is_train:
                label_dict[label] += 1
            data.append((s, start, end, label))
            sentence_len[len(s)] += 1
    print('实例个数有: ', len(data))
    if is_train:
        return data, sentence_len, feature_dict, label_dict
    return data, sentence_len


# def read_word_line(path, is_train=False):
#     """
#     读取一个单词是一行的语料,比如NER
#     如果读取的是train集，需要建立好词频的表，标签
#     Args:
#         path: str, 语料文件路径
#         is_train: boolean, 判断是否是train集
#     Return:
#         data: list
#         sentence_len: dict
#         feature_dict: dict
#         label_dict: dict
#     """
#     data = []
#     sentence_len = Counter()
#     feature_dict = Counter()
#     label_dict = Counter()
#     with open(path, 'r', encoding='utf-8') as input_file:
#         s = []
#         sl = []
#         sr = []
#         t = []
#         label = None
#         target_flag = False
#         for line in input_file:
#             line = line.strip()
#             if len(line) == 0 or line == '':
#                 target_flag = False
#                 if is_train:
#                     label_dict[label] += 1
#                 data.append((s, sl, sr, t, label))
#                 sentence_len[len(s) + len(t)] += 1
#                 s = []
#                 sl = []
#                 sr = []
#                 t = []
#                 label = None
#             else:
#                 strings = line.split(' ')
#                 word = clean_str(strings[0])
#                 words = word.split(' ')
#                 if strings[-1] == 'o':
#                     for w in words:
#                         s.append(w)
#                         if target_flag is False:
#                             sl.append(w)
#                         else:
#                             sr.append(w)
#                         if is_train:
#                             feature_dict[w] += 1
#                 else:
#                     for w in words:
#                         t.append(w)
#                         if is_train:
#                             feature_dict[w] += 1
#                     label = strings[-1].split('-')[-1]
#                     target_flag = True
#
#         if len(s) != 0:
#             if is_train:
#                 label_dict[label] += 1
#             data.append((s, sl, sr, t, label))
#             sentence_len[len(s) + len(t)] += 1
#     print('实例个数有: ', len(data))
#     if is_train:
#         return data, sentence_len, feature_dict, label_dict
#     return data, sentence_len


def read_csv(path, split=','):
    """
    读取csv文件

    Args:
        path: str, csv文件路径
        split: 分隔符号

    Return:
        terms: list
    """
    with open(path, 'r', encoding='utf-8') as file_csv:
        line = file_csv.readline()
        terms = []
        while line:
            line = line.strip()
            if not line:
                line = file_csv.readline()
                continue
            terms.append(line.split(split))
            line = file_csv.readline()
    return terms


def read_pkl(path):
    """
    读取pkl文件

    Args:
        path: str, pkl文件路径

    Return:
        pkl_ob: pkl对象
    """
    file_pkl = codecs.open(path, 'rb')
    return pickle.load(file_pkl)

