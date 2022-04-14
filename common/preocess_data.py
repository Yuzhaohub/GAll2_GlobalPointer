# -*- coding: utf-8 -*-
# @Time    : 2022/4/13 16:10
# @Author  : Fisher
# @File    : preocess_data.py
# @Software: PyCharm
# @Desc    : 数据格式重编码

import os
import json
from tqdm import tqdm

label_list = ["X", 'O', 'B-1', 'I-1', 'B-2', 'I-2', 'B-3', 'I-3', 'B-4', 'I-4', 'B-5', 'I-5', 'B-6', 'I-6',
              'B-7', 'I-7', 'B-8', 'I-8', 'B-9', 'I-9', 'B-10', 'I-10', 'B-11', 'I-11', 'B-12', 'I-12', 'B-13',
              'I-13', 'B-14', 'I-14', 'B-15', 'I-15', 'B-16', 'I-16', 'B-17', 'I-17', 'B-18', 'I-18', 'B-19', 'I-19',
              'B-20', 'I-20', 'B-21', 'I-21', 'B-22', 'I-22', 'B-23', 'I-23', 'B-24', 'I-24', 'B-25', 'I-25', 'B-26',
              'I-26', 'B-28', 'I-28', 'B-29', 'I-29', 'B-30', 'I-30', 'B-31', 'I-31', 'B-32', 'I-32', 'B-33', 'I-33',
              'B-34', 'I-34', 'B-35', 'I-35', 'B-36', 'I-36', 'B-37', 'I-37', 'B-38', 'I-38', 'B-39', 'I-39', 'B-40',
              'I-40', 'B-41', 'I-41', 'B-42', 'I-42', 'B-43', 'I-43', 'B-44', 'I-44', 'B-46', 'I-46', 'B-47', 'I-47',
              'B-48', 'I-48', 'B-49', 'I-49', 'B-50', 'I-50', 'B-51', 'I-51', 'B-52', 'I-52', 'B-53', 'I-53', 'B-54',
              'I-54', "[START]", "[END]"]

ent2id = {
    str(i + 1): v
    for i, v in enumerate(range(54))
}

with open(r'C:\Users\Lenovo\Desktop\GlobalPointer_pytorch\datasets\train_data\ent2id.json', 'w',
          encoding = 'utf-8') as f:
    ent2id_json = json.dumps(ent2id, ensure_ascii = False)
    f.write(ent2id_json)


def process_data(data_path, data_type = "train"):
    """ GlobalPointer数据格式转换 """
    examples = []
    if data_type != 'test':
        words = []
        labels = {}
        index = 0
        start_prefix, end_prefix = None, None
        start, end = -1, -1

        with open(data_path, 'r', encoding = 'utf-8') as f:
            for line in tqdm(f):
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":  # 换行符作为每条样本的拆分
                    if words:
                        examples.append({'text': ''.join(words), 'label': labels})
                        words = []
                        labels = {}

                    start_prefix, end_prefix = None, None
                    start, end = -1, -1
                    index = 0
                else:
                    splits = line.split("\t")
                    word = splits[0].replace("\n", "")
                    words.append(word if len(word) > 0 else ' ')
                    if len(splits) > 1:
                        label = splits[-1].replace("\n", "")
                        prefix = label.split('-')[-1] if '-' in label else 'O'

                        if 'B-' in label:
                            if end_prefix is not None and end - start > 0:
                                if start_prefix not in labels:
                                    labels[start_prefix] = {}
                                lab_text = ''.join(words[start: end + 1])
                                if lab_text not in labels[start_prefix]:
                                    labels[start_prefix][lab_text] = []
                                labels[start_prefix][lab_text].append([start, end])
                                start, end = index, index

                            start = index
                            start_prefix = prefix

                        elif 'I-' in label and start_prefix is not None:
                            if prefix == start_prefix:
                                end = index
                                end_prefix = prefix
                        else:
                            if end - start > 0:
                                if start_prefix not in labels:
                                    labels[start_prefix] = {}
                                lab_text = ''.join(words[start: end + 1])
                                if lab_text not in labels[start_prefix]:
                                    labels[start_prefix][lab_text] = []
                                labels[start_prefix][lab_text].append([start, end])
                                start, end = index, index
                        index += 1
    else:
        with open(data_path, 'r', encoding = 'utf-8') as f:
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":  # 换行符作为每条样本的拆分
                    if words:
                        text = ''.join(words)
                        examples.append({'text': text})
                        words = []
                else:
                    splits = line.split("\t")
                    words.append(splits[0].replace("\n", ""))

    return examples


def save_data(examples, data_path, data_type = 'train'):
    """ 保存数据为json格式 """
    with open(os.path.join(data_path, data_type + '.json'), 'w', encoding = 'utf-8') as f:
        for example in tqdm(examples):
            example_json = json.dumps(example, ensure_ascii = False)
            f.write(example_json)
            f.write('\n')
        print('{}：数据保存完成：{}'.format(data_path, len(examples)))


if __name__ == '__main__':
    data_path = r'C:\Users\Lenovo\Desktop\GlobalPointer_pytorch\datasets\train_data\train.txt'
    examples = process_data(data_path)
    save_data(examples, r'C:\Users\Lenovo\Desktop\GlobalPointer_pytorch\datasets\train_data', data_type = 'train')

    data_path = r'C:\Users\Lenovo\Desktop\GlobalPointer_pytorch\datasets\train_data\dev.txt'
    examples = process_data(data_path)
    save_data(examples, r'C:\Users\Lenovo\Desktop\GlobalPointer_pytorch\datasets\train_data', data_type = 'dev')
    # line = {"text": "办公大励志鼠标垫加厚锁边定制游戏家用可爱创意写字桌垫 努力奔跑 1000x500mm 5mm",
    #         "label": {"5": {"办公": [[0, 1]], "游戏": [[14, 15]], "写字": [[22, 23]]}, "11": {"励志": [[3, 4]]},
    #                   "4": {"鼠标垫": [[5, 7]], "桌垫": [[24, 25]]}, "13": {"加厚": [[8, 9]], "锁边": [[10, 11]]},
    #                   "29": {"定制": [[12, 13]]}, "7": {"家用": [[16, 17]]}, "14": {"可爱": [[18, 19]], "创意": [[20, 21]]},
    #                   "10": {"努力奔跑": [[27, 30]]}, "18": {"1000x500mm": [[32, 41]]}}}
    # text = line['text']
    # label = line['label']
    # print(text[32: 41 + 1])