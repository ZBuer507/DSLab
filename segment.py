# -*- coding: utf-8 -*-
import threading
import os
import jieba
import json

# 试错后总结的一些应当去掉的
list = [" ", "", "\n", ".", "/", "（", "）", "@", "-", "【", "】", "’", "‘", "©", ";", "—", ":"]
#补充后的停用词表
global stopwords
stopwords = [line.strip() for line in open('stopwords(new).txt', encoding='UTF-8').readlines()]
stopwords += list
# 用来存储分词后的的json
global jsons_segmented
jsons_segmented = set()

# python3.9 pyltp安装失败
# 使用jieba分词
def segment(json_list):
    for item in json_list:
        titlelist = []
        for word in jieba.cut(item['title'].replace(' ', "."), HMM=True):
            if word not in stopwords:
                titlelist.append(word)
        textlist = []
        for word2 in jieba.cut(item['parapraghs'].replace(' ', "."), HMM=True):
            if word2 not in stopwords:
                textlist.append(word2)
        data = {
            "url": item['url'],
            "segmented_title": titlelist,
            "segmented_parapraghs": textlist,
            "file_name": item['file_name']}
        # print(data)
        json_segmented = json.dumps(data, ensure_ascii=False)
        jsons_segmented.add(json_segmented)

# 从threading.Thread继承创建一个新的子类segment_thread
# 实例化后调用start()方法启动新线程
class segment_thread(threading.Thread):
    def __init__(self, json_list, name):
        threading.Thread.__init__(self)
        self.json_list = json_list
        self.name = name

    def run(self):
        print ("开始线程：", self.name)
        segment(self.json_list)
        print ("退出线程：", self.name)

def main():
    if not os.path.exists('data/data_craw.json'):
        assert False
    with open('data/data_craw.json', 'r', encoding='utf-8') as f:
        json_list = [json.loads(line) for line in  f]
    length = len(json_list)
    # 4个线程
    n = int(length/4)
    thread1 = segment_thread(json_list[0: n], 1)
    thread2 = segment_thread(json_list[n: n*2], 2)
    thread3 = segment_thread(json_list[n*2: n*3], 3)
    thread4 = segment_thread(json_list[n*3:], 4)
    # 线程的启动与结束
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()
    # 将分词后的json写入文件
    with open('data/preprocessed.json', 'w', encoding='utf-8') as f:
        for item in jsons_segmented:
            f.write(item + '\n')

if __name__ == '__main__':
    main()
