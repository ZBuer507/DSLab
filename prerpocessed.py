import math
import numpy as np
from multiprocessing import Pool, cpu_count
import jieba
import json
import time
import pickle


add_punc = ['[', ']', ':', '【', ' 】', '（', '）', '‘', '’', '{', '}', '⑦', '(', ')', '%', '^', '<', '>', '℃', '.', '-',
            '——', '—', '=', '&', '#', '@', '￥', '$']  # 定义要删除的特殊字符
stopwords = [line.strip() for line in open('stopwords(new).txt', encoding='UTF-8').readlines()]
stopwords = stopwords + add_punc


class BM25:
    def __init__(self, corpus, documents, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = len(corpus)
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer

        if tokenizer:
            corpus = self._tokenize_corpus(corpus)

        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus):
        nd = {}
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                try:
                    nd[word]+=1
                except KeyError:
                    nd[word] = 1

        self.avgdl = num_doc / self.corpus_size
        return nd

    def _tokenize_corpus(self, corpus):
        pool = Pool(cpu_count())
        tokenized_corpus = pool.map(self.tokenizer, corpus)
        return tokenized_corpus

    def _calc_idf(self, nd):
        idf_sum = 0
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score

    def get_batch_scores(self, query, doc_ids):
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score.tolist()

    def get_top_n(self, query, n=5):
        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [self.documents[i] for i in top_n]
    
    def get_top_index(self, query):
        scores = self.get_scores(query)
        top_index = np.argsort(scores)[::-1][0]
        return top_index
    
    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

def dealwords(sent):
    words = list(jieba.cut(sent))
    words = list(filter(lambda x: x not in stopwords, words))
    return words

def segment():
    corpus = set()
    # 读取未分词文件
    with open('lab2-data/prerprocessed/passages_multi_sentences.json', encoding='utf-8') as fin:
        read_results = [json.loads(line.strip()) for line in fin.readlines()]
    for result in read_results:
        #print(result['document'])
        result['document'] = [' '.join(dealwords(sent)) for sent in result['document']]
        for item in result['document']:
            temp = item.split(" ")
            for i in temp:
                # print(i)
                if i not in stopwords:
                    corpus.add(i)
    print("分词结束，开始写入文件...")
    # print(corpus[0])
    # 写回分词后的文件
    with open('lab2-data/prerprocessed/corpus.txt', 'w', encoding='utf-8') as fout:
        for item in corpus:
            fout.write(item + '\n')


def build_BM25Model():
    docs = []  # 所有文档列表,词表示
    documents = []
    # 读取文件
    with open('lab2-data/prerprocessed/passages_multi_sentences.json', encoding='utf-8') as fin:
        read_results = [json.loads(line.strip()) for line in fin.readlines()]

    for result in read_results:
        words_in_document = []
        document = ""
        for sent in result['document']:
            document += sent
            for i in (dealwords(sent)):  # 去停用词
                words_in_document.append(i)
        documents.append(document)
        docs.append(words_in_document)
    print("建立BM25模型...")
    print(len(docs))
    #print(len(documents))
    bm25Model = BM25(docs, documents)
    bm25Model.save_model('lab2-data/prerprocessed/bm25.pkl')


def search():
    with open('lab2-data/prerprocessed/bm25.pkl', "rb") as f:
        bm25 = pickle.load(f)
    query = "盐酸丁二胍什么时候被当做降糖药？"
    print(dealwords(query))
    time1 = time.time()
    line = bm25.get_top_n(query, 1)
    print("查询完成, 用时 {}s".format(time.time() - time1))
    print(line[0])


def train_test():
    with open('lab2-data/prerprocessed/bm25.pkl', "rb") as f:
        bm25 = pickle.load(f)
    with open('lab2-data/prerprocessed/train.json', 'r', encoding='utf-8') as fin:
        items = [json.loads(line.strip()) for line in fin.readlines()]
    pid_label = []
    pid_pre = []
    i = 0
    time1 = time.time()
    for item in items:
        pid_label.append(item['pid'])
        pid_pre.append(bm25.get_top_index(dealwords(item['question'])))
        i += 1
        if i % 100 == 0:
            print("查询 {} 完成, 用时 {}s".format(i, time.time() - time1))
            time1 = time.time()
    eval(pid_label, pid_pre)


def eval(label, pre):
    rr = 0
    rr_rn = len(label)
    for i in range(len(label)):
        if label[i] == pre[i]:
            rr += 1
        else:
            print(label[i], ":", pre[i])
    p = float(rr) / rr_rn

    print("总计:{}, 检索回来的相关文档数:{}, 检索回来的文档总数:{}, Precision:{}".format(len(label), rr, rr_rn, p))


if __name__ == '__main__':
    start = time.time()
    #segment()
    #build_BM25Model()
    #train_test()
    search()
    end = time.time()
    #print("查询用时： ", end - start)
