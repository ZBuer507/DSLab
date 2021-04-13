# -*- coding: utf-8 -*-
import threading
import os
import json
import requests
from bs4 import BeautifulSoup
import urllib.request

# 爬取的页数
N = 100
# 用来存储爬到的网页转换的json
global json_craw
json_craw = set()

def get_urls():
    # 爬取今日哈工大公告公示页面列表中的url
    urls = ["http://today.hit.edu.cn/category/10?page={}".format(str(i)) for i in range(0, N)]
    url_set = set()
    page = 10
    for url in urls:
        web_data = requests.get(url).text
        # 通过分析网页源码可以获知url列表的位置和结构
        # 使用BeautifulSoup进行解析和提取
        soup = BeautifulSoup(web_data, 'html.parser')
        data_list = soup.select("span > span > a")
        for item in data_list:
            link = item.get("href")
            link = "http://today.hit.edu.cn" + link
            url_set.add(link)
        page = page + 1
    return url_set

def craw(urls):
    # 根据爬取到的url列表爬取内容和附件
    url_list = urls
    with open('data/data_craw.json', 'w', encoding='utf-8') as ff:
        for url in url_list:
            data = urllib.request.urlopen(url).read()
            # 解码，忽略异常编码
            data2 = data.decode("utf-8")
            soup = BeautifulSoup(data2, 'html.parser')
            title = soup.select('h3')
            content = soup.select('p')
            # 没有标题，跳过
            if len(title) == 0:
                continue
            text = ""
            for i in range(0, len(content)):
                con = content[i].get_text().strip()
                if (len(con) != 0):
                    text += con
            # 文章过短，跳过
            if len(text) < 100:
                continue
            # 只需要office文档
            file_urls = soup.find_all('span', {"class": "file--x-office-document"})
            names = []
            if len(file_urls):
                for item in file_urls:
                    file = item.select('a')
                    file_name = file[0].get_text().strip()
                    names.append(file_name)
                    fileurl = file[0].get('href')
                    req = requests.get(fileurl, stream = True)
                    # 在相应目录下写入附件
                    with open('data/files/%s' % file_name, 'wb') as f:
                        for chunk in req.iter_content(chunk_size=128):
                            f.write(chunk)
            # 存储
            data = {
                "url": url,
                "title": title[0].get_text().strip(),
                "parapraghs": text,
                "file_name": names}
            json_str = json.dumps(data, ensure_ascii=False)
            # 由于多线程单独文件的读写比较麻烦，需要lock
            # 所以存储在json_craw中，之后统一读写
            json_craw.add(json_str + '\n')

# 从threading.Thread继承创建一个新的子类craw_thread
# 实例化后调用start()方法启动新线程
class craw_thread(threading.Thread):
    def __init__(self, url_list, name):
        threading.Thread.__init__(self)
        self.url_list = url_list
        self.name = name

    def run(self):
        print ("开始线程：", self.name)
        craw(self.url_list)
        print ("退出线程：", self.name)

def main():
    # 检查路径是否存在
    # 不存在则创建相应目录
    if not os.path.exists("data/files"):
        os.mkdir("data/files")
    urls = get_urls()
    urls = list(urls)
    # print(urls)
    length = len(urls)
    # 4个线程
    n = int(length/4)
    thread1 = craw_thread(urls[0: n], 1)
    thread2 = craw_thread(urls[n: n*2], 2)
    thread3 = craw_thread(urls[n*2: n*3], 3)
    thread4 = craw_thread(urls[n*3:], 4)
    # 线程的启动与结束
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()
    # 将爬取到的json写入文件
    with open('data/data_craw.json', 'w', encoding='utf-8') as  f:
        for item in json_craw:
            f.write(item)

if __name__ == '__main__':
    main()
    