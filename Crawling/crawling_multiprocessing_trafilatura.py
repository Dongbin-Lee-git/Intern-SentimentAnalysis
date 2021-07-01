import multiprocessing
import trafilatura
import pandas as pd
import os

#동작 실행 함수
def crawl(url):
    print(url)
    downloaded = trafilatura.fetch_url(url)
    body = trafilatura.extract(downloaded)
    meta = trafilatura.metadata.extract_metadata(downloaded)
    if body and meta is not None:
        db = pd.DataFrame({"title": [meta['title']], "content": [body]})
        if not os.path.exists('result_crawling.csv'):
            db.to_csv('result_crawling.csv', mode='w', encoding='utf-8')
        else:
            db.to_csv('result_crawling.csv', mode='a', header=False, encoding='utf-8')


#Queue 전달 함수
def worker(q):
    for item in iter(q.get, None):
        crawl(item)
        q.task_done()
    q.task_done()

def main():
    # cpu 갯수 확인 : 최대 multi 가능한 process의 수
    print(multiprocessing.cpu_count())

    xlfile = pd.read_excel("f1soft_google_취합본.xlsx")
    print("read excel")

    # 동작 프로세스 개수
    num_procs = 5

    # 큐 데이터
    items = xlfile['url']

    q = multiprocessing.JoinableQueue()

    procs = []

    for i in range(num_procs):
        procs.append(multiprocessing.Process(target=worker, args=(q,)))
        procs[-1].daemon = True
        procs[-1].start()

    for item in items:
        q.put(item)
    q.join()

    for p in procs:
        q.put(None)
    q.join()

    for p in procs:
        p.join()

    # q.__init__(ctx=None)
    print("Finished everything....")
    print("num active children:", multiprocessing.active_children())

if __name__ == '__main__':
    main()


# workbook.close()



