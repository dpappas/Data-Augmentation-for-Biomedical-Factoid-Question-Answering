
# follow instructions of https://github.com/biswa380/t5_multitask_api#usage
from nltk.tokenize import sent_tokenize
from pipelines import pipeline
from elasticsearch import Elasticsearch
import re, xlwt, sys
from pprint import pprint

nlp = pipeline("question-generation")

with open('/home/dpappas/elk_ips.txt') as fp:
    cluster_ips = [line.strip() for line in fp.readlines() if(len(line.strip())>0)]
    fp.close()

es = Elasticsearch(cluster_ips, verify_certs=True, timeout=150, max_retries=10, retry_on_timeout=True)
doc_index       = 'pubmed_abstracts_joint_0_1'

exclude_phrases = ['copyright', 'et al.', 'all rights reserved', '?', '__']

all_questions   = []
bod             = {
    'size':10000,
    "query": {
        "range": {
            "DateCompleted": {
                "gte"   : "01-2019", # sys.argv[2], # "01-2019",
                "lt"    : "02-2019", # sys.argv[3], #"02-2019",
                "format": "MM-yyyy"
            }
        }
    }
}
for item in es.search(index=doc_index, body=bod)['hits']['hits']:
    joint_text  = item['_source']['joint_text']
    title       = joint_text.split('--------------------')[0].strip()
    abstract    = joint_text.split('--------------------')[1].strip()
    # snippets    = sent_tokenize(title) + sent_tokenize(abstract)
    snippets    = sent_tokenize(abstract)
    for snippet in snippets:
        snippet_proc = re.sub("\(.+?\)","",snippet)
        pprint(
            [
                snippet_proc,
                nlp(snippet_proc)
            ]
        )
        if(any(phr in snippet_proc for phr in exclude_phrases)):
            continue
        if(len(snippet_proc.split())<6):
            continue
        if(len(snippet_proc.split())>30):
            continue
        try:
            for pair in nlp(snippet_proc):
                all_questions.append(
                    (
                        pair['question'],
                        snippet,
                        pair['answer'],
                        item['_id']
                    )
                )
        except:
            pass
    print(len(all_questions))
    if(len(all_questions) >= int(sys.argv[4])):
        break

book    = xlwt.Workbook()
sheet1  = book.add_sheet("questions")

for row_num in range(len(all_questions)):
    row = sheet1.row(row_num)
    for col_num in range(len(all_questions[row_num])):
        row.write(col_num, all_questions[row_num][col_num])

book.save(sys.argv[1])

