
# follow instructions of https://github.com/biswa380/t5_multitask_api#usage
from nltk.tokenize import sent_tokenize
from pipelines import pipeline
import re, xlwt, sys, pickle
from tqdm import tqdm
from pprint import pprint
from nltk.corpus import stopwords

nlp         = pipeline("question-generation")

sws         = stopwords.words('english')
train_path  = '/home/dpappas/bioasq_factoid/pubmed_factoid_extracted_data.p'
keep_only   = set([t.strip() for t in 'factoid_snippet,factoid_before_after_1,factoid_before_after_2,factoid_ideal_answer,factoid_joint snippets,factoid_whole abstract'.split(',')])

train_data    =  pickle.load(open(train_path,'rb'))
print(len(train_data))

data_ = []
for (qq, anss, context, type) in tqdm(train_data):
    anss_ = []
    if(len(context)>500):
        continue
    for ans in anss:
        if(len(ans.strip())==0):
            continue
        if(len(ans.split())>4):
            continue
        if(ans.lower() in sws):
            continue
        if(ans.split()[0].lower() in ['the', 'a']):
            ans = ' '.join(ans.split()[1:])
        anss_.append(ans)
    if(len(anss_)>0):
        data_.append((qq, anss_, context, type))

##################################################################################################
train_data      = data_
print(len(train_data))
train_data          = [t for t in train_data if t[-1] == 'factoid_snippet']

all_questions   = []
exclude_phrases = ['copyright', 'et al.', 'all rights reserved', '?', '__']
pbar            = tqdm(train_data)
for _, _, snippet, _ in pbar:
    snippet_proc = re.sub("\(.+?\)","",snippet)
    # pprint([snippet_proc, nlp(snippet_proc)])
    if(any(phr in snippet_proc for phr in exclude_phrases)):
        continue
    if(len(snippet_proc.split())<6):
        continue
    if(len(snippet_proc.split())>30):
        continue
    try:
        res = nlp(snippet_proc)
        for pair in res:
            all_questions.append((pair['question'], snippet, pair['answer'], ''))
    except:
        pass
    pbar.set_description(str(len(all_questions)))


import xlsxwriter

def write_row(ws, row, data):
    for col in range(len(data)):
        ws.write(row, col, data[col])

workbook    = xlsxwriter.Workbook('/home/dpappas/bioasq_factoid/t5_bioasq_snippets_only.xlsx')
worksheet1  = workbook.add_worksheet('questions')

for row_num in range(len(all_questions)):
    write_row(worksheet1, row_num, all_questions[row_num])

workbook.close()
