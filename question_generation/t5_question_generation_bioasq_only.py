
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

# book    = xlwt.Workbook()
# sheet1  = book.add_sheet("questions")
#
# for row_num in range(len(all_questions)):
#     row = sheet1.row(row_num)
#     for col_num in range(len(all_questions[row_num])):
#         row.write(col_num, all_questions[row_num][col_num])
#
# book.save('/home/dpappas/bioasq_factoid/t5_bioasq_snippets_only.xlsx')

import xlsxwriter
# 3817

def write_row(ws, row, data):
    for col in range(len(data)):
        ws.write(row, col, data[col])

workbook    = xlsxwriter.Workbook('/home/dpappas/bioasq_factoid/t5_bioasq_snippets_only.xlsx')
worksheet1  = workbook.add_worksheet('questions')
# write_row(worksheet1, 0, ['url', 'type', 'excel_file', 'Trademark Phrase', 'Snippet'])

for row_num in range(len(all_questions)):
    write_row(worksheet1, row_num, all_questions[row_num])

workbook.close()


'''

cd t5_multitask_api/
source venv/bin/activate

python3.6 t5_question_generation.py questions_2.xls "02-2019"  "03-2019" 40000
python3.6 t5_question_generation.py questions_3.xls "03-2019"  "04-2019" 40000
python3.6 t5_question_generation.py questions_4.xls "04-2019"  "05-2019" 40000
python3.6 t5_question_generation.py questions_5.xls "05-2019"  "06-2019" 40000
python3.6 t5_question_generation.py questions_6.xls "06-2019"  "07-2019" 40000
python3.6 t5_question_generation.py questions_6.xls "01-2019"  "02-2019" 40000


abstract = "LQTS is typically inherited in an autosomal dominant manner. An exception is LQTS associated with sensorineural deafness (known as Jervell and Lange-Nielsen syndrome), which is inherited in an autosomal recessive manner. Most individuals diagnosed with LQTS have an affected parent. The proportion of LQTS caused by a de novo pathogenic variant is small. Each child of an individual with autosomal dominant LQTS has a 50% risk of inheriting the pathogenic variant. Penetrance of the disorder may vary. Prenatal testing for pregnancies at increased risk and preimplantation genetic diagnosis are possible once the pathogenic variant(s) have been identified in the family.".strip()

snippets = [
"Coronavirus disease 2019 is a respiratory infection caused by severe acute respiratory syndrome coronavirus 2 originating in Wuhan China in 2019.",
]

abstract = "Within the last five years, the State of Texas has experienced either transmission or outbreaks of Ebola, chikungunya, West Nile, and Zika virus infections. Autochthonous transmission of neglected parasitic and bacterial diseases has also become increasingly reported. The rise of such emerging and neglected tropical diseases (NTDs) has not occurred by accident but instead reflects rapidly evolving changes and shifts in a "new" Texas beset by modern and globalizing forces that include rapid expansions in population together with urbanization and human migrations, altered transportation patterns, climate change, steeply declining vaccination rates, and a new paradigm of poverty known as "blue marble health." Summarized here are the major NTDs now affecting Texas. In addition to the vector-borne viral diseases highlighted above, there also is a high level of parasitic infections, including Chagas disease, trichomoniasis, and possibly leishmaniasis and toxocariasis, as well as typhus-group rickettsiosis, a vector-borne bacterial infection. I also highlight some of the key shifts in emerging and neglected disease patterns, partly due to an altered and evolving economic and ecological landscape in the new Texas, and provide some preliminary disease burden estimates for the major prevalent and incident NTDs.".strip()
'''



