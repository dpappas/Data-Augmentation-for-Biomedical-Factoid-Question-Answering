
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

exclude_phrases = [
    'copyright',
    'et al.',
    'all rights reserved',
    '?',
    '__'
]

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

# book.save("questions.xls")
book.save(sys.argv[1])



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



