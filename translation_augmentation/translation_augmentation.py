
import os, pickle, re, json, random
from pprint import pprint
from tqdm import tqdm
from deep_translator import GoogleTranslator
import pickle

pprint(GoogleTranslator.get_supported_languages(as_dict=True))

d = pickle.load(open('/home/dpappas/bioasq_factoid/pubmed_factoid_extracted_data.p','rb'))

d = [t for t in d if t[-1]=='factoid_snippet']

languages = ['fr', 'es', 'de']

new_created_data    = set()
for q, answers, context, _ in tqdm(d):
    ##################################################################################
    for dest_lang in languages:
        text_in_dest = GoogleTranslator(source='en', target=dest_lang).translate(q)
        text_in_en = GoogleTranslator(source=dest_lang, target='en').translate(text_in_dest)
        if (q != text_in_en):
            new_created_data.add((text_in_en, tuple(answers), context, 'transl_q_{}'.format(dest_lang), q))
        ##############################################################################
        text_in_dest = GoogleTranslator(source='en', target=dest_lang).translate(context)
        text_in_en = GoogleTranslator(source=dest_lang, target='en').translate(text_in_dest)
        if (context != text_in_en and any(ans.lower() in text_in_en.lower() for ans in answers)):
            new_created_data.add((q, tuple(answers), text_in_en, 'transl_c_{}'.format(dest_lang), context))
    print(len(new_created_data))
    ##################################################################################

pickle.dump(new_created_data, open('/home/dpappas/bioasq_factoid/pubmed_factoid_translate_all_ion.p','wb'))









