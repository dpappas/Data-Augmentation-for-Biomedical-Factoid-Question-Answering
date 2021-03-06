# Data-Augmentation-for-Biomedical-Factoid-Question-Answering

This repository includes the code to train and evaluate all models mentioned in the paper 
"Data Augmentation for Biomedical Factoid Question Answering" presented in BIONLP 2022 workshop of ACL.

The data used in the paper can be found in the following webpage: 
http://nlp.cs.aueb.gr/publications.html 

After downloading the data you should unzip the file and change the paths in the `train.py` file 

## How to train:

You could run python train.py --help to see all parameters.


An example training can be seen below

`python train.py 
--train_path=/home/dpappas/bioasq_factoid/pubmed_factoid_extracted_data.p 
--dev_path=/home/dpappas/bioasq_factoid/pubmed_factoid_extracted_data_dev.p 
--keep_only=factoid_snippet 
--batch_size=16 
--augment_with=w2v_embed 
--how_many_aug=10000 
--augment_strategy=separate 
--prefix=w2v_embed_10k_albert`

## How to eval:

After training you could evaluate on dev set or test set using the trained model.

`python3.6 eval.py 
--trained_model_path=some_model_path.pth.tar 
--data_path=pubmed_factoid_extracted_data_test.p 
--model_name=ktrapeznikov/albert-xlarge-v2-squad-v2 
--transformer_size=2048`


