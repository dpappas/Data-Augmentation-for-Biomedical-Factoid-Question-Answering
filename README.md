# Data-Augmentation-for-Biomedical-Factoid-Question-Answering


## How to train:

You could run python train.py --help to see all parameters.


An example training can be seen below

`python train.py \
--train_path=/home/dpappas/bioasq_factoid/pubmed_factoid_extracted_data.p \
--dev_path=/home/dpappas/bioasq_factoid/pubmed_factoid_extracted_data_dev.p \
--keep_only=factoid_snippet \
--batch_size=16 \
--augment_with=w2v_embed \
--how_many_aug=10000 \
--augment_strategy=separate \
--prefix=w2v_embed_10k_albert`

## How to eval:

After training you could evaluate on dev set or test set using the trained model.

`python3.6 eval.py \
--trained_model_path=some_model_path.pth.tar \
--data_path=pubmed_factoid_extracted_data_dev.p \
--model_name=ktrapeznikov/albert-xlarge-v2-squad-v2 \
--transformer_size=2048`



