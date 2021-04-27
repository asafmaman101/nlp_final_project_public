import os
import sys
sys.path.append('/home/fodl/asafmaman/PycharmProjects/nlp_final_project_private')
from datetime import datetime
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs

from utils import load_data, clean_unnecessary_spaces


def import_datasets_legacy():
    print("importing data...")
    # Google Data
    print(os.getcwd())
    train_df = pd.read_csv("data/train.tsv", sep="\t").astype(str)
    eval_df = pd.read_csv("data/dev.tsv", sep="\t").astype(str)
    train_df = train_df.loc[train_df["label"] == "1"]
    eval_df = eval_df.loc[eval_df["label"] == "1"]
    train_df = train_df.rename(columns={"sentence1": "input_text", "sentence2": "target_text"})
    eval_df = eval_df.rename(columns={"sentence1": "input_text", "sentence2": "target_text"})
    train_df = train_df[["input_text", "target_text"]]
    eval_df = eval_df[["input_text", "target_text"]]
    train_df["prefix"] = "paraphrase"
    eval_df["prefix"] = "paraphrase"

    # MSRP Data
    train_df = pd.concat([train_df, load_data("data/msr_paraphrase_train.txt", "#1 String", "#2 String", "Quality"), ])
    eval_df = pd.concat([eval_df, load_data("data/msr_paraphrase_test.txt", "#1 String", "#2 String", "Quality"), ])

    # Quora Data
    # The Quora Dataset is not separated into train/test, so we do it manually the first time.
    df = load_data("data/quora_duplicate_questions.tsv", "question1", "question2", "is_duplicate")
    q_train, q_test = train_test_split(df)
    q_train.to_csv("data/quora_train.tsv", sep="\t")
    q_test.to_csv("data/quora_test.tsv", sep="\t")

    # The code block above only needs to be run once.
    # After that, the two lines below are sufficient to load the Quora dataset.
    # q_train = pd.read_csv("data/quora_train.tsv", sep="\t")
    # q_test = pd.read_csv("data/quora_test.tsv", sep="\t")
    train_df = pd.concat([train_df, q_train])
    eval_df = pd.concat([eval_df, q_test])


    train_df = train_df[["prefix", "input_text", "target_text"]]
    eval_df = eval_df[["prefix", "input_text", "target_text"]]
    train_df = train_df.dropna()
    eval_df = eval_df.dropna()
    train_df["input_text"] = train_df["input_text"].apply(clean_unnecessary_spaces)
    train_df["target_text"] = train_df["target_text"].apply(clean_unnecessary_spaces)
    eval_df["input_text"] = eval_df["input_text"].apply(clean_unnecessary_spaces)
    eval_df["target_text"] = eval_df["target_text"].apply(clean_unnecessary_spaces)

    print('data imported --- v')
    print(train_df)


    return eval_df, train_df


def main():
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.ERROR)

    model_args = Seq2SeqArgs()
    model_args.eval_batch_size = 4
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_steps = 5000
    model_args.evaluate_during_training_verbose = True
    model_args.fp16 = False
    model_args.learning_rate = 5e-5
    model_args.max_seq_length = 128
    model_args.num_train_epochs = 5
    model_args.overwrite_output_dir = False
    model_args.reprocess_input_data = True
    model_args.save_eval_checkpoints = True
    model_args.save_steps = -1
    model_args.save_model_every_epoch = True
    model_args.train_batch_size = 4
    model_args.use_multiprocessing = False
    model_args.do_sample = True
    model_args.num_beams = None
    model_args.num_return_sequences = 3
    model_args.max_length = 128
    model_args.top_k = 50
    model_args.top_p = 0.95
    model_args.n_gpu = 1
    experiment_name = "bart-large-paws"
    model_args.output_dir = experiment_name
    model_args.best_model_dir = 'best_model/' + experiment_name
    model_args.wandb_experiment = experiment_name
    model_args.wandb_project = "NLP Project experiments"

    encoder_decoder_name = "facebook/bart-large"

    train_df = pd.read_csv('/home/fodl/asafmaman/PycharmProjects/nlp_final_project_private/'
                           'paraphrasing/data/cleaned_labeled/'
                           'paws_train_clean.csv')
    eval_df = pd.read_csv('/home/fodl/asafmaman/PycharmProjects/nlp_final_project_private/'
                          'paraphrasing/data/cleaned_labeled/'
                          'paws_test_clean_no_train_overlap.csv')

    train_df = train_df[train_df['is_duplicate'] == 1][['sentence1', 'sentence2']]
    train_df['prefix'] = 'paraphrase'
    train_df = train_df.rename(columns={"sentence1": "input_text", "sentence2": "target_text"})
    # positive = positive.rename(columns={"sentence2": "input_text", "sentence1": "target_text"})
    train_df = train_df[['input_text', 'target_text', 'prefix']]
    train_df = train_df.dropna()

    eval_df = eval_df[eval_df['is_duplicate'] == 1][['sentence1', 'sentence2']]
    eval_df['prefix'] = 'paraphrase'
    eval_df = eval_df.rename(columns={"sentence1": "input_text", "sentence2": "target_text"})
    # eval_df = eval_df.rename(columns={"sentence2": "input_text", "sentence1": "target_text"})
    eval_df = eval_df[['input_text', 'target_text', 'prefix']]
    eval_df = eval_df.dropna()

    model = Seq2SeqModel(encoder_decoder_type="bart",
                         encoder_decoder_name=encoder_decoder_name,
                         args=model_args, cuda_device=3)
    print(train_df)
    model.train_model(train_df, eval_data=eval_df)


if __name__ == '__main__':
    main()

