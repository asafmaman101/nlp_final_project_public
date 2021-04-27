import os
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm


def load_data(file_path, input_text_column, target_text_column, label_column, keep_label=1):
    df = pd.read_csv(file_path, sep="\t", error_bad_lines=False)
    df = df.loc[df[label_column] == keep_label]
    df = df.rename(columns={input_text_column: "input_text", target_text_column: "target_text"})
    df = df[["input_text", "target_text"]]
    df["prefix"] = "paraphrase"

    return df


def clean_unnecessary_spaces(out_string):
    if not isinstance(out_string, str):
        warnings.warn(f">>> {out_string} <<< is not a string.")
        out_string = str(out_string)
    out_string = (
        out_string.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
    )
    return out_string


def calculate_accuracy_and_f1(positive_values, negative_values, threshold=None):

    if type(positive_values) is list:
        positive_values = torch.tensor(positive_values)
    if type(negative_values) is list:
        negative_values = torch.tensor(negative_values)

    if threshold is None:
        best = defaultdict(float)
        for threshold in tqdm(torch.linspace(0,15,3000), desc='seaching threshold'):

            true_positive = (positive_values < threshold).sum().item()
            true_negative = (negative_values >= threshold).sum().item()

            total_positive_samples = len(positive_values)
            total_negative_samples = len(negative_values)

            false_negative = total_positive_samples - true_positive
            false_positive = total_negative_samples - true_negative

            if (true_positive + false_positive) == 0:
                continue
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)

            if (total_positive_samples + total_negative_samples) == 0:
                continue

            if (total_positive_samples + total_negative_samples) == 0:
                continue
            accuracy = (true_positive + true_negative) / (total_positive_samples + total_negative_samples)

            if (precision + recall) == 0:
                continue
            f1 = 2 * (precision * recall) / (precision + recall)

            if best['accuracy'] < accuracy:
                best['accuracy'] = accuracy
                best['acc_pre'] = precision
                best['acc_rec'] = recall
                best['acc_thd'] = threshold
                best['acc_f1'] = f1
            if best['f1'] < f1:
                best['f1'] = f1
                best['f1_pre'] = precision
                best['f1_rec'] = recall
                best['f1_thd'] = threshold
                best['f1_acc'] = accuracy

        print(f"thd:{best['acc_thd']:.6f}\taccuracy: {best['accuracy']:.6f}\tprecision: {best['acc_pre']:.6f}\trecall: {best['acc_rec']:.6f}\tF1: {best['acc_f1']:.6f}")
        print(f"thd:{best['f1_thd']:.6f}\taccuracy: {best['f1_acc']:.6f}\tprecision: {best['f1_pre']:.6f}\trecall: {best['f1_rec']:.6f}\tF1: {best['f1']:.6f}")
    else:
        true_positive = (positive_values < threshold).sum().item()
        # true_positive = (positive_values >= threshold).sum().item()
        true_negative = (negative_values >= threshold).sum().item()
        # true_negative = (negative_values < threshold).sum().item()

        total_positive_samples = len(positive_values)
        total_negative_samples = len(negative_values)

        false_negative = total_positive_samples - true_positive
        false_positive = total_negative_samples - true_negative

        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)

        accuracy = (true_positive + true_negative) / (total_positive_samples + total_negative_samples)

        f1 = 2 * (precision * recall) / (precision + recall)

        print(f"thd:{threshold:.6f}\taccuracy: {accuracy:.6f}\tprecision: {precision:.6f}\trecall: {recall:.6f}\tF1: {f1:.6f}")

    return accuracy, f1

# for thd in torch.linspace(0.958, 0.968, 100):
#     accuracy, f1 = calculate_accuracy_and_f1(p, n, thd)
def plot_histograms(positive_losses, negative_losses, save_path: str = None, show_plot=True, bins=50,
                    plot_title='no title specified', x_min=0, x_max=1):

    positive_n, _, _ = plt.hist([positive_losses], bins=np.linspace(x_min, x_max, bins), alpha=0.5, label='positve')
    negative_n, _, _ = plt.hist([negative_losses], bins=np.linspace(x_min, x_max, bins), alpha=0.5, label='negative')

    plt.title(plot_title)
    plt.legend()

    if save_path:
        now_utc_str = datetime.utcnow().strftime("%Y_%m_%d-%H_%M_%S")
        save_path_with_timestamp = os.path.join(save_path,'_', now_utc_str, '.png')
        os.makedirs(save_path_with_timestamp, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f'plot was saved to: {save_path}')

    if show_plot:
        plt.show()

    return positive_n, negative_n


def import_cleaned_data(dataset_to_evaluate, inversed):
    eval_df = None
    if dataset_to_evaluate == 'mrpc':
        eval_df = pd.read_csv('data/cleaned_labeled/mrpc_test_clean_dropped.csv')
    elif dataset_to_evaluate == 'mrpc-test-no-dups':
        eval_df = pd.read_csv('data/cleaned_labeled/mrpc_test_clean_dropped_no_duplicates_with_train.csv')
    elif dataset_to_evaluate == 'mrpc-train':
        eval_df = pd.read_csv('data/cleaned_labeled/mrpc_train_clean_dropped.csv')
    elif dataset_to_evaluate == 'paws':
        eval_df = pd.read_csv('data/cleaned_labeled/paws_test_clean.csv')
    elif dataset_to_evaluate == 'paws-dev':
        eval_df = pd.read_csv('data/cleaned_labeled/paws_dev_clean.csv').iloc[:4000]
    elif dataset_to_evaluate == 'qqp':
        eval_df = pd.read_csv('data/cleaned_labeled/qqp_test_clean.csv').iloc[:10000]
    elif dataset_to_evaluate == 'qqp-dev':
        eval_df = pd.read_csv('data/cleaned_labeled/qqp_test_clean.csv').iloc[:50000]
    elif dataset_to_evaluate == 'qqp-test':
        eval_df = pd.read_csv('data/cleaned_labeled/qqp_test_clean.csv').iloc[50000:]
    elif dataset_to_evaluate == 'all-test':
        eval_df = pd.read_csv('data/cleaned_labeled/all_test_clean.csv').iloc[:5000]
    else:
        raise AssertionError ('no such dataset exists.')
    positive = eval_df[eval_df['is_duplicate'] == 1][['sentence1', 'sentence2']]
    positive['prefix'] = 'paraphrase'
    if inversed:
        positive = positive.rename(columns={"sentence2": "input_text", "sentence1": "target_text"})
    else:
        positive = positive.rename(columns={"sentence1": "input_text", "sentence2": "target_text"})
    positive = positive[['input_text', 'target_text', 'prefix']]
    positive = positive.dropna()
    negative = eval_df[eval_df['is_duplicate'] == 0][['sentence1', 'sentence2']]
    negative['prefix'] = 'paraphrase'
    if inversed:
        negative = negative.rename(columns={"sentence2": "input_text", "sentence1": "target_text"})
    else:
        negative = negative.rename(columns={"sentence1": "input_text", "sentence2": "target_text"})
    negative = negative[['input_text', 'target_text', 'prefix']]
    negative = negative.dropna()
    return negative, positive


def import_data(keep_label, num_of_samples=None):
    # # Google Data
    train_df = pd.read_csv("data/train.tsv", sep="\t").astype(str)
    eval_df = pd.read_csv("data/dev.tsv", sep="\t").astype(str)
    train_df = train_df.loc[train_df["label"] == str(keep_label)]
    eval_df = eval_df.loc[eval_df["label"] == str(keep_label)]
    train_df = train_df.rename(columns={"sentence1": "input_text", "sentence2": "target_text"})
    eval_df = eval_df.rename(columns={"sentence1": "input_text", "sentence2": "target_text"})
    train_df = train_df[["input_text", "target_text"]]
    eval_df = eval_df[["input_text", "target_text"]]
    train_df["prefix"] = "paraphrase"
    eval_df["prefix"] = "paraphrase"

    # # MSRP Data
    # train_df = pd.concat([train_df, load_data("data/msr_paraphrase_train.txt", "#1 String", "#2 String", "Quality"),])
    # eval_df = pd.concat([eval_df, load_data("data/msr_paraphrase_test.txt", "#1 String", "#2 String", "Quality"),])

    # Quora Data
    # The Quora Dataset is not separated into train/test, so we do it manually the first time.
    # df = load_data("data/quora_duplicate_questions.tsv", "question1", "question2", "is_duplicate",
    #                keep_label=keep_label)
    # q_train, q_test = train_test_split(df)
    # q_train.to_csv("data/quora_train.tsv", sep="\t")
    # q_test.to_csv("data/quora_test.tsv", sep="\t")

    # The code block above only needs to be run once.
    # After that, the two lines below are sufficient to load the Quora dataset.
    # q_train = pd.read_csv("data/quora_train.tsv", sep="\t")
    # q_test = pd.read_csv("data/quora_test.tsv", sep="\t")
    # train_df = pd.concat([train_df, q_train])
    # eval_df = pd.concat([eval_df, q_test])

    # train_df = q_train
    # eval_df = q_test

    train_df = train_df[["prefix", "input_text", "target_text"]]
    eval_df = eval_df[["prefix", "input_text", "target_text"]]
    train_df = train_df.dropna()
    eval_df = eval_df.dropna()
    train_df["input_text"] = train_df["input_text"].apply(clean_unnecessary_spaces)
    train_df["target_text"] = train_df["target_text"].apply(clean_unnecessary_spaces)
    eval_df["input_text"] = eval_df["input_text"].apply(clean_unnecessary_spaces)
    eval_df["target_text"] = eval_df["target_text"].apply(clean_unnecessary_spaces)

    if num_of_samples:
        eval_df = eval_df.iloc[:num_of_samples]

    return train_df, eval_df