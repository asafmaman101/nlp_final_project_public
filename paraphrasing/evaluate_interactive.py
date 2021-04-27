import os
from datetime import datetime
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs

from utils import load_data, clean_unnecessary_spaces

import numpy as np
import matplotlib.pyplot as plt

import json
from datetime import datetime


def main():
    model_args = Seq2SeqArgs()
    model_args.eval_batch_size = 4
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_steps = 2500
    model_args.evaluate_during_training_verbose = True
    model_args.fp16 = False
    model_args.learning_rate = 5e-5
    model_args.max_seq_length = 128
    model_args.num_train_epochs = 2
    model_args.overwrite_output_dir = False
    model_args.reprocess_input_data = True
    model_args.save_eval_checkpoints = False
    model_args.save_steps = -1
    model_args.train_batch_size = 16
    model_args.use_multiprocessing = False
    model_args.do_sample = True
    model_args.num_beams = None
    model_args.num_return_sequences = 3
    model_args.max_length = 128
    model_args.top_k = 50
    model_args.top_p = 0.95
    model_args.n_gpu = 1
    model_args.wandb_project = None

    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.ERROR)

    model = Seq2SeqModel(encoder_decoder_type="bart",
                         encoder_decoder_name="outputs_23-04-2021/checkpoint-144205-epoch-5",
                         args=model_args, cuda_device=2)

    while True:
        print('first sentence:')
        input_text = input()
        if input_text == 'exit':
            break
        print('second sentence:')
        target_text = input()
        prefix='paraphrase'
        d = dict(input_text=input_text,
                 target_text=target_text,
                 prefix=prefix
                 )


        eval_df = pd.DataFrame([[input_text,target_text,prefix]], columns=d.keys())
        prediction, losses = model.project_inference_method(eval_df)
        print(prediction)
        print(losses)




if __name__ == '__main__':
    main()
