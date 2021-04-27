import logging

from paraphrasing.utils import plot_histograms, import_cleaned_data
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs

from paraphrasing.utils import calculate_accuracy_and_f1


def main():
    model_args = Seq2SeqArgs()
    model_args.eval_batch_size = 1  # dont change
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
    model_args.train_batch_size = 1  # DON'T CHANGE!!!
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

    dataset_to_evaluate = 'mrpc'
    cuda_device = 3
    threshold = None
    inversed = False
    score_type = 'loss'

    # encoder_decoder_name = "facebook/bart-base"
    # encoder_decoder_name = "facebook/bart-large"
    # encoder_decoder_name = "bart-base-all"
    # encoder_decoder_name = "bart-large-all"
    # encoder_decoder_name = "bart-base-mrpc"
    # encoder_decoder_name = "bart-large-mrpc"
    # encoder_decoder_name = "bart-base-paws"
    # encoder_decoder_name = "bart-large-paws"
    encoder_decoder_name = "bart-base-qqp"
    # encoder_decoder_name = "bart-large-qqp"

    print(dataset_to_evaluate)
    print(encoder_decoder_name)

    negative, positive = import_cleaned_data(dataset_to_evaluate, inversed)

    model = Seq2SeqModel(encoder_decoder_type="bart",
                         encoder_decoder_name=encoder_decoder_name,
                         args=model_args, cuda_device=cuda_device)

    if score_type == 'probs':
        positive_losses = model.project_inference_method(positive)
        negative_losses = model.project_inference_method(negative)
    elif score_type == 'loss':
        positive_losses = model.ce_losses(positive)
        negative_losses = model.ce_losses(negative)
    else:
        raise AssertionError("score_type has to be one of 'loss' or 'probs'")

    plot_histograms(positive_losses=positive_losses, negative_losses=negative_losses,
                    plot_title=(dataset_to_evaluate + '/' + encoder_decoder_name),
                    x_min=0.5, x_max=8)

    calculate_accuracy_and_f1(positive_losses, negative_losses, threshold=threshold)


if __name__ == '__main__':
    main()
