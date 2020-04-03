""" Script to run the training.
"""
import os
import sys
import torch
from argparse import ArgumentParser

from model import komodis

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--path_to_data", type=str, default="data/", help="Directory of the dataset")
    parser.add_argument("--model_checkpoint", type=str, default="gpt",
                        help="Short name of the model")
    parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidates for training")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--lm_coef", type=float, default=1.0, help="LM loss coefficient")
    parser.add_argument("--mc_coef", type=float, default=1.0, help="Multiple-choice loss coefficient")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--eval_before_start", action='store_true',
                        help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--max_input_length", type=int, default=256, help="The maximum length of sequences for "
                                                                          "training. All samples are padded to that "
                                                                          "length.")
    parser.add_argument("--attitude_sentences", action="store_true", help="If set, the attitudes are generated as "
                                                                          "real sentences instead of single tokens.")
    parser.add_argument("--debug", action="store_true", help="If true only a slice of the data is processed and "
                                                             "some samples are displayed on console.")
    args = parser.parse_args()

    # Make sure that the current working directory equals the directory of this script.
    os.chdir(os.path.dirname(__file__))

    if args.dataset == "komodis":
        trainer = komodis.KomodisTrainer(path_to_pretrained_model="data/pretrained_models/gpt2/",
                                         path_to_vocab_file="data/tokenizers/gpt2-vocab.json",
                                         path_to_merges_file="data/tokenizers/gpt2-merges.txt",
                                         hparams=args)
    else:
        print("{} not implemented.".format(args.dataset))
        sys.exit()

    trainer.train()
