""" Base class for a trainer.

This class includes some basic PyTorch code for multi GPU training, metrics, train and validation code.



"""
import os
import torch
from datetime import datetime as dt
from pathlib import Path
from pprint import pformat
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from transformers import CONFIG_NAME, WEIGHTS_NAME

from abc import ABCMeta, abstractmethod


class BaseTrainer(metaclass=ABCMeta):
    def __init__(self, hparams):
        self.hps = hparams          # A dict. Model parameters, see train.py for more information.
        self.dataset_obj = None     # An instance of class KomodisDataset.
        self.model = None           # The PyTorch model.
        self.tokenizer = None       # The transformers tokenizer.
        self.optimizer = None       # A PyTorch optimizer.

    def train(self):
        """ """
        loader_train, sampler_train = self.dataset_obj.get_torch_features(split="train",
                                                                          batch_size=self.hps.train_batch_size,
                                                                          num_wrong_utts=self.hps.num_candidates - 1)
        loader_valid, sampler_valid = self.dataset_obj.get_torch_features(split="valid",
                                                                          batch_size=self.hps.valid_batch_size,
                                                                          num_wrong_utts=self.hps.num_candidates - 1)

        # Print (readable) values of the first three samples, if debug mode.
        if self.hps.debug:
            print("*** SAMPLES ***")
            input_ids = loader_train.dataset.tensors[0].tolist()
            token_type_ids = loader_train.dataset.tensors[4].tolist()
            mc_labels = loader_train.dataset.tensors[3].tolist()

            for sample in range(3):
                print("SAMPLE {}:".format(sample))
                input_ids_txt = [self.tokenizer.decode(input_ids[sample][i],
                                                       clean_up_tokenization_spaces=False)
                                 for i in range(self.hps.num_candidates)]
                token_type_ids_txt = [self.tokenizer.decode(token_type_ids[sample][i])
                                      for i in range(self.hps.num_candidates)]

                for num in range(self.hps.num_candidates):
                    print("{}. candidate:".format(num+1))
                    print("INPUT IDS: {}".format(input_ids_txt[num]))
                    print("TOKEN TYPE IDS: {}".format(token_type_ids_txt[num]))
                print("Correct utterance: {}".format(mc_labels[sample] + 1))

        trainer = Engine(self.update)
        evaluator = Engine(self.inference)

        # Evaluation at the end of each epoch, as well as at the beginning of the training.
        trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(loader_valid))
        if self.hps.n_epochs < 1:
            trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(loader_valid))
        if self.hps.eval_before_start:
            trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(loader_valid))

        # Adding learning rate.
        args = {
            "loader_train": loader_train
        }
        self.learningrate(trainer=trainer, args=args)

        # Adding metrics to the trainer
        RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
        metrics = self.metrics()
        for name, metric in metrics.items():
            metric.attach(evaluator, name)

        # Printing evaluation results at the end of each epoch.
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED,
                                    lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        # Save model weights after each epoch.
        log_dir = self.mk_logdir(self.hps.model_checkpoint)
        checkpoint_handler = ModelCheckpoint(log_dir, "checkpoint", save_interval=1, n_saved=3)
        trainer.add_event_handler(Events.EPOCH_COMPLETED,
                                  checkpoint_handler,
                                  {"mymodel": getattr(self.model, "module", self.model)})
        torch.save(self.hps, log_dir / "model_training_args.bin")
        getattr(self.model, "module", self.model).config.to_json_file(os.path.join(log_dir, CONFIG_NAME))
        self.tokenizer.save_pretrained(log_dir)

        # Run the training
        trainer.run(loader_train, max_epochs=self.hps.n_epochs)

        os.rename(checkpoint_handler._saved[-1][1][-1], os.path.join(log_dir, WEIGHTS_NAME))

    @staticmethod
    def mk_logdir(name_model):
        """ """
        timestamp = dt.now().strftime("%b%d_%H-%M-%S")
        logdir = Path("results", timestamp + "_" + name_model)
        return logdir

    @abstractmethod
    def update(self, engine, batch):
        raise NotImplementedError

    @abstractmethod
    def inference(self, engine, batch):
        raise NotImplementedError

    @abstractmethod
    def learningrate(self, trainer, args):
        raise NotImplementedError

    @abstractmethod
    def metrics(self):
        return {}
