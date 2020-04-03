""" This class trains the moviecorpus corpus on a GPT-2 model.

"""
import math
import torch
from torch.nn.parallel import DataParallel
import logging
import transformers

from ignite.contrib.handlers import PiecewiseLinear
from ignite.engine import Events
from ignite.metrics import Loss, Accuracy, MetricsLambda
from transformers import AdamW

import train_base
import dataset


class KomodisTrainer(train_base.BaseTrainer):
    def __init__(self, path_to_pretrained_model,    # A string. The directory of the pretrained model.
                 path_to_vocab_file,                # A string. The directory of the tokenizer vocabularies file.
                 path_to_merges_file,               # A string. The directory of the tokenizer merges file.
                 hparams,                           # A dict. The training hyperparameters.
                 path_to_encoded_data=None):        # A string. The path to the already processed dataset.
        super().__init__(hparams)
        self.path_to_encoded_data = path_to_encoded_data

        self.logger = logging.getLogger(__file__)

        # create tokenizer and pretrained model
        self.model = transformers.GPT2DoubleHeadsModel.from_pretrained(path_to_pretrained_model)
        self.tokenizer = transformers.GPT2Tokenizer(vocab_file=path_to_vocab_file,
                                                    merges_file=path_to_merges_file)
        self.model.to(self.hps.device)

        # prepare dataset
        self.dataset_obj = dataset.KomodisDataset(tokenizer=self.tokenizer, hparams=hparams,
                                                  path_to_data=self.hps.path_to_data,
                                                  debug=self.hps.debug)
        if path_to_encoded_data is None:
            self.dataset_obj.load_txt_dataset()
            self.dataset_obj.tokenize_dataset()
        else:
            self.dataset_obj.load_dataset(path_to_encoded_data)

        self.optimizer = AdamW(self.model.parameters(), lr=self.hps.lr, correct_bias=True)

        num_of_tokens = len(self.tokenizer.encoder)
        self.model.resize_token_embeddings(new_num_tokens=num_of_tokens + self.dataset_obj.num_added_tokens)

        # TODO: Currently this is fixed to the use of 4 gpus. This should be more generic!
        self.model = DataParallel(self.model, device_ids=[0, 1, 2, 3])

    def learningrate(self, trainer, args):
        """ Linear decreasing learning rate. """
        scheduler = PiecewiseLinear(self.optimizer, "lr",
                                    [(0, self.hps.lr), (self.hps.n_epochs * len(args["loader_train"]), 0.0)])
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    def metrics(self):
        """ Metrics.
            nll     negative log-likelihood
            acc     classification accuracy for next-response selection
            ppl     perplexity
        """
        metrics = {
            "nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1), output_transform=lambda x: (x[0][0], x[1][0])),
            "acc": Accuracy(output_transform=lambda x: (x[0][1], x[1][1]))}
        metrics["ppl"] = MetricsLambda(math.exp, metrics["nll"])
        return metrics

    def update(self, engine, batch):
        """ Update step for PyTorch. """
        self.model.train()
        batch = tuple(input_tensor.to(self.hps.device) for input_tensor in batch)
        input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
        (lm_loss), (mc_loss), *_ = self.model(
            input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
            mc_labels=mc_labels, lm_labels=lm_labels)
        loss = (lm_loss * self.hps.lm_coef + mc_loss * self.hps.mc_coef) / self.hps.gradient_accumulation_steps
        loss.sum().backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hps.max_norm)
        if engine.state.iteration % self.hps.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.mean().item()

    def inference(self, engine, batch):
        """ Inference step for PyTorch. """
        self.model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(self.hps.device) for input_tensor in batch)
            input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
            self.logger.info(self.tokenizer.decode(input_ids[0, -1, :].tolist()))
            # if we dont send labels to model, it doesnt return losses
            lm_logits, mc_logits, *_ = self.model(
                input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
            )
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

            return (lm_logits_flat_shifted, mc_logits), (lm_labels_flat_shifted, mc_labels)
