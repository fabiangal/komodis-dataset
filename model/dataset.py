""" Reading, processing and converting the KOMODIS dataset into torch features

"""
import os
import json
import copy
import pickle
import numpy as np
from tqdm import tqdm

from itertools import chain

import torch
from torch.utils.data import DataLoader, TensorDataset


SPECIAL_TOKENS = ["<SST>", "<END>", "<PAD>", "<SPK:S>", "<SPK:O>", "<DEL:MOVIE>", "<DEL:ACTOR>", "<DEL:DIRECTOR>",
                  "<DEL:WRITER>", "<DEL:YEAR>", "<DEL:BUDGET>", "<DEL:CERTIFICATE>", "<DEL:COUNTRY>", "<DEL:GENRE0>",
                  "<DEL:GEMRE1>", "<FACT:MOVIE>", "<FACT:ACTOR>", "<FACT:DIRECTOR>", "<FACT:WRITER>", "<FACT:PLOT>",
                  "<OPINION:MOVIE>", "<OPINION:ACTOR>", "<OPINION:DIRECTOR>", "<OPINION:WRITER>", "<OPINION:COUNTRY>",
                  "<OPINION:GENRE>", "<OPINION:BUDGET>", "<OPINION:CERTIFICATE>", "<OPRATE:0>", "<OPRATE:1>",
                  "<OPRATE:2>", "<OPRATE:3>", "<OPRATE:4>", "<OPRATE:5>"]
ATTR_TO_SPECIAL_TOKENS = {"bos_token": "<SST>", "eos_token": "<END>", "pad_token": "<PAD>",
                          "additional_special_tokens": ("<SPK:S>",
                                                        "<SPK:O>",
                                                        "<DEL:MOVIE>",
                                                        "<DEL:ACTOR0>",
                                                        "<DEL:ACTOR1>",
                                                        "<DEL:DIRECTOR>",
                                                        "<DEL:WRITER>",
                                                        "<DEL:YEAR>",
                                                        "<DEL:BUDGET>",
                                                        "<DEL:CERTIFICATE>",
                                                        "<DEL:COUNTRY>",
                                                        "<DEL:GENRE0>",
                                                        "<DEL:GEMRE1>",
                                                        "<FACT:MOVIE>",
                                                        "<FACT:ACTOR0>",
                                                        "<FACT:ACTOR1>",
                                                        "<FACT:DIRECTOR>",
                                                        "<FACT:WRITER>",
                                                        "<FACT:PLOT>",
                                                        "<OPINION:MOVIE>",
                                                        "<OPINION:ACTOR0>",
                                                        "<OPINION:ACTOR1>",
                                                        "<OPINION:DIRECTOR>",
                                                        "<OPINION:WRITER>",
                                                        "<OPINION:COUNTRY>",
                                                        "<OPINION:GENRE>",
                                                        "<OPINION:BUDGET>",
                                                        "<OPINION:CERTIFICATE>",
                                                        "<OPRATE:0>",
                                                        "<OPRATE:1>",
                                                        "<OPRATE:2>",
                                                        "<OPRATE:3>",
                                                        "<OPRATE:4>",
                                                        "<OPRATE:5>")}

TOKEN_ENCODING_MAPPING = {
    "movie#0": "<DEL:MOVIE>",
    "actor#0": "<DEL:ACTOR0>",
    "actor#1": "<DEL:ACTOR1>",
    "director#0": "<DEL:DIRECTOR>",
    "writer#0": "<DEL:WRITER>",
    "year#0": "<DEL:YEAR>",
    "budget#0": "<DEL:BUDGET>",
    "certificate#0": "<DEL:CERTIFICATE>",
    "country#0": "<DEL:COUNTRY>",
    "genre#0": "<DEL:GENRE0>",
    "genre#1": "<DEL:GEMRE1>",
}


class KomodisDataset:
    """ The KOMODIS dataset class.

    INFO: At some point of creating the scripts, we accidently used the term "to binarize" instead of "to tokenize".
    If some functions use that term ("binarize") we always mean "tokenize". TODO: Fix the "binarize"-names :-)

    """
    def __init__(self,
                 tokenizer,     # A tokenizer like the GPT-2 tokenizer from the transformers package.
                 hparams,       # A dict of training properties. See train.py!
                 path_to_data,  # A Path object or string to the current directory of the dataset.
                 debug=False):  # A Boolean, if True only a slice of the data is loaded.
        self.path_to_data = os.path.join(path_to_data, "dataset.json")
        self.debug = debug
        self.tokenizer = tokenizer
        self.hparams = hparams
        self.num_added_tokens = self.tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKENS)
        ids = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[3:])
        self.special_tokens_dict = dict(zip(SPECIAL_TOKENS[3:], ids))
        self.dataset = None

    def load_txt_dataset(self):
        """ Loads the raw dataset.

        """
        with open(self.path_to_data, "r") as f:
            data = json.load(f)

        # split into train, validation and test sets.
        self.dataset = KomodisDataset._split_binarized_corpus(data)

    def tokenize_dataset(self, path_to_save=None):
        """ Preprocessing of the raw data.

        This includes:
            - tokenization
            - fact encoding and preparation
            - attitude encoding and preparation

        Args:
            path_to_save    A Boolean. If True the processed data will be saved on hard disc. Authors note: The idea is
                            to save some time between trainings, e. g. for hyperparameter search. Don't use the pre-
                            processed data with different properties, that require different preprocessing!
        """
        assert self.dataset is not None  # Dataset needs to be loaded first!

        # iterates over all dialogues separated in train, valid and test sets
        for split, dialogues in self.dataset.items():
            print("Processing {} data ...".format(split))
            for num, dialogue in enumerate(tqdm(dialogues)):
                # tokenize dialogue
                utterances = [d["utterance"] for d in dialogue["dialogue"]]
                dialogue["dialogue_processed"] = KomodisDataset._replace_special_tokens(utterances)
                dialogue["dialogue_processed"] = \
                    KomodisDataset._replace_special_moviecorpus_tokens(dialogue["dialogue_processed"])
                dialogue["dialogue_binarized"] = [self.tokenizer.encode(d) for d in dialogue["dialogue_processed"]]

                # process facts
                dialogue["facts_binarized"] = self._process_and_binarize_facts(dialogue["facts"])

                # process attitudes
                dialogue["attitudes_binarized"] = \
                    self._process_and_binarize_attitudes(dialogue["attitudes"],
                                                         attitude_sentences=self.hparams.attitude_sentences)

                if self.debug and num > 32:
                    break

        # --- saving data ---
        if path_to_save is not None:
            path = os.path.join(self.path_to_data, path_to_save)
            if not os.path.exists(os.path.dirname(path)):
                os.mkdir(os.path.dirname(path))
            with open(path, "wb") as f:
                pickle.dump(self.dataset, f)

    def get_torch_features(self, split, batch_size, num_wrong_utts=1, distributed=False):
        """ Creates torch features for training.

        Args:
            split           A string. Determins which split (train, valid or test) you want to use.
            batch_size      An Integer. Simply feed the batchsize of your model.
            num_wrong_utts  An Integer. The number of random distractors for the classification loss.
            distributed     A Boolean. Not implemented yet.

        """
        if distributed:
            print("WARNING: Not implemented yet!")

        # load preprocessed data (or preprocess if needed)
        if "dialogue_binarized" not in self.dataset[split][0]:
            self.tokenize_dataset()
        samples = self._convert_dialogues_to_samples(split=split, num_wrong_utts=num_wrong_utts)

        # create features
        features = {
            "input_ids": [],
            "mc_token_ids": [],
            "lm_labels": [],
            "mc_labels": [],
            "token_type_ids": []
        }

        # Converts all samples into processed sequences for the gpt-2 model.
        for sample in samples:
            num_wrongs = 0
            # Randomly choose the position of the correct answer.
            true_answer_id = int(np.random.rand() * (1 + num_wrong_utts))
            # Generate all sequences (the correct one and all wrong ones).
            for num in range(1 + num_wrong_utts):
                if num == true_answer_id:
                    seqs = self._convert_sample_to_sequences(facts=sample["facts"],
                                                             attitudes=sample["attitudes"],
                                                             history=sample["dialogue_history"],
                                                             reply=sample["label_utterance"][0],
                                                             lm_labels=True)
                else:
                    seqs = self._convert_sample_to_sequences(facts=sample["facts"],
                                                             attitudes=sample["attitudes"],
                                                             history=sample["dialogue_history"],
                                                             reply=sample["wrong_utterances"][num_wrongs],
                                                             lm_labels=False)
                    num_wrongs += 1

                features["input_ids"].append(seqs["input_ids"])
                features["token_type_ids"].append(seqs["token_type_ids"])
                features["mc_token_ids"].append(seqs["mc_token_ids"])
                features["lm_labels"].append(seqs["lm_labels"])
                features["mc_labels"].append(true_answer_id)

        # padding
        features_padded = self._pad_features(features, padding=self.tokenizer.pad_token_id)

        features_combined = {
            "input_ids": [],
            "mc_token_ids": [],
            "lm_labels": [],
            "mc_labels": [],
            "token_type_ids": []
        }

        # Reformat the sequences
        for num in tqdm(range(int(len(features["input_ids"])/(1 + num_wrong_utts)))):
            sst = num * (1 + num_wrong_utts)
            end = sst + 1 + num_wrong_utts
            input_ids = features_padded["input_ids"][sst:end]
            token_type_ids = features_padded["token_type_ids"][sst:end]
            mc_token_ids = features_padded["mc_token_ids"][sst:end]
            lm_labels = features_padded["lm_labels"][sst:end]
            mc_labels = features_padded["mc_labels"][sst]
            features_combined["input_ids"].append(input_ids)
            features_combined["token_type_ids"].append(token_type_ids)
            features_combined["mc_token_ids"].append(mc_token_ids)
            features_combined["lm_labels"].append(lm_labels)
            features_combined["mc_labels"].append(mc_labels)

        # PyTorch conversion
        torch_features = []
        for key, value in features_combined.items():
            torch_features.append(torch.tensor(value))
        dataset = TensorDataset(*torch_features)

        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = None

        # Only shuffle the train set. Ignore the distributed flag here.
        if not distributed or split != "train":
            shuffle = False
        else:
            shuffle = True

        loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, shuffle=shuffle)

        return loader, sampler

    def load_dataset(self, path):
        """ Loads a (already processed) dataset into the object.
        """
        with open(path, "rb") as f:
            ds = pickle.load(f)
            for split in ["train", "valid", "test"]:
                if split not in ds:
                    raise Exception("Dataset does not contain {} data.".format(split))
                if "dialogue_binarized" not in ds[split][0]:
                    raise Exception("Data is not binarized!")
            self.dataset = ds

    def _process_and_binarize_facts(self, facts, inference=False):
        """ Processes and tokenizes the facts from the dataset.
        """
        def create_seqs_with_delex():
            """ The sequences if delexicalisation is used """
            if fact["relation"] in ["has_trivia", "has_plot"]:
                facts_processed[speaker].append((
                    fact_tokenized,
                    len_fact * self.tokenizer.encode(token)
                ))

        # For inference we don't need first- and second-speaker separation.
        if inference:
            facts_processed = {"inference": []}
            speakers = ["inference"]
            facts = {"inference": facts}
        else:
            facts_processed = {"first_speaker": [], "second_speaker": []}
            speakers = ["first_speaker", "second_speaker"]

        for speaker in speakers:
            ss_facts = facts[speaker]
            for fact in ss_facts:
                fact_tokenized = self.tokenizer.encode(fact["object"])
                len_fact = len(fact_tokenized)
                token = KomodisDataset._get_correct_fact_token(subject=fact["subject"], relation=fact["relation"])
                create_seqs_with_delex()

        if inference:
            facts_processed = facts_processed["inference"]

        return facts_processed

    def _process_and_binarize_attitudes(self, attitudes, attitude_sentences, inference=False):
        """ Processes and tokenizes the attitudes from the dataset.
        """
        # For inference we don't need first- and second-speaker separation.
        if inference:
            attitudes_processed = {"inference": []}
            speakers = ["inference"]
            attitudes = {"inference": attitudes}
        else:
            attitudes_processed = {"first_speaker": [], "second_speaker": []}
            speakers = ["first_speaker", "second_speaker"]

        for speaker in speakers:
            ss_atts = attitudes[speaker]
            for num, attitude in enumerate(ss_atts):
                token = KomodisDataset._get_correct_att_token(subject=attitude["subject"],
                                                              relation=attitude["relation"])
                if attitude_sentences:
                    repl_att_sent = KomodisDataset._replace_special_tokens([attitude["source"]])
                    ss = self.tokenizer.encode(repl_att_sent[0])
                    attitudes_processed[speaker].append((
                        ss,
                        len(ss) * self.tokenizer.encode(token)
                    ))
                else:
                    attitudes_processed[speaker].append((
                        self.tokenizer.encode("<OPRATE:{}>".format(attitude["object"])),
                        self.tokenizer.encode(token)
                    ))

        if inference:
            attitudes_processed = attitudes_processed["inference"]

        return attitudes_processed

    @staticmethod
    def _get_correct_fact_token(subject, relation):
        if relation == "has_plot":
            return "<FACT:PLOT>"
        if subject == "movie#0":
            return "<FACT:MOVIE>"
        if subject == "actor#0":
            return "<FACT:ACTOR0>"
        if subject in ["actor#1", "actor#2"]:
            return "<FACT:ACTOR1>"
        if subject == "writer#0":
            return "<FACT:WRITER>"
        if subject == "director#0":
            return "<FACT:DIRECTOR>"

    @staticmethod
    def _get_correct_att_token(subject, relation):
        if relation == "has_bot_certificate_attitude":
            return "<OPINION:CERTIFICATE>"
        if relation == "has_bot_budget_attitude":
            return "<OPINION:BUDGET>"
        if subject == "movie#0":
            return "<OPINION:MOVIE>"
        if subject == "actor#0":
            return "<OPINION:ACTOR0>"
        if subject in ["actor#1", "actor#2"]:
            return "<OPINION:ACTOR1>"
        if subject == "writer#0":
            return "<OPINION:WRITER>"
        if subject == "director#0":
            return "<OPINION:DIRECTOR>"
        if subject in ["genre#0", "genre#1"]:
            return "<OPINION:GENRE>"
        if subject == "country#0":
            return "<OPINION:COUNTRY>"

    @staticmethod
    def _replace_special_tokens(utterances):
        """ In the original dataset, the special tokens differ from the ones defined here. This function replaces
         the original tokens with the ones from the tokenizer.
         """
        utterances_fixed = []
        for utterance in utterances:
            for original, new in TOKEN_ENCODING_MAPPING.items():
                utterance = utterance.replace(original, new)
            utterances_fixed.append(utterance)
        return utterances_fixed

    @staticmethod
    def _replace_special_moviecorpus_tokens(dialogue):
        """ Replaces [eou] tokens and add [end] tokens.
        """
        new_dialogue = []
        for utterance in dialogue:
            tokens = utterance.split(" ")
            new_tokens = []
            for i in range(len(tokens)):
                if i == 0:
                    new_tokens.append(tokens[i])
                else:
                    if tokens[i] in ["[eou]", "[EOU]"]:
                        if tokens[i - 1] in ["?", ".", ",", "!", ";", ":"]:
                            continue
                        else:
                            new_tokens.append(".")
                    else:
                        new_tokens.append(tokens[i])
            new_dialogue.append(" ".join(new_tokens))
        return new_dialogue

    def convert_clear_txt_to_sequences(self, facts, attitudes, history, reply=None):
        """ For inference only! """
        facts = self._process_and_binarize_facts(facts, inference=True)
        attitudes = self._process_and_binarize_attitudes(attitudes,
                                                         attitude_sentences=self.hparams.attitude_sentences,
                                                         inference=True)
        history = self._replace_special_tokens(history)
        history = [self.tokenizer.encode(x) for x in history]
        if reply is None:
            reply = []
        else:
            reply = [reply]
        seqs = self._convert_sample_to_sequences(facts, attitudes, history, reply, lm_labels=False, inference=True)
        return seqs

    def _convert_sample_to_sequences(self,
                                     facts,
                                     attitudes,
                                     history,
                                     reply,
                                     lm_labels=True,
                                     inference=False):
        facts_input_ids = list(chain(*[x[0] for x in facts]))
        facts_token_type_ids = list(chain(*[x[1] for x in facts]))
        atts_input_ids = list(chain(*[x[0] for x in attitudes]))
        atts_token_type_ids = list(chain(*[x[1] for x in attitudes]))

        sequence = copy.deepcopy([[self.tokenizer.bos_token_id] + facts_input_ids + atts_input_ids] + history + [reply])
        if not inference:
            sequence[-1] += [self.tokenizer.eos_token_id]
        sequence = [sequence[0]] + [[self.special_tokens_dict["<SPK:S>"] if (len(sequence) - i) % 2
                                     else self.special_tokens_dict["<SPK:O>"]] + s
                                    for i, s in enumerate(sequence[1:])]
        seqs = {
            "input_ids": list(chain(*sequence))
        }

        def cond(i):
            if i % 2:
                return self.special_tokens_dict["<SPK:O>"]
            return self.special_tokens_dict["<SPK:S>"]

        seqs["token_type_ids"] = [self.tokenizer.bos_token_id] + facts_token_type_ids + atts_token_type_ids + \
                                 [cond(i) for i, s in enumerate(sequence[1:]) for _ in s]

        seqs["mc_token_ids"] = len(seqs["input_ids"]) - 1
        if lm_labels:
            seqs["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
        else:
            seqs["lm_labels"] = [-1] * len(seqs["input_ids"])

        return seqs

    def _convert_dialogue_to_samples(self,
                                     dialogue,
                                     split,
                                     num_wrong_utts=1,
                                     history_length=(1, 15),
                                     max_length=32):
        """ Converts one dialogue in all possible samples, given some settings.

        history_length  (a, b) Number of history utterances between a and b, if possible.

        """
        # create wrong utterances
        num_needed_wrong_utts = (len(dialogue["dialogue_binarized"]) - 1) * num_wrong_utts
        wrong_utts = self._compute_wrong_dialogues(split=split,
                                                   num=num_needed_wrong_utts,
                                                   only_movie=dialogue["movie_title"],
                                                   exclude_id=dialogue["dialogue_id"])

        samples = []
        for num in range(len(dialogue["dialogue_binarized"]) - 1):
            # determine which speaker is system for current sample
            if num % 2 == 0:
                speaker = "second_speaker"
            else:
                speaker = "first_speaker"

            # boundaries for wrong utterances
            sst = num_wrong_utts * num
            end = sst + num_wrong_utts

            # number of previous utterances
            r = np.random.randint(history_length[0], history_length[1] + 1)
            lower = num + 1 - r
            if lower < 0:
                lower = 0

            # check for max length
            t = 0
            skip = False
            while True:
                len_hist = len(list(chain(*dialogue["dialogue_binarized"][lower + t:num + 1])))
                len_label = len(dialogue["dialogue_binarized"][num + 1])
                len_facts = len(list(chain(*[x[0] for x in dialogue["facts_binarized"][speaker]])))
                len_atts = len(list(chain(*[x[0] for x in dialogue["attitudes_binarized"][speaker]])))
                len_wut = max([len(x) for x in wrong_utts[sst:end]])

                # 3 tokens:              start token, end token, token for reply
                # (num + 1 - lower -t):  plus one token per utterance in the history
                num_special_tokens = 3 + (num + 1 - lower - t)

                # check both length: correct utterance and the longest wrong utterance
                if (len_hist + len_facts + len_atts + len_wut + num_special_tokens) <= max_length and \
                        (len_hist + len_label + len_facts + len_atts + num_special_tokens) <= max_length:
                    break

                t += 1

                if lower + t == num + 1:
                    skip = True
                    break

            if not skip:
                samples.append({
                    "label_utterance": [dialogue["dialogue_binarized"][num + 1]],
                    "dialogue_history": dialogue["dialogue_binarized"][lower + t:num + 1],
                    "facts": dialogue["facts_binarized"][speaker],
                    "attitudes": dialogue["attitudes_binarized"][speaker],
                    "wrong_utterances": wrong_utts[sst:end],
                })

        return samples

    def _convert_dialogues_to_samples(self, split, num_wrong_utts=1):
        samples = []
        print("Processing {}-data.".format(split))
        for dialogue in tqdm(self.dataset[split]):
            if self.debug and ("dialogue_binarized" not in dialogue or len(samples) > 64):
                break
            samples += self._convert_dialogue_to_samples(dialogue, split, num_wrong_utts, (3, 5),
                                                         self.hparams.max_input_length)
        return samples

    def _compute_wrong_dialogues(self, split, num, only_movie=None, exclude_id=None):
        """ Returns random wrong utterances

        Args:
            split       A string. One of: "train", "valid", "test".
            num         An integer. Number of wrong utterances.
            only_movie  A string. If given, only utterances of the given movie are returned.
            exclude_id  An integer. If given, the movie with that prepared_id is ignored.

        """
        if only_movie is not None:
            candidates = [movie for movie in self.dataset[split] if movie['movie_title'] == only_movie]
        else:
            candidates = self.dataset[split]

        utterances = []
        for movie in candidates:
            if exclude_id is not None:
                if movie["dialogue_id"] == exclude_id:
                    continue
            # --- if not processed (can happen in debug mode), process: ---
            if "dialogue_binarized" not in movie:
                new_utterances = [d["utterance"] for d in movie["dialogue"]]
                movie["dialogue_processed"] = KomodisDataset._replace_special_tokens(new_utterances)
                movie["dialogue_processed"] = \
                    KomodisDataset._replace_special_moviecorpus_tokens(movie["dialogue_processed"])
                movie["dialogue_binarized"] = [self.tokenizer.encode(d) for d in movie["dialogue_processed"]]
            # --------------------------------------------------------------
            utterances += movie['dialogue_binarized']
        np.random.shuffle(utterances)

        return utterances[:num]

    @staticmethod
    def _split_binarized_corpus(dataset):
        """ Splits the corpus in train, eval and test. """

        # collect movie titles
        titles = []
        for dialogue in dataset:
            if dialogue['movie_title'] not in titles:
                titles.append(dialogue['movie_title'])
        num_train = int(len(titles) * 0.8)
        num_eval = int(len(titles) * 0.1)
        np.random.shuffle(titles)
        titles_split = {
            "train": titles[0:num_train],
            "valid": titles[num_train:num_train + num_eval],
            "test": titles[num_train + num_eval:]
        }
        dataset_splitted = {
            "train": [],
            "valid": [],
            "test": []
        }
        for dialogue in dataset:
            if dialogue['movie_title'] in titles_split['train']:
                dataset_splitted["train"].append(dialogue)
            elif dialogue['movie_title'] in titles_split['valid']:
                dataset_splitted["valid"].append(dialogue)
            elif dialogue['movie_title'] in titles_split['test']:
                dataset_splitted["test"].append(dialogue)

        return dataset_splitted

    def _pad_features(self, features, padding):
        max_l = max(len(feature) for feature in features["input_ids"])
        if self.debug:
            # If debug mode, we want to see if the maximum size of input data fits into the vram for training.
            max_l = self.hparams.max_input_length
        for name in ["input_ids", "token_type_ids", "lm_labels"]:
            features[name] = [x + [padding if name != "lm_labels" else -1] * (max_l - len(x)) for x in features[name]]
        return features
