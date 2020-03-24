"""
Training script useful for debugging UDify and AllenNLP code
"""

import os
import copy
import datetime
import logging
import argparse
import glob

from allennlp.common import Params
from allennlp.common.util import import_submodules
from allennlp.commands.train import train_model

from udify import util

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="", type=str, help="Log dir name")
parser.add_argument("--base_config", default="config/udify_base.json", type=str, help="Base configuration file")
parser.add_argument("--config", default=[], type=str, nargs="+", help="Overriding configuration files")
parser.add_argument("--dataset_dir", default="data/ud-treebanks-v2.5", type=str, help="The path containing all UD treebanks")
parser.add_argument("--batch_size", default=32, type=int, help="The batch size used by the model; the number of training sentences is divided by this number.")
parser.add_argument("--device", default=None, type=int, help="CUDA device; set to -1 for CPU")
parser.add_argument("--resume", type=str, help="Resume training with the given model")
parser.add_argument("--lazy", default=None, action="store_true", help="Lazy load the dataset")
parser.add_argument("--cleanup_archive", action="store_true", help="Delete the model archive")
parser.add_argument("--replace_vocab", action="store_true", help="Create a new vocab and replace the cached one")
parser.add_argument("--archive_bert", action="store_true", help="Archives the finetuned BERT model after training")
parser.add_argument("--predictor", default="udify_predictor", type=str, help="The type of predictor to use")

args = parser.parse_args()

log_dir_name = args.name
if not log_dir_name:
    file_name = args.config[0] if args.config else args.base_config
    log_dir_name = os.path.basename(file_name).split(".")[0]

if not args.name == "multilingual":
    train_file = args.name + "-ud-train.conllu"
    pathname = os.path.join(args.dataset_dir, "*", train_file)
    train_path = glob.glob(pathname).pop()
    treebank_path = os.path.dirname(train_path)

    if train_path:
        logger.info(f"found training file: {train_path}, calculating the warmup and start steps")
    
        f = open(train_path, 'r', encoding="utf-8")
        sentence_count = 0
        for line in f.readlines():
            if line.isspace():
                sentence_count += 1
        num_warmup_steps = round(sentence_count / args.batch_size)

configs = []

if not args.resume:
    serialization_dir = os.path.join("logs", log_dir_name, datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"))

    overrides = {}
    if args.device is not None:
        overrides["trainer"] = {"cuda_device": args.device}
    if args.lazy is not None:
        overrides["dataset_reader"] = {"lazy": args.lazy}
    configs.append(Params(overrides))
    for config_file in args.config:
        configs.append(Params.from_file(config_file))
    configs.append(Params.from_file(args.base_config))
else:
    serialization_dir = args.resume
    configs.append(Params.from_file(os.path.join(serialization_dir, "config.json")))

train_params = util.merge_configs(configs)

if not args.name == "multilingual":
    # overwrite the default params with the language-specific ones
    for param in train_params:
        if param == "train_data_path":
            train_params["train_data_path"] = os.path.join(treebank_path, f"{args.name}-ud-train.conllu")
        if param == "validation_data_path":
            train_params["validation_data_path"] = os.path.join(treebank_path, f"{args.name}-ud-dev.conllu")
        if param == "test_data_path":
            train_params["test_data_path"] = os.path.join(treebank_path, f"{args.name}-ud-test.conllu")
        
        if param == "vocabulary":
            train_params["vocabulary"]["directory_path"] = f"data/vocab/{args.name}/vocabulary"
        
        if param == "trainer":
            for sub_param in train_params["trainer"]:
                if sub_param == "learning_rate_scheduler":
                    train_params["trainer"]["learning_rate_scheduler"]["warmup_steps"] = num_warmup_steps
                    train_params["trainer"]["learning_rate_scheduler"]["start_step"] = num_warmup_steps
                    
                    logger.info(f"changing warmup and start steps for {train_path} to {num_warmup_steps}")
                
if "vocabulary" in train_params:
    # Remove this key to make AllenNLP happy
    train_params["vocabulary"].pop("non_padded_namespaces", None)

predict_params = train_params.duplicate()

import_submodules("udify")

try:
    util.cache_vocab(train_params)
    train_model(train_params, serialization_dir, recover=bool(args.resume))
except KeyboardInterrupt:
    logger.warning("KeyboardInterrupt, skipping training")

dev_file = predict_params["validation_data_path"]
test_file = predict_params["test_data_path"]

dev_pred, dev_eval, test_pred, test_eval = [
    os.path.join(serialization_dir, name)
    for name in ["dev.conllu", "dev_results.json", "test.conllu", "test_results.json"]
]

if dev_file != test_file:
    util.predict_and_evaluate_model(args.predictor, predict_params, serialization_dir, dev_file, dev_pred, dev_eval)

util.predict_and_evaluate_model(args.predictor, predict_params, serialization_dir, test_file, test_pred, test_eval)

if args.archive_bert:
    bert_config = "config/archive/bert-base-multilingual-cased/bert_config.json"
    util.archive_bert_model(serialization_dir, bert_config)

util.cleanup_training(serialization_dir, keep_archive=not args.cleanup_archive)
