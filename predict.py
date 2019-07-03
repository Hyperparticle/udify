"""
Predict conllu files given a trained model
"""

import os
import shutil
import logging
import argparse

from allennlp.common import Params
from allennlp.common.util import import_submodules
from allennlp.models.archival import archive_model

from udify import util

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("archive_dir", type=str, help="The directory where model.tar.gz resides")
parser.add_argument("input_file", type=str, help="The input file to predict")
parser.add_argument("--pred_file", default="pred.conllu", type=str, help="The filename to output the file")
parser.add_argument("--eval_file", default=None, type=str,
                    help="Evaluate the prediction and store it in the given filename")
parser.add_argument("--archive_latest", action="store_true", help="Archive the latest trained model")
parser.add_argument("--device", default=None, type=int, help="CUDA device number; set to -1 for CPU")
parser.add_argument("--lazy", action="store_true", help="Lazy load dataset")
parser.add_argument("--batch_size", default=1, type=int, help="The size of each prediction batch")
parser.add_argument("--sigmorphon", action="store_true", help="Use Sigmorphon evaluation instead of UD")

args = parser.parse_args()

import_submodules("udify")

overrides = {}
if args.device is not None:
    overrides["trainer"] = {"cuda_device": args.device}
if args.lazy:
    overrides["dataset_reader"] = {"lazy": args.lazy}
configs = [Params(overrides), Params.from_file(os.path.join(args.archive_dir, "config.json"))]
params = util.merge_configs(configs)

if args.archive_latest:
    archive_model(args.archive_dir)

pred_file = os.path.join(args.archive_dir, args.pred_file)
# pred_file = args.pred_file

if not args.eval_file:
    util.predict_model("udify_predictor", params, args.archive_dir, args.input_file, pred_file)
else:
    eval_file = os.path.join(args.archive_dir, args.eval_file)
    util.predict_and_evaluate_model("udify_predictor", params, args.archive_dir, args.input_file, pred_file,
                                    eval_file, batch_size=args.batch_size)
