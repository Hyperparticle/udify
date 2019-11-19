"""
Predict conllu files given a trained model
"""

import os
import shutil
import logging
import argparse
import tarfile
from pathlib import Path

from allennlp.common import Params
from allennlp.common.util import import_submodules
from allennlp.models.archival import archive_model

from udify import util

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("archive", type=str, help="The archive file")
parser.add_argument("input_file", type=str, help="The input file to predict")
parser.add_argument("pred_file", type=str, help="The output prediction file")
parser.add_argument("--eval_file", default=None, type=str,
                    help="If set, evaluate the prediction and store it in the given file")
parser.add_argument("--device", default=0, type=int, help="CUDA device number; set to -1 for CPU")
parser.add_argument("--batch_size", default=1, type=int, help="The size of each prediction batch")
parser.add_argument("--lazy", action="store_true", help="Lazy load dataset")
parser.add_argument("--raw_text", action="store_true", help="Input raw sentences, one per line in the input file.")

args = parser.parse_args()

import_submodules("udify")

archive_dir = Path(args.archive).resolve().parent

if not os.path.isfile(archive_dir / "weights.th"):
    with tarfile.open(args.archive) as tar:
        tar.extractall(archive_dir)

config_file = archive_dir / "config.json"

overrides = {}
if args.device is not None:
    overrides["trainer"] = {"cuda_device": args.device}
if args.lazy:
    overrides["dataset_reader"] = {"lazy": args.lazy}
configs = [Params(overrides), Params.from_file(config_file)]
params = util.merge_configs(configs)

predictor = "udify_predictor" if not args.raw_text else "udify_text_predictor"

if not args.eval_file:
    util.predict_model_with_archive(predictor, params, archive_dir, args.input_file, args.pred_file,
                                    batch_size=args.batch_size)
else:
    util.predict_and_evaluate_model_with_archive(predictor, params, archive_dir, args.input_file,
                                                 args.pred_file, args.eval_file, batch_size=args.batch_size)
