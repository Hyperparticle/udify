"""
Creates vocab files for all treebanks in the given directory
"""

import os
import json
import logging
import argparse

from allennlp.commands.make_vocab import make_vocab_from_params
from allennlp.common import Params
from allennlp.common.util import import_submodules

from udify import util

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", default="data/ud-treebanks-v2.3", type=str,
                    help="The path containing all UD treebanks")
parser.add_argument("--output_dir", default="data/vocab", type=str, help="The path to save all vocabulary files")
parser.add_argument("--treebanks", default=[], type=str, nargs="+",
                    help="Specify a list of treebanks to use; leave blank to default to all treebanks available")
parser.add_argument("--params_file", default=None, type=str, help="The path to the vocab params")
args = parser.parse_args()

import_submodules("udify")

params_file = util.VOCAB_CONFIG_PATH if not args.params_file else args.params_file

treebanks = sorted(util.get_ud_treebank_files(args.dataset_dir, args.treebanks).items())
for treebank, (train_file, dev_file, test_file) in treebanks:
    logger.info(f"Creating vocabulary for treebank {treebank}")

    if not train_file:
        logger.info(f"No training data for {treebank}, skipping")
        continue

    overrides = json.dumps({
        "train_data_path": train_file,
        "validation_data_path": dev_file,
        "test_data_path": test_file
    })
    params = Params.from_file(params_file, overrides)
    output_file = os.path.join(args.output_dir, treebank)

    make_vocab_from_params(params, output_file)
