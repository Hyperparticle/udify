"""
Extracts a BERT archive from an existing model
"""

import logging
import argparse

from udify import util

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("archive_dir", type=str, help="The directory where model.tar.gz resides")
parser.add_argument("--output_path", default=None, type=str, help="The path to output the file")

args = parser.parse_args()

bert_config = "config/archive/bert-base-multilingual-cased/bert_config.json"
util.archive_bert_model(args.archive_dir, bert_config, args.output_path)
