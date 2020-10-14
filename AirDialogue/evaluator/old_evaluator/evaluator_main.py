# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module builds a falsk server for visualization."""

import argparse
from os.path import expanduser
from tensorflow import gfile
from collections import Counter
import numpy as np
import json

from infer_utils import evaluate as evaluate_infer

from tqdm import tqdm
import tensorflow as tf


def add_arguments(parser):
  """Build ArgumentParser."""
  parser.add_argument(
      '--true_data', type=str, default='', help='Path to the true data file.')
  parser.add_argument(
      '--true_kb', type=str, default='', help='Path to the kb file.')
  parser.add_argument(
      '--pred_data', type=str, default='', help='Path to the prediction file.')
  parser.add_argument(
      '--task',
      type=str,
      default='infer',
      help='type of the task, one of |human|infer|selfplay|')
  parser.add_argument(
      '--output',
      type=str,
      default='score.json',
      help='output path for score json.')

def score_inference(flags):
  assert flags.true_data and flags.pred_data
  expanded_true_data = expanduser(flags.true_data)
  expanded_pred_data = expanduser(flags.pred_data)
  infer_bleu = evaluate_infer(expanded_true_data, expanded_pred_data, 'bleu')
  print 'infer bleu: ', infer_bleu
  return {'bleu': infer_bleu}


def main(flags):
  if flags.task == 'infer':
    score = score_inference(flags)
  else:
    score = score_inference(flags)

  with tf.gfile.GFile(flags.output, 'w') as f:
    f.write(json.dumps(score))


if __name__ == '__main__':
  this_parser = argparse.ArgumentParser()
  add_arguments(this_parser)
  FLAGS, unparsed = this_parser.parse_known_args()
  main(FLAGS)
