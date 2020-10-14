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
from tensorflow.compat.v1 import gfile
from collections import Counter
import numpy as np
import json
import sys

from metrics.bleu import compute_bleu
from infer_utils import evaluate as evaluate_infer

from tqdm import tqdm
import tensorflow.compat.v1 as tf

FLAGS = None

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
      '--infer_metrics',
      type=str,
      default='bleu:brief',
      help='For infer task, choose one of multiple metric in (bleu:all|rouge:all|kl:all) or (bleu:brief|kl:brief),'
      ' this will give you a single number metric. (bleu|kl) is equivalent to (belu:brief|kl:brief) ')
  parser.add_argument(
      '--output',
      type=str,
      default='score.json',
      help='output path for score json.')

def score_inference(flags):
  assert flags.true_data and flags.pred_data
  expanded_true_data = expanduser(flags.true_data)
  expanded_pred_data = expanduser(flags.pred_data)

  infer_metrics = flags.infer_metrics.split(',')
  results = {}

  for metric in infer_metrics:
      infer_result = evaluate_infer(expanded_true_data, expanded_pred_data, metric)
      metric = metric.split(":")[0]
      print('infer ', metric, ': ', infer_result)
      results[metric] = infer_result
  return results


def main(flags):
  if flags.task == 'infer':
    score = score_inference(flags)
  else:
    score = score_inference(flags)

  with tf.gfile.GFile(flags.output, 'w') as f:
    f.write(json.dumps(score))


def run_main(unused):
  main(FLAGS)


if __name__ == '__main__':
  this_parser = argparse.ArgumentParser()
  add_arguments(this_parser)
  FLAGS, unparsed = this_parser.parse_known_args()
  tf.app.run(main=run_main, argv=[sys.argv[0]] + unparsed)
