# coding=utf-8
## Alex Shah
## Purpose: To define a new problem set in tensor2tensor

"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

import tensorflow as tf

FLAGS = tf.flags.FLAGS

# End-of-sentence marker.
EOS = text_encoder.EOS_ID

@registry.register_problem
class TranslateEndeWmtBpe32k(translate.TranslateProblem):
  """Problem spec for Shah EnEs Translator."""

  @property
  def targeted_vocab_size(self):
    return 15000

  @property
  def vocab_name(self):
    return "vocab.shahenes"

  def feature_encoders(self, data_dir):
    vocab_filename = os.path.join(data_dir, self.vocab_file)
    encoder = text_encoder.TokenTextEncoder(vocab_filename, replace_oov="UNK")
    return {"inputs": encoder, "targets": encoder}

  def generator(self, data_dir, tmp_dir, train):
    """Token generator Shah En Es Translator."""
    dataset_path = ("es.txt"
                    if train else "es.txt")
    train_path = _get_wmt_ende_bpe_dataset(tmp_dir, dataset_path)
    token_tmp_path = os.path.join(tmp_dir, self.vocab_file)
    token_path = os.path.join(data_dir, self.vocab_file)
    tf.gfile.Copy(token_tmp_path, token_path, overwrite=True)
    with tf.gfile.GFile(token_path, mode="a") as f:
      f.write("UNK\n")  # Add UNK to the vocab.
    token_vocab = text_encoder.TokenTextEncoder(token_path, replace_oov="UNK")
    return translate.token_generator(train_path + ".en", train_path + ".es",
                                     token_vocab, EOS)

  @property
  def input_space_id(self):
    return problem.SpaceID.SHAH_EN_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.SHAH_ES_TOK

