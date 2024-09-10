# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import math
import os
from functools import partial
import re

# from fvcore.common.checkpoint import PeriodicCheckpointer
import torch

from data.datasets import ImageNet
from utils.config import setup


torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--no-resume", action="store_true", help="Whether to not attempt to resume from the checkpoint directory. ")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )

    return parser

def main(args):
    cfg = setup(args)
    dataset_str = cfg.train.dataset_path
    dataset_str = dataset_str.replace('=C:', '=C;')
    # print(dataset_str)
    tokens = dataset_str.split(":")
    for i in range(len(tokens)):
        tokens[i] = tokens[i].replace(';',':')
    print(tokens)
    # exit()
    name = tokens[0]
    kwargs = {}
    for token in tokens[1:]:
        key, value = token.split("=")
        assert key in ("root", "extra", "split")
        kwargs[key] = value

    for split in ImageNet.Split:
        dataset = ImageNet(split=split, root=kwargs['root'], extra=kwargs['extra'])
        dataset.dump_extra()


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)