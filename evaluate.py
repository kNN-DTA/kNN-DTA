#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from argparse import Namespace
from itertools import chain
from typing import final

import torch
from omegaconf import DictConfig

import numpy as np
import pandas as pd
# import swalign
import math
import datetime

from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import metrics, progress_bar
from fairseq.utils import reset_logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
# Get a logger with specified name
logger = logging.getLogger("fairseq_cli.validate")
# the embedding search space is twice as large as the raw similarity calculation space

def add_custom_arguments(parser):
    # parser.add_argument('--arch', default='roberta_dti_mlm_regress',
    #                     help='The model architecture')

    parser.add_argument('--T', type=float, metavar='T', default=1000,
                        help='Temperature T for softmax on paired cls embedding distance')

    parser.add_argument('--T-0', type=float, metavar='T', default=1000,
                        help='Temperature T for softmax on molecule cls embedding distance')
    
    parser.add_argument('--T-1', type=float, metavar='T', default=1000,
                        help='Temperature T for softmax on protein cls embedding distance')

    parser.add_argument('--k', type=int, metavar='k', default=16,
                        help='k nearest neighbors for paired cls embedding')

    parser.add_argument('--k-0', type=int, metavar='k', default=16,
                        help='k nearest neighbors for molecule cls embedding')
    
    parser.add_argument('--k-1', type=int, metavar='k', default=16,
                        help='k nearest neighbors for protein cls embedding')

    parser.add_argument('--l', type=float, metavar='l', default=0.8,
                        help='The prediction weight')

    parser.add_argument('--l-update', type=float, metavar='l', default=1,
                        help='The weight used to update prediction')
    
    parser.add_argument('--knn-embedding-weight-0', type=float, metavar='l', default=0.8,
                        help='The knn embedding weight') 

    parser.add_argument('--knn-embedding-weight-1', type=float, metavar='l', default=0.8,
                        help='The knn embedding weight')   

    parser.add_argument('--alpha', type=float, metavar='a', default=0.707,
                    help='Alpha for embedding-wise search')      

#################################################################################################
    parser.add_argument('--sim', type=str, default='L2', choices=['L2', 'cosine', 'attn', 'dot'],
                        help='The similarity metric for search. Note that --sim attn is used with use-attn-cal at the same time.')

    parser.add_argument('--label-use-attn-cal', action='store_true',
                        help='Use attention calculation when doing label-wise search')

    parser.add_argument('--embedding-use-attn-cal', action='store_true',
                        help='Use attention calculation when doing embedding-wise search')

    parser.add_argument('--label-use-mean-cal', action='store_true',
                    help='Use mean calculation when doing label-wise search')

    parser.add_argument('--embedding-use-mean-cal', action='store_true',
                        help='Use mean calculation when doing embedding-wise search')

    parser.add_argument('--result-file-path', type=str, default='tmp.tsv',
                        help='Where to save the result tsv file')
    
    return parser



def main(cfg: DictConfig, override_args=None):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)
   
    utils.import_user_module(cfg.common)

    reset_logging()

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    if cfg.distributed_training.distributed_world_size > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
        data_parallel_rank = distributed_utils.get_data_parallel_rank()
    else:
        data_parallel_world_size = 1
        data_parallel_rank = 0

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [cfg.common_eval.path],
        arg_overrides=overrides,
        suffix=cfg.checkpoint.checkpoint_suffix,
    )
    model = models[0]

    # Move models to GPU
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Print args
    logger.info(saved_cfg)

    # Build criterion
    criterion = task.build_criterion(saved_cfg.criterion)
    criterion.eval()

    for subset in cfg.dataset.valid_subset.split(","):
        try:
            task.load_dataset(subset, combine=False, epoch=1, task_cfg=saved_cfg.task)
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)

        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=cfg.dataset.max_tokens,
            max_sentences=cfg.dataset.batch_size,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                *[m.max_positions() for m in models],
            ),
            ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
            seed=cfg.common.seed,
            num_shards=data_parallel_world_size,
            shard_id=data_parallel_rank,
            num_workers=cfg.dataset.num_workers,
            data_buffer_size=cfg.dataset.data_buffer_size,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            prefix=f"valid on '{subset}' subset",
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        )

        log_outputs = []

        id_tensor_list = []
        prediction_tensor_list = []
        target_tensor_list = []

        # Iterate over the 'subset' dataset
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            # _loss, _sample_size, log_output = task.valid_step(sample, model, criterion)
            model.eval()
            with torch.no_grad():
                _sample_size, log_output = criterion(model, sample)
            
            
            final_prediction = log_output['prediction'].squeeze()
            target = log_output['target']
            log_output_tmp = {'final_prediction': final_prediction, 'target': target, 'sample_size': log_output['sample_size'], 'ntokens': log_output['ntokens'], 'nsentences': log_output['nsentences']}
            progress.log(log_output_tmp, step=i)
            log_outputs.append(log_output_tmp)

            id_tensor_list.append(sample['id'].detach().cpu().numpy())
            prediction_tensor_list.append(final_prediction.detach().cpu().numpy())
            target_tensor_list.append(target.detach().cpu().numpy())

            logger.info(f'batch:{i}')

        if data_parallel_world_size > 1:
            log_outputs = distributed_utils.all_gather_list(
                log_outputs,
                max_size=cfg.common.all_gather_list_size,
                group=distributed_utils.get_data_parallel_group(),
            )
            log_outputs = list(chain.from_iterable(log_outputs))

        with metrics.aggregate() as agg:
            task.reduce_metrics(log_outputs, criterion)
            log_output = agg.get_smoothed_values()

        progress.print(log_output, tag=subset, step=i)

        id_list, prediction_list, target_list = [], [], []
        for i in id_tensor_list:
            if i.ndim == 1:
                for j in i:
                    id_list.append(j)
            else: # i.ndim == 0
                id_list.append(i)

        for i in prediction_tensor_list:
            if i.ndim == 1:
                for j in i:
                    prediction_list.append(j)
            else: # i.ndim == 0
                prediction_list.append(i)

        for i in target_tensor_list:
            if i.ndim == 1:
                for j in i:
                    target_list.append(j)
            else: # i.ndim == 0
                target_list.append(i)

        df = pd.DataFrame({'prediction': prediction_list, 'target': target_list}, index=np.array(id_list))
        # 调整 training set 本身的顺序（原本为 fairseq 随机循环 batch 随机的顺序）
        df.sort_index(inplace=True)
        
        df.to_csv(f'{cfg.criterion.result_file_path}', index=False, sep='\t')

        logger.info(f"{cfg.dataset.valid_subset} done")

def cli_main():
    parser = options.get_validation_parser()
    parser = add_custom_arguments(parser)
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_validation_parser()
    override_parser = add_custom_arguments(override_parser)
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    distributed_utils.call_main(
        convert_namespace_to_omegaconf(args), main, override_args=override_args
    )


if __name__ == "__main__":
    cli_main()
