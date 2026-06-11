# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import argparse
import copy
import csv
import json
import math
import os
import os.path as osp
import sys
import time
import types
import warnings


def _ensure_conda_libstdcpp_preloaded():
    """Re-exec once with conda libstdc++ preloaded to avoid SciPy ABI issues."""
    if os.environ.get('INTERNIMAGE_LIBSTDCXX_PRELOADED') == '1':
        return
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if not conda_prefix:
        return
    libstdcpp_path = osp.join(conda_prefix, 'lib', 'libstdc++.so.6')
    if not osp.exists(libstdcpp_path):
        return
    ld_preload = os.environ.get('LD_PRELOAD', '')
    preload_parts = [p for p in ld_preload.split(':') if p]
    if libstdcpp_path in preload_parts:
        return
    new_env = os.environ.copy()
    new_env['LD_PRELOAD'] = (
        f'{libstdcpp_path}:{ld_preload}' if ld_preload else libstdcpp_path)
    new_env['INTERNIMAGE_LIBSTDCXX_PRELOADED'] = '1'
    os.execvpe(sys.executable, [sys.executable] + sys.argv, new_env)


_ensure_conda_libstdcpp_preloaded()

import mmcv
import mmcv_custom  # noqa: F401,F403
import mmseg_custom  # noqa: F401,F403
import torch
import torch.distributed as dist
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         load_state_dict, wrap_fp16_model)
from mmcv.utils import Config, DictAction, get_git_hash
from mmseg import __version__
from mmseg.apis import (init_random_seed, multi_gpu_test, set_random_seed,
                        single_gpu_test, train_segmentor)
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import (collect_env, get_device, get_root_logger,
                         setup_multi_processes)
from blur_preprocessing import BlurPreprocessing, SparsifyRGB


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--load-from',
                        help='the checkpoint file to load weights from')
    parser.add_argument('--resume-from',
                        help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument('--gpu-id',
                            type=int,
                            default=0,
                            help='id of gpu to use '
                            '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff_seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
        'not be supported in version v0.22.0. Override some settings in the '
        'used config, the key-value pair in xxx=yyy format will be merged '
        'into config file. If the value to be overwritten is a list, it '
        'should be like key="[a,b]" or key=a,b It also allows nested '
        'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
        'marks are necessary and that no white space is allowed.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically.')
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='skip training and only run post-validation.')
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        default=None,
        help='checkpoint path for test-only mode or post-validation override.')
    parser.add_argument(
        '--skip-post-validation',
        action='store_true',
        help='skip automatic post-validation after training.')
    parser.add_argument(
        '--preprocessing',
        type=str,
        default=None,
        choices=['baseline', 'black_white', 'color_opponency', 'single_color'],
        help='input preprocessing type to apply before model forward.')
    parser.add_argument(
        '--sparsity',
        type=float,
        default=0.0,
        help='sparsity level to apply for input preprocessing, e.g. "0.7" or "70%".')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options. '
                      '--options will not be supported in version v0.22.0.')
        args.cfg_options = args.options

    return args


def _setup_log_hooks(cfg, logger):
    """Ensure text + tensorboard hooks are present for training logs."""
    if cfg.get('log_config') is None:
        cfg.log_config = dict(interval=50, hooks=[])

    hooks = cfg.log_config.get('hooks', [])
    if not isinstance(hooks, list):
        logger.warning('`cfg.log_config.hooks` is not a list, skip hook setup.')
        return

    hook_types = {hook.get('type') for hook in hooks if isinstance(hook, dict)}
    if 'TextLoggerHook' not in hook_types:
        hooks.insert(0, dict(type='TextLoggerHook', by_epoch=False))
        hook_types.add('TextLoggerHook')

    has_tb_backend = True
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    except Exception:
        has_tb_backend = False

    if not has_tb_backend:
        if 'TensorboardLoggerHook' in hook_types:
            hooks = [
                hook for hook in hooks
                if not (isinstance(hook, dict)
                        and hook.get('type') == 'TensorboardLoggerHook')
            ]
            logger.warning(
                'TensorBoard backend is unavailable; removed '
                'TensorboardLoggerHook from log_config.')
        else:
            logger.warning('TensorBoard backend not found; skip TensorBoard logging.')
    elif 'TensorboardLoggerHook' not in hook_types:
        hooks.append(dict(type='TensorboardLoggerHook', by_epoch=False))
        logger.info('Enabled TensorboardLoggerHook in log_config.')

    cfg.log_config.hooks = hooks


def _build_blur_preprocessor(preprocessing, work_dir, sparsity=0.0):
    if preprocessing == 'baseline':
        return SparsifyRGB(
            sparsity_threshold=sparsity,
            sparsity_type='percentage',
        )
    if preprocessing == 'color_opponency':
        return BlurPreprocessing(
            blur_bool=True,
            blur_depth=5,
            single_color=False,
            color_opponency=True,
            channels=3,
            path=osp.abspath(work_dir),
            training=False,
            black_white=False,
            normalize=False,
            sparsity_threshold=sparsity,
            sparsity_type='percentage',
            change_range=(0, 1))
    if preprocessing == 'black_white':
        return BlurPreprocessing(
            blur_bool=True,
            blur_depth=5,
            single_color=False,
            color_opponency=False,
            channels=3,
            path=osp.abspath(work_dir),
            training=False,
            black_white=True,
            normalize=False,
            sparsity_threshold=sparsity,
            sparsity_type='percentage',
            change_range=(0, 1))
    if preprocessing == 'single_color':
        return BlurPreprocessing(
            blur_bool=True,
            blur_depth=5,
            single_color=True,
            color_opponency=False,
            channels=3,
            path=osp.abspath(work_dir),
            training=False,
            black_white=False,
            normalize=False,
            sparsity_threshold=sparsity,
            sparsity_type='percentage',
            change_range=(0, 1))
    raise ValueError(f'Unsupported preprocessing mode: {preprocessing}')


def _run_blur_preprocessor(blur_preprocessor, img_tensor):
    blur_preprocessor = blur_preprocessor.to(device=img_tensor.device,
                                             dtype=img_tensor.dtype)
    processed = blur_preprocessor(img_tensor)
    if isinstance(processed, tuple):
        return processed[0]
    return processed


def _apply_preprocessing_to_img(blur_preprocessor, img):
    if blur_preprocessor is None:
        return img
    if torch.is_tensor(img):
        return _run_blur_preprocessor(blur_preprocessor, img)
    if isinstance(img, list):
        return [
            _run_blur_preprocessor(blur_preprocessor, item)
            if torch.is_tensor(item) else item for item in img
        ]
    if isinstance(img, tuple):
        return tuple(
            _run_blur_preprocessor(blur_preprocessor, item)
            if torch.is_tensor(item) else item for item in img)
    return img


def _attach_preprocessing_hook(model, cfg, logger, context, sparsity=0.0):
    preprocessing = str(cfg.get('preprocessing', 'baseline'))
    blur_preprocessor = _build_blur_preprocessor(preprocessing, cfg.work_dir, sparsity=sparsity)
    if blur_preprocessor is None:
        logger.info(f'Preprocessing [{context}]: baseline (disabled).')
        return

    object.__setattr__(model, '_original_forward_without_blur', model.forward)
    object.__setattr__(model, '_blur_preprocessor', blur_preprocessor)

    def _forward_with_blur(self, *args, **kwargs):
        if args:
            args = list(args)
            args[0] = _apply_preprocessing_to_img(self._blur_preprocessor, args[0])
            args = tuple(args)
        elif 'img' in kwargs:
            kwargs['img'] = _apply_preprocessing_to_img(self._blur_preprocessor,
                                                        kwargs['img'])
        return self._original_forward_without_blur(*args, **kwargs)

    model.forward = types.MethodType(_forward_with_blur, model)
    logger.info(f'Preprocessing [{context}]: enabled `{preprocessing}`.')


def _normalize_dataloader_cfg(cfg, logger):
    workers = int(cfg.data.get('workers_per_gpu', 0))
    if workers <= 0:
        if cfg.data.get('persistent_workers', None):
            logger.warning(
                'workers_per_gpu=0 detected; forcing `data.persistent_workers=False`.')
        cfg.data.persistent_workers = False


def _resolve_checkpoint_path(work_dir, checkpoint_path=None):
    """Find checkpoint for post-validation."""
    if checkpoint_path is not None:
        ckpt_path = osp.abspath(checkpoint_path)
        if not osp.isfile(ckpt_path):
            raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
        return ckpt_path

    preferred = ['best_mIoU.pth', 'latest.pth']
    for filename in preferred:
        candidate = osp.join(work_dir, filename)
        if osp.isfile(candidate):
            return candidate

    pth_files = [
        osp.join(work_dir, f) for f in os.listdir(work_dir) if f.endswith('.pth')
    ]
    if not pth_files:
        raise FileNotFoundError(
            f'No checkpoint found in work_dir: {osp.abspath(work_dir)}')
    pth_files.sort(key=lambda p: osp.getmtime(p), reverse=True)
    return pth_files[0]


def _default_post_validation(cfg):
    test_pipeline = copy.deepcopy(cfg.data.val.pipeline)
    return dict(
        metric='mIoU',
        acdc_subsets=[
            'acdc_rain_val', 'acdc_fog_val', 'acdc_night_val', 'acdc_snow_val'
        ],
        datasets=[
            dict(name='cityscapes_val', **copy.deepcopy(cfg.data.val)),
            dict(
                name='acdc_rain_val',
                type='CityscapesDataset',
                data_root='/ceph/sagnihot/datasets/ACDC',
                img_dir='rgb_anon/rain/val',
                ann_dir='gt/rain/val',
                img_suffix='_rgb_anon.png',
                seg_map_suffix='_gt_labelTrainIds.png',
                pipeline=copy.deepcopy(test_pipeline)),
            dict(
                name='acdc_fog_val',
                type='CityscapesDataset',
                data_root='/ceph/sagnihot/datasets/ACDC',
                img_dir='rgb_anon/fog/val',
                ann_dir='gt/fog/val',
                img_suffix='_rgb_anon.png',
                seg_map_suffix='_gt_labelTrainIds.png',
                pipeline=copy.deepcopy(test_pipeline)),
            dict(
                name='acdc_night_val',
                type='CityscapesDataset',
                data_root='/ceph/sagnihot/datasets/ACDC',
                img_dir='rgb_anon/night/val',
                ann_dir='gt/night/val',
                img_suffix='_rgb_anon.png',
                seg_map_suffix='_gt_labelTrainIds.png',
                pipeline=copy.deepcopy(test_pipeline)),
            dict(
                name='acdc_snow_val',
                type='CityscapesDataset',
                data_root='/ceph/sagnihot/datasets/ACDC',
                img_dir='rgb_anon/snow/val',
                ann_dir='gt/snow/val',
                img_suffix='_rgb_anon.png',
                seg_map_suffix='_gt_labelTrainIds.png',
                pipeline=copy.deepcopy(test_pipeline)),
            dict(
                name='darkzurich_val',
                type='CityscapesDataset',
                data_root='/ceph/sagnihot/datasets/DarkZurich',
                img_dir='rgb_anon/val/night',
                ann_dir='gt/val/night',
                img_suffix='_rgb_anon.png',
                seg_map_suffix='_gt_labelTrainIds.png',
                pipeline=copy.deepcopy(test_pipeline)),
        ])


def _get_post_validation(cfg):
    post_validation = copy.deepcopy(cfg.get('post_validation', {}))
    default_cfg = _default_post_validation(cfg)

    if 'metric' not in post_validation:
        post_validation['metric'] = default_cfg['metric']
    if 'datasets' not in post_validation:
        post_validation['datasets'] = default_cfg['datasets']
    if 'acdc_subsets' not in post_validation:
        post_validation['acdc_subsets'] = default_cfg['acdc_subsets']
    return post_validation


def _extract_miou(metric_dict):
    if 'mIoU' in metric_dict:
        return float(metric_dict['mIoU'])
    for key, value in metric_dict.items():
        if key.lower().endswith('miou'):
            return float(value)
    return float('nan')


def _extract_metric_value(metric_dict, metric_name):
    if metric_name in metric_dict:
        return float(metric_dict[metric_name])
    metric_name_lower = metric_name.lower()
    for key, value in metric_dict.items():
        if key.lower() == metric_name_lower:
            return float(value)
    return float('nan')


def _evaluate_on_datasets(cfg, checkpoint_path, distributed, logger, args=None):
    post_val = _get_post_validation(cfg)
    metric = post_val.get('metric', 'mIoU')
    dataset_cfgs = post_val.get('datasets', [])
    acdc_subsets = set(post_val.get('acdc_subsets', []))

    if not dataset_cfgs:
        logger.warning('No post-validation datasets configured, skipping.')
        return []

    eval_model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    eval_model.train_cfg = None
    _attach_preprocessing_hook(eval_model, cfg, logger, context='post-validation', sparsity=args.sparsity)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(eval_model)
    checkpoint = load_checkpoint(eval_model, checkpoint_path, map_location='cpu')

    if hasattr(eval_model, 'module'):
        load_state_dict(eval_model.module, checkpoint['state_dict'], strict=False)
    else:
        load_state_dict(eval_model, checkpoint['state_dict'], strict=False)

    if distributed:
        eval_model = MMDistributedDataParallel(
            eval_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
    else:
        eval_model = MMDataParallel(eval_model.cuda(), device_ids=[cfg.gpu_ids[0]])

    rank, _ = get_dist_info()
    rows = []
    workers_per_gpu = int(cfg.data.get('workers_per_gpu', 0))
    persistent_workers = bool(cfg.data.get('persistent_workers', False))
    if workers_per_gpu <= 0:
        persistent_workers = False

    for raw_dataset_cfg in dataset_cfgs:
        dataset_cfg = copy.deepcopy(raw_dataset_cfg)
        dataset_name = dataset_cfg.pop('name')
        logger.info(f'Post-validation on `{dataset_name}` started.')

        dataset = build_dataset(dataset_cfg)
        if hasattr(eval_model, 'module'):
            eval_model.module.CLASSES = dataset.CLASSES
            eval_model.module.PALETTE = dataset.PALETTE
        else:
            eval_model.CLASSES = dataset.CLASSES
            eval_model.PALETTE = dataset.PALETTE

        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=workers_per_gpu,
            persistent_workers=persistent_workers,
            dist=distributed,
            shuffle=False)

        torch.cuda.empty_cache()
        if not distributed:
            results = single_gpu_test(
                eval_model,
                data_loader,
                show=False,
                out_dir=None,
                efficient_test=False,
                opacity=0.5,
                pre_eval=True)
        else:
            tmpdir = osp.join(cfg.work_dir, '.post_eval_tmp')
            mmcv.mkdir_or_exist(tmpdir)
            results = multi_gpu_test(
                eval_model,
                data_loader,
                tmpdir=tmpdir,
                gpu_collect=False,
                efficient_test=False,
                pre_eval=True)

        if rank == 0:
            metric_dict = dataset.evaluate(results, metric=metric)
            miou = _extract_miou(metric_dict)
            macc = _extract_metric_value(metric_dict, 'mAcc')
            aacc = _extract_metric_value(metric_dict, 'aAcc')
            logger.info(
                f'Post-validation `{dataset_name}`: {metric}={miou:.6f}')
            rows.append(
                dict(
                    dataset=dataset_name,
                    metric=metric,
                    mIoU=miou,
                    mAcc=macc,
                    aAcc=aacc,
                    num_samples=len(dataset),
                    details_json=json.dumps(metric_dict, sort_keys=True)))

        if distributed:
            dist.barrier()

    if rank == 0:
        acdc_rows = [
            row for row in rows
            if row['dataset'] in acdc_subsets and not math.isnan(row['mIoU'])
        ]
        acdc_scores = [row['mIoU'] for row in acdc_rows]
        if acdc_scores:
            acdc_mean = sum(acdc_scores) / len(acdc_scores)
            acdc_macc_scores = [
                row['mAcc'] for row in acdc_rows
                if 'mAcc' in row and not math.isnan(row['mAcc'])
            ]
            acdc_aacc_scores = [
                row['aAcc'] for row in acdc_rows
                if 'aAcc' in row and not math.isnan(row['aAcc'])
            ]
            acdc_macc_mean = (
                sum(acdc_macc_scores) / len(acdc_macc_scores)
                if acdc_macc_scores else float('nan'))
            acdc_aacc_mean = (
                sum(acdc_aacc_scores) / len(acdc_aacc_scores)
                if acdc_aacc_scores else float('nan'))
            rows.append(
                dict(
                    dataset='acdc_4_mean',
                    metric=metric,
                    mIoU=acdc_mean,
                    mAcc=acdc_macc_mean,
                    aAcc=acdc_aacc_mean,
                    num_samples=len(acdc_scores),
                    details_json=json.dumps(
                        {row['dataset']: row['mIoU']
                         for row in sorted(acdc_rows, key=lambda item: item['dataset'])},
                        sort_keys=True)))
            logger.info(f'Post-validation `acdc_4_mean`: {metric}={acdc_mean:.6f}')

    return rows


def _write_post_validation_csv(cfg, checkpoint_path, rows):
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    csv_path = osp.join(cfg.work_dir, 'post_validation_results.csv')
    file_exists = osp.isfile(csv_path)
    seed = cfg.get('seed', None)
    preprocessing = str(cfg.get('preprocessing', 'baseline'))
    sparsity = str(cfg.get('sparsity', '0'))
    fieldnames = [
        'timestamp', 'config', 'checkpoint', 'dataset', 'metric', 'mIoU',
        'mAcc', 'aAcc', 'seed', 'preprocessing', 'sparsity', 'num_samples',
        'details_json'
    ]
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                dict(
                    timestamp=timestamp,
                    config=cfg.filename,
                    checkpoint=checkpoint_path,
                    dataset=row.get('dataset'),
                    metric=row.get('metric'),
                    mIoU=row.get('mIoU', float('nan')),
                    mAcc=row.get('mAcc', float('nan')),
                    aAcc=row.get('aAcc', float('nan')),
                    seed=seed,
                    preprocessing=preprocessing,
                    sparsity=sparsity,
                    num_samples=row.get('num_samples'),
                    details_json=row.get('details_json', '{}')))
    return csv_path


def _safe_cfg_text(cfg):
    try:
        return cfg.pretty_text
    except TypeError:
        if hasattr(cfg, 'text'):
            return cfg.text
        return json.dumps(cfg._cfg_dict.to_dict(), indent=2, sort_keys=True)


def _safe_dump_cfg(cfg, out_path):
    try:
        cfg.dump(out_path)
    except TypeError:
        with open(out_path, 'w') as f:
            f.write(_safe_cfg_text(cfg))


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]
    if args.preprocessing is not None:
        cfg.preprocessing = args.preprocessing
    elif cfg.get('preprocessing', None) is None:
        cfg.preprocessing = 'baseline'

    cfg.auto_resume = args.auto_resume

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # gpu_ids is used to calculate iter when resuming checkpoint
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    _safe_dump_cfg(cfg, osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # set multi-process settings
    setup_multi_processes(cfg)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    cfg_text = _safe_cfg_text(cfg)
    logger.info(f'Config:\n{cfg_text}')
    _setup_log_hooks(cfg, logger)
    _normalize_dataloader_cfg(cfg, logger)

    # set random seeds
    cfg.device = get_device()
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    if args.test_only:
        with torch.inference_mode():
            if args.checkpoint_path is None:
                raise ValueError('`--test-only` requires `--checkpoint-path`.')
            if args.skip_post_validation:
                logger.info('`--skip-post-validation` set in test-only mode, exiting.')
                return
            checkpoint_path = _resolve_checkpoint_path(cfg.work_dir,
                                                    args.checkpoint_path)
            logger.info(
                f'Running test-only post-validation with checkpoint: {checkpoint_path}')
            rows = _evaluate_on_datasets(cfg, checkpoint_path, distributed, logger, args=args)
            rank, _ = get_dist_info()
            if rank == 0 and rows:
                csv_path = _write_post_validation_csv(cfg, checkpoint_path, rows)
                logger.info(f'Post-validation CSV saved to: {csv_path}')
            return

    model = build_segmentor(cfg.model,
                            train_cfg=cfg.get('train_cfg'),
                            test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    _attach_preprocessing_hook(model, cfg, logger, context='training', sparsity=args.sparsity)

    # SyncBN is not support for DP
    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        model = revert_sync_batchnorm(model)

    # logger.info(model)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    # passing checkpoint meta for saving best checkpoint
    if cfg.checkpoint_config is not None and cfg.checkpoint_config.get('meta'):
        meta.update(cfg.checkpoint_config.meta)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_segmentor(model,
                    datasets,
                    cfg,
                    distributed=distributed,
                    validate=(not args.no_validate),
                    timestamp=timestamp,
                    meta=meta)

    if args.skip_post_validation:
        logger.info('Post-validation skipped due to `--skip-post-validation`.')
        return

    checkpoint_path = _resolve_checkpoint_path(cfg.work_dir, args.checkpoint_path)
    logger.info(f'Running post-validation with checkpoint: {checkpoint_path}')
    rows = _evaluate_on_datasets(cfg, checkpoint_path, distributed, logger, args=args)
    rank, _ = get_dist_info()
    if rank == 0 and rows:
        csv_path = _write_post_validation_csv(cfg, checkpoint_path, rows)
        logger.info(f'Post-validation CSV saved to: {csv_path}')


if __name__ == '__main__':
    main()
