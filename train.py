import os
import sys
import argparse
import wandb
from pathlib import Path
from shutil import rmtree

from src.common_utils.config import Config
from src.common_utils.utils import load_checkpoint, terminate_launch, find_last_run
from src.star_gan.model_utils import train_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser("StarGan pytorch")
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--logging', action='store_true')
    parser.add_argument('--wandb-run', type=str, default='default_name run')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dataset', type=str, default='celeba')
    args = parser.parse_args()

    print("Setup...")
    print(f"Device: '{args.device}'")

    config = Config.from_file(args.config)

    if not os.path.exists(config.checkpoints.save_path):
        os.mkdir(config.checkpoints.save_path)

    if args.logging:
        wandb.init(project=config.wandb.project_name, name=args.wandb_run, config=config.training)

    ckpt = None
    if args.resume:
        try:
            lst_run = find_last_run(config.checkpoints.save_path)
            last_run_last_ckpt_p = Path(config.checkpoints.save_path) / f"run{lst_run}/last.pth"
            ckpt = load_checkpoint(last_run_last_ckpt_p)
        except FileNotFoundError as e:
            msg = f"last checkpoint not found {e}"
            terminate_launch(msg, args.logging)

    if args.dataset not in config.data:
        msg = f"Such dataset is not available: {args.dataset}"
        terminate_launch(msg, args.logging)

    if not args.logging:
        config.wandb.log_step = None

    config.device = args.device
    train_model(config, ckpt)

    if args.logging:
        wandb.finish()

    print("Training is ended")
    print(f"Checkpoints are saved in {config.checkpoints.save_path}")
