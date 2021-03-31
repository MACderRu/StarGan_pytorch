import sys
import argparse
import wandb
from pathlib import Path

from src.common_utils.config import Config
from src.common_utils.utils import load_checkpoint, terminate_launch
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

    if args.logging:
        wandb.init(project=config.wandb.project_name, name=args.wandb_run, config=config.training)

    ckpt = None

    if args.resume:
        try:
            ckpt_path = Path(config.checkpoints.save_path) / "last_ckpt.pth"
            ckpt = load_checkpoint(ckpt_path)
        except FileNotFoundError as e:
            msg = f"last checkpoint not found {e}"
            terminate_launch(msg, args.logging)

    if args.dataset not in config.data:
        msg = f"Such dataset is not available: {args.dataset}"
        terminate_launch(msg, args.logging)

    config.device = args.device
    train_model(config, ckpt)

    if args.logging:
        wandb.finish()

    print("Training is ended")
    print(f"Checkpoints are saved in {config.checkpoints.save_path}")
