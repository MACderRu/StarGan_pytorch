import argparse

from src.common_utils.config import Config


if __name__ == '__main__':
    parser = argparse.ArgumentParser("StarGan pytorch")
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--wandb-run', type=str, default='default_run_name')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    print('Setup...')
    print(f'device: {args.device}')

    config = Config.from_file(args.config)

    train_model(config)

    print('Training is ended')
    print(f'Checkpoints are saved in {config.checkpoints.save_path}')
