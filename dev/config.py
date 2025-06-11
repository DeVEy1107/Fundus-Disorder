import argparse
import os
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for Fundus Disorder visual model")

    parser.add_argument('--data_dir', type=str, help='Directory containing dataset')
    parser.add_argument('--img_dir', type=str, help='Directory containing images')
    parser.add_argument('--train_json', type=str, help='Path to train split JSON')
    parser.add_argument('--val_json', type=str, help='Path to validation split JSON')
    parser.add_argument('--test_json', type=str, help='Path to test split JSON')
    parser.add_argument('--model_name', type=str, default='frcnn', help='Model architecture name')
    parser.add_argument('--num_classes', type=int, default=12, help='Number of classes (including background)')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of worker processes for data loading')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay for optimizer')
    parser.add_argument('--step_size', type=int, default=15, help='Step size for learning rate decay')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for learning rate decay')
    parser.add_argument('--img_size', type=int, nargs=2, default=[512, 512], help='Image size (height width)')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--print_freq', type=int, default=5, help='Print frequency during training')
    parser.add_argument('--save_freq', type=int, default=10, help='Checkpoint save frequency (in epochs)')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # Save arguments to a JSON file
    config_path = os.path.join(args.save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    return vars(args)

if __name__ == "__main__":
    args = parse_args()
    print("Parsed arguments:")
    for key, value in args.items():
        print(f"{key}: {value}")