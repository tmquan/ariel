#!/bin/bash
# Example training scripts for Ariel

echo "=========================================="
echo "Ariel Training Examples"
echo "=========================================="
echo ""

# Example 1: Basic training
echo "1. Basic training with default config"
echo "   Command: python code/main.py"
echo ""

# Example 2: Custom batch size and epochs
echo "2. Training with custom batch size and epochs"
echo "   Command: python code/main.py data.batch_size=8 training.max_epochs=150"
echo ""

# Example 3: Multi-GPU training
echo "3. Multi-GPU training"
echo "   Command: python code/main.py training.devices=4 training.strategy=ddp"
echo ""

# Example 4: Mixed precision training
echo "4. Mixed precision training (faster, less memory)"
echo "   Command: python code/main.py training.precision=16-mixed"
echo ""

# Example 5: Custom train/val split
echo "5. Custom train/validation split (90/10)"
echo "   Command: python code/main.py data.train_val_split=0.1"
echo ""

# Example 6: Larger model
echo "6. Training with larger model"
echo "   Command: python code/main.py model.net_config.init_filters=64 model.net_config.feature_dim=128"
echo ""

# Example 7: Fast development run
echo "7. Fast development run (test everything quickly)"
echo "   Command: python code/main.py training.fast_dev_run=True"
echo ""

# Example 8: Hyperparameter sweep
echo "8. Hyperparameter sweep over embedding dimensions"
echo "   Command: python code/main.py --multirun model.net_config.emb_dim=8,16,32"
echo ""

# Example 9: Resume from checkpoint
echo "9. Resume training from checkpoint"
echo "   Command: python code/main.py ckpt_path=checkpoints/last.ckpt"
echo ""

# Example 10: WandB logging
echo "10. Training with Weights & Biases logging"
echo "    Command: python code/main.py logger=wandb project_name=my-project"
echo ""

echo "=========================================="
echo "To run any example, copy the command"
echo "=========================================="

