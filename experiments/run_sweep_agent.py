import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import wandb
from experiments.train import train

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_sweep_agent.py <sweep-id> [count]")
        sys.exit(1)
    sweep_id = sys.argv[1]
    if len(sys.argv) > 2:
        count = int(sys.argv[2])
    else:
        count = 10
    wandb.agent(sweep_id, function=train, count=count)
