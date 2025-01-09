#%%
import wandb, random; from datetime import datetime 
from pathlib import Path
wandb_dir = Path(__file__).resolve().parent / "wandb_logs"
wandb_dir.mkdir(parents=True, exist_ok=True)
#%%
epochs = 140; lr = 0.01
now = datetime.now()
formatted_time = now.strftime("%Y_%m_%d_%H_%M")

run = wandb.init(
    project="MY_FIRST_W&B_PROJECT",
    name=formatted_time,
    dir=wandb_dir,
    config={     # Track hyperparameters and run metadata
        "learning_rate": lr,
        "epochs": epochs,
    },
)
offset = random.random() / 5
print(f"lr: {lr}")
for epoch in range(2, epochs):
    acc = 1 - 2** - epoch - random.random() / epoch - offset
    loss = 2** - epoch + random.random() / epoch + offset
    print(f"Epoch {epoch}: loss={loss} acc={acc}")
    wandb.log({"epoch": epoch, "loss": loss, "metric_name": acc}, step=epoch)