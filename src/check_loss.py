
import wandb

wandb.login(key='4a2b30e83a65bab71d5963b2392b02b499febd7b')
run = wandb.Api().run("/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/code/Difix3D/Difix3D/outputs/difix_80001/train_80001/wandb/offline-run-20250715_074954-mwvx6gsw/run-mwvx6gsw.wandb")
print(run.config)  # 查看配置
print(run.history())  # 查看指标