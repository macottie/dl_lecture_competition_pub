import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
import wandb
from termcolor import cprint
from tqdm import tqdm
from datetime import datetime
from scipy.signal import butter, filtfilt
from torch.optim.lr_scheduler import OneCycleLR

from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier
from src.utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")

# EarlyStopping クラス定義
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): 精度が改善されないエポック数。この数を超えると訓練が停止されます。
            verbose (bool): 早期終了のメッセージを出力するかどうか。
            delta (float): 前のベストスコアと新しいスコアの最小の差異。この差異を超えた場合のみ改善とみなします。
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model_state_dict):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model_state_dict)
        elif score < self.best_score - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model_state_dict)
            self.counter = 0

    def save_checkpoint(self, val_loss, model_state_dict):
        ''' ベストモデルを保存 '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model...')
        torch.save(model_state_dict, 'checkpoint.pt')
        self.val_loss_min = val_loss

def run(args: DictConfig):
    # シード値の設定
    set_seed(args.seed)
    # 実行日時に基づいたログディレクトリの作成
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join("/content/drive/My Drive/dl_lecture_competition_pub/outputs", current_time)
    os.makedirs(logdir, exist_ok=True)

    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}

    train_set = ThingsMEGDataset("train", "/content/drive/My Drive/dl_lecture_competition_pub/data")
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset("val", "/content/drive/My Drive/dl_lecture_competition_pub/data")
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset("test", "/content/drive/My Drive/dl_lecture_competition_pub/data")
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ------------------
    #       Model
    # ------------------
    model = BasicConvClassifier(
        train_set.num_classes, train_set.seq_len, train_set.num_channels
    ).to(args.device)

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)  # L2正則化係数を1e-5に設定
    scheduler = OneCycleLR(optimizer,
                        max_lr=args.lr * 3,
                        epochs=args.epochs,
                        steps_per_epoch=len(train_loader))


    # ------------------
    #   Start training
    # ------------------
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)

    # EarlyStoppingインスタンスの作成
    early_stopping = EarlyStopping(patience=5, verbose=True)

    step_count=0

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_loss, train_acc, val_loss, val_acc = [], [], [], []

        model.train()
        for X, y, _ in tqdm(train_loader, desc="Train"):
            X, y = X.to(args.device), y.to(args.device)
            step_count += 1
            y_pred = model(X)
            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

            
        model.eval()

        for X, y, _ in tqdm(val_loader, desc="Validation"):
            X, y = X.to(args.device), y.to(args.device)

            with torch.no_grad():
                y_pred = model(X)

            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({
                "train_loss": np.mean(train_loss),
                "train_acc": np.mean(train_acc),
                "val_loss": np.mean(val_loss),
                "val_acc": np.mean(val_acc),
                "learning_rate": scheduler.get_last_lr()[0],
            })

        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)

        # 早期終了のチェック
        early_stopping(np.mean(val_loss), model.state_dict())

        if early_stopping.early_stop:
            print("早期終了")
            break


    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = []
    model.eval()
    for X, _ in tqdm(test_loader, desc="Validation"):
        X = X.to(args.device)
        with torch.no_grad():
            preds.append(model(X).detach().cpu())

    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run(config)