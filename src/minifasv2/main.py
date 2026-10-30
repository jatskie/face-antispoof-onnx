"""Training loop for MultiFTNet: classification + Fourier-spectrum auxiliary loss."""

import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
from torch import optim
from torch.nn import CrossEntropyLoss, MSELoss

from src.minifasv2.data import get_train_valid
from src.minifasv2.model import MultiFTNet, load_weights


class Trainer:
    def __init__(self, config, resume_from=None, transfer_learning=False):
        self.config = config
        self.step = 0
        self.validation_step = 0
        self.start_epoch = 0
        self.best_validation_accuracy = 0.0
        self.resume_from = resume_from
        self.transfer_learning = transfer_learning
        self.train_loader, self.valid_loader = get_train_valid(config)
        self.log_train_every = max(1, len(self.train_loader) // config.log_loss_per_epoch)
        self.log_valid_every = max(1, len(self.valid_loader) // config.log_loss_per_epoch)
        self.mid_epoch_save_every = max(100, len(self.train_loader) // 4)

    def train_model(self):
        self._init_model_param()
        if self.resume_from and os.path.exists(self.resume_from):
            if self.transfer_learning:
                print(f"Transfer learning from {self.resume_from} (fresh optimizer)")
                load_weights(self.model.module, self.resume_from, include_ft=False)
            else:
                self._load_checkpoint(self.resume_from)
        self._train_stage()

    def _init_model_param(self):
        self.classification_criterion = CrossEntropyLoss()
        self.fourier_criterion = MSELoss()
        model = MultiFTNet(
            num_classes=self.config.num_classes, conv6_kernel=self.config.kernel_size
        )
        self.model = torch.nn.DataParallel(model.to(self.config.device))
        self.optimizer = optim.SGD(
            self.model.module.parameters(),
            lr=self.config.lr,
            weight_decay=5e-4,
            momentum=self.config.momentum,
        )
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.config.milestones, self.config.gamma
        )

    def _train_stage(self):
        self.writer = SummaryWriter(self.config.log_path)
        print(
            f"Training epochs {self.start_epoch + 1}-{self.config.epochs} | "
            f"lr {self.config.lr} | batch {self.config.batch_size} | "
            f"milestones {self.config.milestones}"
        )
        for epoch in range(self.start_epoch, self.config.epochs):
            start = time.time()
            train_metrics = self._train_epoch(epoch)
            self.lr_scheduler.step()
            valid_metrics, per_class_accuracy = self._validate()

            is_best = valid_metrics["validation_accuracy"] > self.best_validation_accuracy
            if is_best:
                self.best_validation_accuracy = valid_metrics["validation_accuracy"]

            print(
                f"\nEpoch {epoch + 1}/{self.config.epochs} ({time.time() - start:.0f}s) | "
                f"train loss {train_metrics['train_loss']:.4f} "
                f"acc {train_metrics['train_accuracy']:.2%} | "
                f"valid loss {valid_metrics['validation_loss']:.4f} "
                f"acc {valid_metrics['validation_accuracy']:.2%} "
                f"(real {per_class_accuracy[0]:.2%}, spoof {per_class_accuracy[1]:.2%})"
                + (" | new best" if is_best else "")
            )
            self._save_checkpoint(
                epoch + 1, {**train_metrics, **valid_metrics}, epoch_end=True, is_best=is_best
            )
        self.writer.close()

    def _train_epoch(self, epoch):
        self.model.train()
        totals = np.zeros(4)  # loss, accuracy, classification loss, fourier loss
        for batch_idx, (sample, fourier_target, labels) in enumerate(self.train_loader):
            totals += self._train_batch(sample, fourier_target, labels)
            self.step += 1
            # .tolist() -> plain floats, so checkpoints stay loadable with weights_only=True
            loss, accuracy, cls_loss, ft_loss = (totals / (batch_idx + 1)).tolist()
            metrics = {
                "train_loss": loss,
                "train_accuracy": accuracy,
                "train_classification_loss": cls_loss,
                "train_fourier_loss": ft_loss,
            }

            if self.step % self.log_train_every == 0:
                log_step = self.step // self.log_train_every
                self.writer.add_scalar("Loss/train", loss, log_step)
                self.writer.add_scalar("Acc/train", accuracy, log_step)
                self.writer.add_scalar("Loss_cls/train", cls_loss, log_step)
                self.writer.add_scalar("Loss_ft/train", ft_loss, log_step)
                self.writer.add_scalar(
                    "Learning_rate", self.optimizer.param_groups[0]["lr"], log_step
                )
            if self.step % self.mid_epoch_save_every == 0 and batch_idx > 0:
                # Resuming from this restarts the interrupted epoch from its first batch.
                self._save_checkpoint(epoch, {**metrics, "batch_idx": batch_idx})
            print(
                f"\rEpoch {epoch + 1} train {(batch_idx + 1) / len(self.train_loader):6.1%} | "
                f"loss {loss:.4f} (cls {cls_loss:.4f}, ft {ft_loss:.4f}) | acc {accuracy:.4f}",
                end="",
                flush=True,
            )
        return metrics

    def _train_batch(self, sample, fourier_target, labels):
        self.optimizer.zero_grad()
        labels = labels.to(self.config.device)
        logits, fourier = self.model(sample.to(self.config.device))
        classification_loss = self.classification_criterion(logits, labels)
        fourier_loss = self.fourier_criterion(fourier, fourier_target.to(self.config.device))
        loss = 0.5 * classification_loss + 0.5 * fourier_loss
        loss.backward()
        self.optimizer.step()
        accuracy = (logits.argmax(1) == labels).float().mean()
        return np.array(
            [loss.item(), accuracy.item(), classification_loss.item(), fourier_loss.item()]
        )

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        total_loss, total_accuracy = 0.0, 0.0
        preds, targets = [], []
        for batch_idx, (sample, labels) in enumerate(self.valid_loader):
            labels = labels.to(self.config.device)
            logits = self.model(sample.to(self.config.device))
            total_loss += self.classification_criterion(logits, labels).item()
            total_accuracy += (logits.argmax(1) == labels).float().mean().item()
            preds.append(logits.argmax(1).cpu())
            targets.append(labels.cpu())

            self.validation_step += 1
            if self.validation_step % self.log_valid_every == 0:
                log_step = self.validation_step // self.log_valid_every
                self.writer.add_scalar("Acc/valid", total_accuracy / (batch_idx + 1), log_step)
                self.writer.add_scalar("Loss_cls/valid", total_loss / (batch_idx + 1), log_step)

        n = max(1, len(self.valid_loader))
        cm = confusion_matrix(
            torch.cat(targets), torch.cat(preds), labels=list(range(self.config.num_classes))
        )
        per_class_accuracy = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
        metrics = {"validation_loss": total_loss / n, "validation_accuracy": total_accuracy / n}
        return metrics, per_class_accuracy

    def _save_checkpoint(self, epoch, metrics, epoch_end=False, is_best=False):
        """Save latest checkpoint; at epoch end also save per-epoch/best and training_info.json.

        `epoch` is the epoch training resumes from: the next one at epoch end,
        the interrupted one mid-epoch.
        """
        state = {
            "epoch": epoch,
            "step": self.step,
            "validation_step": self.validation_step,
            "best_validation_accuracy": self.best_validation_accuracy,
            **metrics,
        }
        checkpoint = {
            **state,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.lr_scheduler.state_dict(),
            "config": {
                key: getattr(self.config, key)
                for key in ("lr", "batch_size", "input_size", "num_classes", "epochs", "milestones")
            },
        }
        path = self.config.model_path
        torch.save(checkpoint, os.path.join(path, "checkpoint_latest.pth"))
        if not epoch_end:
            return

        torch.save(checkpoint, os.path.join(path, f"checkpoint_epoch_{epoch - 1:03d}.pth"))
        if is_best:
            torch.save(checkpoint, os.path.join(path, "checkpoint_best.pth"))
        with open(os.path.join(path, "training_info.json"), "w") as f:
            json.dump({**state, "epoch": epoch - 1}, f, indent=2)

    def _load_checkpoint(self, checkpoint_path):
        print(f"Resuming from {checkpoint_path}")
        checkpoint = load_weights(self.model.module, checkpoint_path)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = checkpoint["epoch"]
        self.step = checkpoint.get("step", 0)
        # Older checkpoints (including those in models/checkpoints/) use the short names.
        self.validation_step = checkpoint.get("validation_step", checkpoint.get("val_step", 0))
        self.best_validation_accuracy = checkpoint.get(
            "best_validation_accuracy", checkpoint.get("best_val_acc", 0.0)
        )
        if "batch_idx" in checkpoint:
            print(f"Mid-epoch checkpoint: restarting epoch {self.start_epoch + 1} from batch 0")
