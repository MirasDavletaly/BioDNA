import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from typing import List, Tuple, Optional, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CancerDataset(Dataset):

    def __init__(
        self,
        sequences: List[str],
        labels: np.ndarray,
        tokenizer,
        max_length: int = 512,
    ):
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.sequences[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         self.labels[idx],
        }


class DNABERTCancerClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "zhihan1996/DNABERT-2-117M",
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.model_name = model_name

        logger.info(f"Загружаем DNABERT-2: {model_name}")

        # DNABERT-2 backbone
        self.encoder = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        hidden_size = self.encoder.config.hidden_size
        logger.info(f"   Hidden size: {hidden_size}")

        if freeze_backbone:
            logger.info(" Backbone заморожен — обучаем только classifier head")
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            # Заморозим только первые N слоёв
            n_freeze = 6
            for i, layer in enumerate(self.encoder.encoder.layer[:n_freeze]):
                for param in layer.parameters():
                    param.requires_grad = False
            logger.info(f"Заморожено первых {n_freeze} слоёв BERT")

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout * 0.7),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]

        logits = self.classifier(cls_output)
        return logits

class LightweightKmerClassifier(nn.Module):
    def __init__(self, vocab_size: int = 4**6 + 5, embed_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm  = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.attn  = nn.Linear(hidden * 2, 1)
        self.fc    = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.embed(input_ids)                      # [B, L, E]
        out, _ = self.lstm(x)                          # [B, L, 2H]
        attn_w = torch.softmax(self.attn(out), dim=1)  # [B, L, 1]
        ctx = (out * attn_w).sum(dim=1)                # [B, 2H]
        return self.fc(ctx)
class CancerClassifierTrainer:

    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
    ):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"🖥Устройство: {self.device}")

        self.model = model.to(self.device)
        self.lr    = learning_rate
        self.wd    = weight_decay

        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
    ) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in dataloader:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"].to(self.device)

            optimizer.zero_grad()
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss   = self.criterion(logits, labels)
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()
            if scheduler:
                scheduler.step()

            total_loss += loss.item()
            preds       = logits.argmax(dim=-1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

        return total_loss / len(dataloader), correct / total

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        total_loss = 0.0

        for batch in dataloader:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"].to(self.device)

            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss   = self.criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

        from sklearn.metrics import (
            accuracy_score, roc_auc_score, f1_score,
            precision_score, recall_score,
        )

        metrics = {
            "loss":      total_loss / len(dataloader),
            "accuracy":  accuracy_score(all_labels, all_preds),
            "auc":       roc_auc_score(all_labels, all_probs),
            "f1":        f1_score(all_labels, all_preds, zero_division=0),
            "precision": precision_score(all_labels, all_preds, zero_division=0),
            "recall":    recall_score(all_labels, all_preds, zero_division=0),
        }
        return metrics, np.array(all_labels), np.array(all_preds), np.array(all_probs)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        save_path: Optional[str] = None,
    ) -> Dict:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=self.wd,
        )
        total_steps  = len(train_loader) * epochs
        warmup_steps = total_steps // 10
        scheduler    = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        history  = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": []}
        best_auc = 0.0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, scheduler)
            val_metrics, _, _, _  = self.evaluate(val_loader)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_auc"].append(val_metrics["auc"])

            logger.info(
                f"Epoch {epoch:02d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} "
                f"Acc: {val_metrics['accuracy']:.4f} "
                f"AUC: {val_metrics['auc']:.4f} "
                f"F1: {val_metrics['f1']:.4f}"
            )

            # Сохраняем лучшую модель
            if val_metrics["auc"] > best_auc and save_path:
                best_auc = val_metrics["auc"]
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"Сохранена лучшая модель (AUC={best_auc:.4f})")

        return history