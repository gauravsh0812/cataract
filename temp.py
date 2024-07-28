import os
import yaml
import random
import time
import math
import numpy as np
import torch 
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer, AdamW, get_scheduler
from datasets import load_dataset
from box import Box
from peft import get_peft_model, LoraConfig
from src.utils import *
from src.training import train
from src.testing import evaluate
from dataset.preprocessing import preprocess

with open("config.yaml") as f:
    cfg = Box(yaml.safe_load(f))

class Llama3Model(nn.Module):
    def __init__(self, model_name):
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.model = LlamaForCausalLM.from_pretrained(model_name).to(self.device)
        self._apply_lora()

    def _apply_lora(self):
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)

    def tokenize_dataset(self, dataset):
        def tokenize_function(examples):
            return self.tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)
        tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask"])
        return tokenized_datasets

def train_model(self, rank=None):
    set_random_seed(self.cfg.general.seed)
    FOLDER = ["trained_models", "logs"]
    for f in FOLDER:
        if not os.path.exists(f):
            os.mkdir(f)
    loss_file = open("logs/loss_file.txt", "w")

    if torch.cuda.is_available() and self.cfg.general.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cfg.general.gpus)
        device = torch.device(f"cuda:{self.cfg.general.gpus}")
        (
            train_dataloader,
            test_dataloader,
            val_dataloader,
            num_classes
        ) = preprocess(self.cfg.training.general.batch_size)
        model = Llama3Model(num_classes).to(device)

    print("MODEL: ")
    print(f"The model has {count_parameters(model)} trainable parameters")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(
        params=model.parameters(),
        lr=self.cfg.training.general.learning_rate,
        weight_decay=self.cfg.training.general.weight_decay,
        betas=self.cfg.training.general.betas,
    )

    if self.cfg.training.scheduler.isScheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.cfg.training.scheduler.scheduler_step_size,
            gamma=self.cfg.training.scheduler.scheduler_gamma,
        )
    else:
        scheduler = None

    best_valid_loss = float("inf")

    if not self.cfg.general.load_trained_model_for_testing:
        count_es = 0
        for epoch in range(self.cfg.training.general.epochs):
            if count_es <= self.cfg.training.general.early_stopping:
                start_time = time.time()

                train_loss = train(
                    model,
                    train_dataloader,
                    self.cfg.dataset.path_to_data,
                    optimizer,
                    criterion,
                    self.cfg.training.general.clip,
                    device,
                    use_ddp=self.cfg.general.ddp,
                    rank=rank,
                    load_tensors=self.cfg.dataset.load_image_tensors,
                )

                val_loss = evaluate(
                    model,
                    val_dataloader,
                    self.cfg.dataset.path_to_data,
                    criterion,
                    device,
                    load_tensors=self.cfg.dataset.load_image_tensors
                )

                if self.cfg.training.scheduler.isScheduler:
                    scheduler.step()

                end_time = time.time()
                epoch_mins, epoch_secs = epoch_time(start_time, end_time)

                if (not self.cfg.general.ddp) or (self.cfg.general.ddp and rank == 0):
                    torch.save(
                        model.state_dict(),
                        f"trained_models/latest_model.pt",
                    )

                if val_loss < best_valid_loss:
                    best_valid_loss = val_loss
                    count_es = 0
                    if (not self.cfg.general.ddp) or (self.cfg.general.ddp and rank == 0):
                        torch.save(
                            model.state_dict(),
                            f"trained_models/{self.cfg.model_name}.pt",
                        )
                else:
                    count_es += 1

                if (not self.cfg.general.ddp) or (self.cfg.general.ddp and rank == 0):
                    print(
                        f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s"
                    )
                    print(
                        f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}"
                    )
                    print(
                        f"\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}"
                    )

                    loss_file.write(
                        f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s\n"
                    )
                    loss_file.write(
                        f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}\n"
                    )
                    loss_file.write(
                        f"\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}\n"
                    )
            else:
                print(
                    f"Terminating the training process as the validation loss hasn't been reduced from last {self.cfg.training.general.early_stopping} epochs."
                )
                break

        print(
            "best model saved as:  ",
            f"trained_models/{self.cfg.model_name}.pt",
        )

    if self.cfg.general.ddp:
        dist.destroy_process_group()

    time.sleep(3)

    print(
        "loading best saved model: ",
        f"trained_models/{self.cfg.model_name}.pt",
    )
    model.load_state_dict(
        torch.load(
            f"trained_models/{self.cfg.model_name}.pt"
        )
    )

    test_loss = evaluate(
        model,
        test_dataloader,
        self.cfg.dataset.path_to_data,
        criterion,
        device,
        load_tensors=self.cfg.dataset.load_image_tensors
    )

    if (not self.cfg.general.ddp) or (self.cfg.general.ddp and rank == 0):
        print(
            f"| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f}"
        )
        loss_file.write(
            f"| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f}"
        )

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

def set_random_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

if __name__ == "__main__":
    model_name = "Llama3"
    llama_model = Llama3Model(model_name, cfg)
    llama_model.train_model()
