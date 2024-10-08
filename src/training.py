# -*- coding: utf-8 -*-
import torch, os
from tqdm.auto import tqdm

def train(
    model,
    train_dataloader,
    data_path,
    optimizer,
    criterion,
    clip,
    device,
    use_ddp=False,
    rank=None,
    load_tensors=False,
):
    # train mode is ON i.e. dropout and normalization tech. will be used
    model.train()

    epoch_loss = 0

    tset = tqdm(iter(train_dataloader))

    for i, (imgs, lbls) in enumerate(tset):
        
        lbls = torch.tensor(lbls).long().to(device)

        if load_tensors:
            _imgs = []
            for i in imgs:
                tnsr = torch.load(i)
                _imgs.append(tnsr)
            
            imgs = torch.stack(_imgs).to(device)

        # setting gradients to zero
        optimizer.zero_grad()

        output = model(
            imgs,
            device,
        )    # (B, num_classes)

        loss = criterion(output.contiguous().view(-1, output.shape[-1]), lbls.contiguous())
        
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

        if (not use_ddp) or (use_ddp and rank == 0):
            desc = 'Loss: %.4f - Learning Rate: %.6f' % (loss.item(), optimizer.param_groups[0]['lr'])
            tset.set_description(desc)

    net_loss = epoch_loss / len(train_dataloader)
    return net_loss