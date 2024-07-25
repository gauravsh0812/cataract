# -*- coding: utf-8 -*-

import torch, os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def evaluate(
    model,
    test_dataloader,
    data_path,
    criterion,
    device,
    # is_test=False,
    load_tensors=False,
):
    model.eval()
    epoch_loss = 0
    
    all_preds = []
    all_labels = []

    # if is_test:
    #     labels_file = open("logs/predictions.txt","w")
    #     labels_file.write("Images \t Prediction \n")

    with torch.no_grad():
        for i, (imgs, lbls) in enumerate(test_dataloader):
            
            lbls = torch.tensor(lbls).long().to(device)

            if load_tensors:

                _imgs = []
                for i in imgs:
                    name = os.path.basename(i).split(".")[0]
                    tnsr = torch.load(f"{data_path}/frame_tensors/{name}.pt")#.squeeze(0)
                    _imgs.append(tnsr)
                
                _imgs = torch.stack(_imgs).to(device)

            output = model(
                imgs,
                device,
            )  # (B, num_classes)
            
            loss = criterion(
                output.contiguous().view(-1, output.shape[-1]), 
                lbls.contiguous())

            epoch_loss += loss.item()
        
        # if is_test:
        # Accumulate predictions and true labels
        preds = torch.argmax(output, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(lbls.cpu().numpy())

        precision_macro = precision_score(all_labels, all_preds, average='macro')
        recall_macro = recall_score(all_labels, all_preds, average='macro')
        f1_macro = f1_score(all_labels, all_preds, average='macro')

        precision_micro = precision_score(all_labels, all_preds, average='micro')
        recall_micro = recall_score(all_labels, all_preds, average='micro')
        f1_micro = f1_score(all_labels, all_preds, average='micro')

        precision_weighted = precision_score(all_labels, all_preds, average='weighted')
        recall_weighted = recall_score(all_labels, all_preds, average='weighted')
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')

        accuracy = accuracy_score(all_labels, all_preds)

        print("Scores:")
        print("accuracy: ", accuracy)
        print("MACRO precision, recall, F1: ", precision_macro, recall_macro, f1_macro)
        print("MICRO precision, recall, F1: ", precision_micro, recall_micro, f1_micro)
        print("WGT precision, recall, F1: ", precision_weighted, recall_weighted, f1_weighted)


        net_loss = epoch_loss / len(test_dataloader)
        return net_loss   