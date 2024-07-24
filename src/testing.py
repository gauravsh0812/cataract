# -*- coding: utf-8 -*-

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

def get_key_by_value(dictionary, target_value):
    for key, value in dictionary.items():
        if value == target_value:
            return key
    return None  # Return None if the value is not found

def evaluate(
    model,
    test_dataloader,
    criterion,
    device,
    qtn_tokenizer,
    ans_vocab,
    is_test=False,
):
    model.eval()
    epoch_loss = 0
    allans, allpreds = [],[]

    if is_test:
        labels_file = open("logs/predictions.txt","w")
        labels_file.write("Images \t Qtns \t Target \t Prediction \n")

    with torch.no_grad():
        for i, (imgs, qtn_ids, qtn_attns, ans) in enumerate(test_dataloader):
            qtn_attns = qtn_attns.to(device)
            qtn_ids = qtn_ids.to(device)
    
            ans = torch.tensor(ans).long().to(device)

            # _imgs = []
            # for i in imgs:
            #     name = os.path.basename(i).split(".")[0]
            #     tnsr = torch.load(f"/data/gauravs/surgicalGPT/cholec80/image_tensors/{name}.pt")#.squeeze(0)
            #     _imgs.append(tnsr)
            
            # _imgs = torch.stack(_imgs).to(device)

            output = model(
                imgs,
                qtn_ids,
                qtn_attns,
                device,
            )
            
            pred = torch.argmax(output, dim=-1)

            """
            q shape:  torch.Size([16, 32])
            a shape:  torch.Size([16, 3])
            ouptut shape:  torch.Size([16, 3, 52])
            pred shape:  torch.Size([16, 3])
            """

            loss = criterion(
                output.contiguous().view(-1, output.shape[-1]), 
                ans.contiguous().view(-1))

            epoch_loss += loss.item()
            
            for b in range(ans.shape[0]):
                a = get_key_by_value(ans_vocab, ans[b])
                p = get_key_by_value(ans_vocab, pred[b])
                
                allans.append(a)
                allpreds.append(p)

                if is_test:        
                    im = imgs[b]
                    q = qtn_tokenizer.decode(qtn_ids[b,:])
                    # a = "".join([ans_vocab.itos[i] for i in ans[b,:]])
                    # p = "".join([ans_vocab.itos[i] for i in pred[b,:]])
                    
                    labels_file.write(
                        f"{im} \t {q} \t {a} \t {p} \n"
                    )
        
        accuracy = accuracy_score(allans, allpreds)
        print("accuracy: ", accuracy)
        
        if is_test:
            cm = confusion_matrix(allans, allpreds)
            print(cm)

            print(">>>>>>> weighted scores...")
            precision = precision_score(allans, allpreds, average='weighted')
            recall = recall_score(allans, allpreds, average='weighted')
            f1 = f1_score(allans, allpreds, average='weighted')
            print("precision: ", precision)
            print("recall: ", recall)
            print("F1 score: ", f1)

            print(">>>>>>> macro scores...")
            precision = precision_score(allans, allpreds, average='macro')
            recall = recall_score(allans, allpreds, average='macro')
            f1 = f1_score(allans, allpreds, average='macro')
            print("precision: ", precision)
            print("recall: ", recall)
            print("F1 score: ", f1)

            print(">>>>>>> micro scores...")
            precision = precision_score(allans, allpreds, average='micro')
            recall = recall_score(allans, allpreds, average='micro')
            f1 = f1_score(allans, allpreds, average='micro')
            print("precision: ", precision)
            print("recall: ", recall)
            print("F1 score: ", f1)
        
        net_loss = epoch_loss / len(test_dataloader)
        return net_loss   