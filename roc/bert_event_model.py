from transformers import *
import torch
import torch.nn as nn
class BertForMultipleChoice_NTN(BertForMultipleChoice):
    def create(self,cuda_num,large):
        self.cuda_num=cuda_num
        self.NTN_concat=nn.Linear(240,large).cuda(self.cuda_num)
        self.BiLSTM=nn.LSTM(input_size=100,hidden_size = 120,num_layers = 2,batch_first=True,bidirectional=True).cuda(self.cuda_num)
    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None, position_ids=None,head_mask=None,inputs_embeds=None,labels=None,emb=None):
        num_choices = input_ids.shape[1]
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]
        if emb != None:
            BiLSTM_emb=torch.FloatTensor(len(emb),240).cuda(self.cuda_num)
            for i in range(len(emb)):
                outputrq1,(hn,cn)=self.BiLSTM(torch.stack(emb[i],dim=1))
                BiLSTM_emb[i]=outputrq1[0][outputrq1.size()[1]-1]
            event_chain_emb=self.NTN_concat(BiLSTM_emb)
            prj_emb=torch.cat([event_chain_emb,event_chain_emb],dim=0)
            pooled_output = self.dropout(pooled_output+prj_emb)
        else:
            pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs
        return outputs
