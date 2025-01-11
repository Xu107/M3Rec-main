import torch
from torch import nn
from mamba_ssm import Mamba
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class M3Rec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(M3Rec, self).__init__(config, dataset)

        #prepare for multi-modal features
        dataset=config["dataset"]
        feature_path=f"./dataset/{dataset}/"
        img_feat,text_feat=nn.Embedding.from_pretrained(torch.load(feature_path+"img_feat.pt"),freeze=True),\
                           nn.Embedding.from_pretrained(torch.load(feature_path+"text_feat.pt"),freeze=True)
        # device = torch.device("cuda:"+config["gpu_id"])
        self.img_feat=nn.Embedding.from_pretrained(torch.cat(
            (torch.zeros(1,img_feat.weight.shape[-1]),img_feat.weight),dim=0
        ),freeze=True)
        self.text_feat=nn.Embedding.from_pretrained(torch.cat(
            (torch.zeros(1,text_feat.weight.shape[-1]),text_feat.weight),dim=0
        ),freeze=True)
        self.img_alpha=torch.nn.Parameter(torch.tensor([1.]))
        self.text_beta=torch.nn.Parameter(torch.tensor([1.]))


        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]

        # Hyperparameters for Mamba block
        self.d_state = config["d_state"]
        self.d_conv = config["d_conv"]
        self.expand = config["expand"]

        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.img_trans = nn.Sequential(
            nn.Linear(self.img_feat.weight.shape[-1], self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.text_trans = nn.Sequential(
            nn.Linear(self.text_feat.weight.shape[-1], self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.img_LayerNorm=nn.LayerNorm(self.hidden_size,eps=1e-12)
        self.text_LayerNorm=nn.LayerNorm(self.hidden_size,eps=1e-12)

        self.mamba_block=nn.ModuleList([MambaBlock(
                d_model=self.hidden_size,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout_prob,
                num_layers=self.num_layers,
            ) for _ in range(self.num_layers)])

        self.id_mamba_ffn=nn.ModuleList([FeedForward(d_model=self.hidden_size, inner_size=self.hidden_size * 4,
                                        dropout=self.dropout_prob) for _ in range(self.num_layers)])
        self.img_mamba_ffn=nn.ModuleList([FeedForward(d_model=self.hidden_size, inner_size=self.hidden_size * 4,
                                        dropout=self.dropout_prob) for _ in range(self.num_layers)])
        self.text_mamba_ffn=nn.ModuleList([FeedForward(d_model=self.hidden_size, inner_size=self.hidden_size * 4,
                                        dropout=self.dropout_prob) for _ in range(self.num_layers)])

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)
        img_emb=self.img_feat(item_seq)
        text_emb=self.text_feat(item_seq)

        img_emb=self.img_trans(img_emb)
        text_emb=self.text_trans(text_emb)

        item_emb = self.dropout(item_emb)
        item_emb = self.LayerNorm(item_emb)

        img_emb=self.dropout(img_emb)
        text_emb=self.dropout(text_emb)
        img_emb=self.img_LayerNorm(img_emb)
        text_emb=self.text_LayerNorm(text_emb)

        for i in range(self.num_layers):
            item_emb=self.mamba_block[i](item_emb)
            item_emb=self.id_mamba_ffn[i](item_emb)

        for i in range(self.num_layers):
            with torch.no_grad():
                img_emb = self.mamba_block[i](img_emb)
            img_emb = self.img_mamba_ffn[i](img_emb)
        for i in range(self.num_layers):
            with torch.no_grad():
                text_emb = self.mamba_block[i](text_emb)
            text_emb = self.text_mamba_ffn[i](text_emb)

        mm_emb=torch.cat((self.img_alpha*img_emb,self.text_beta*text_emb,item_emb),dim=-1)

        seq_output = self.gather_indexes(mm_emb, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb=torch.concat((self.img_alpha*self.img_trans(self.img_feat.weight),\
                          self.text_beta*self.text_trans(self.text_feat.weight),\
                          self.item_embedding.weight),dim=-1)
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_id_emb = self.item_embedding(test_item)
        test_item_img_emb = self.img_alpha*self.img_trans(self.img_feat(test_item))
        test_item_text_emb = self.text_beta*self.text_trans(self.text_feat(test_item))
        test_item_emb=torch.cat((test_item_img_emb,test_item_text_emb,test_item_id_emb),dim=-1)

        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = torch.cat((self.img_alpha * self.img_trans(self.img_feat.weight) , \
                        self.text_beta * self.text_trans(self.text_feat.weight) , \
                        self.item_embedding.weight),dim=-1)

        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        # print(self.img_alpha,self.text_beta)
        return scores


class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.mamba = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = FeedForward(d_model=d_model, inner_size=d_model * 4, dropout=dropout)

    def forward(self, input_tensor):
        hidden_states = self.mamba(input_tensor)
        if self.num_layers == 1:  # one Mamba layer without residual connection
            hidden_states = self.LayerNorm(self.dropout(hidden_states))
        else:  # stacked Mamba layers with residual connections
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)
        hidden_states = self.ffn(hidden_states)
        return hidden_states

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.mamba = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.mamba(input_tensor)
        if self.num_layers == 1:  # one Mamba layer without residual connection
            hidden_states = self.LayerNorm(self.dropout(hidden_states))
        else:  # stacked Mamba layers with residual connections
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)
        return hidden_states

class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
