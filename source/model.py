import torch
import torch.nn as nn
from transformers import AutoModel

class MultiInputCafeBERT(nn.Module):
    """
    Multi-Input model: encode riêng context, prompt, response bằng cùng encoder CafeBERT,
    lấy CLS của mỗi phần, concat lại rồi đưa qua classifier.
    """

    def __init__(self, model_name="uitnlp/CafeBERT", hidden_dropout=0.3, num_labels=3, tokenizer=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        # Nếu tokenizer có thêm special tokens (markers), resize lại embedding
        if tokenizer is not None:
            self.encoder.resize_token_embeddings(len(tokenizer))

        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(hidden_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(
        self,
        context_input_ids, context_attention_mask,
        prompt_input_ids, prompt_attention_mask,
        response_input_ids, response_attention_mask
    ):
        # Encode từng input
        out_ctx  = self.encoder(input_ids=context_input_ids, attention_mask=context_attention_mask)
        out_pr   = self.encoder(input_ids=prompt_input_ids, attention_mask=prompt_attention_mask)
        out_resp = self.encoder(input_ids=response_input_ids, attention_mask=response_attention_mask)

        # Lấy CLS token (vị trí 0)
        cls_ctx  = out_ctx.last_hidden_state[:, 0, :]
        cls_pr   = out_pr.last_hidden_state[:, 0, :]
        cls_resp = out_resp.last_hidden_state[:, 0, :]

        # Ghép 3 vector CLS lại
        h = torch.cat([cls_ctx, cls_pr, cls_resp], dim=1)  # [B, 3*hidden]
        h = self.dropout(h)
        logits = self.classifier(h)
        return logits
