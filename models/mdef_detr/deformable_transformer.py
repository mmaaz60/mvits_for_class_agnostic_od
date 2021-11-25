from typing import Optional
import math
import torch
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, normal_
from models.ops.modules import MSDeformAttn
from transformers import RobertaModel, RobertaTokenizerFast
from models.deformable_modules import DeformableTransformerEncoderLayer, DeformableTransformerEncoder, \
    DeformableTransformerDecoderLayer, DeformableTransformerDecoder, _get_clones, _get_activation_fn


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4,
                 pass_pos_and_query=True,
                 text_encoder_type="roberta-base",
                 freeze_text_encoder=False,
                 ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        img_text_layer = ImageTextTransformerLayer(d_model, 8, dim_feedforward, dropout, activation)
        self.img_text_attn = ImageTextTransformer(img_text_layer, 6)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

        self.pass_pos_and_query = pass_pos_and_query
        self.tokenizer = RobertaTokenizerFast.from_pretrained(text_encoder_type)
        self.text_encoder = RobertaModel.from_pretrained(text_encoder_type)

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        self.expander_dropout = 0.1
        config = self.text_encoder.config
        self.resizer = FeatureResizer(
            input_feat_size=config.hidden_size,
            output_feat_size=d_model,
            dropout=self.expander_dropout,
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(
            self,
            srcs=None,
            masks=None,
            query_embed=None,
            pos_embeds=None,
            text=None,
            encode_and_save=True,
            text_memory=None,
            img_memory=None,
            text_attention_mask=None,
            spatial_shapes=None,
            level_start_index=None,
            valid_ratios=None,
    ):
        assert query_embed is not None

        # prepare input for encoder
        if encode_and_save:
            src_flatten = []
            mask_flatten = []
            lvl_pos_embed_flatten = []
            spatial_shapes = []

            # Encode the text
            device = srcs[0].device
            if isinstance(text[0], str):
                # Encode the text
                tokenized = self.tokenizer.batch_encode_plus(text, padding="longest", return_tensors="pt").to(
                    device)
                encoded_text = self.text_encoder(**tokenized)

                # Transpose memory because pytorch's attention expects sequence first
                text_memory = encoded_text.last_hidden_state.transpose(0, 1)
                # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
                text_attention_mask = tokenized.attention_mask.ne(1).bool()

                # Resize the encoder hidden states to be of the same d_model as the decoder
                text_memory_resized = self.resizer(text_memory)
            else:
                # The text is already encoded, use as is.
                text_attention_mask, text_memory_resized, tokenized = text

            for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):  # multi-scale implemented
                bs, c, h, w = src.shape
                spatial_shape = (h, w)
                spatial_shapes.append(spatial_shape)
                src = src.flatten(2).transpose(1, 2)
                mask = mask.flatten(1)
                pos_embed = pos_embed.flatten(2).transpose(1, 2)
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
                lvl_pos_embed_flatten.append(lvl_pos_embed)
                src_flatten.append(src)
                mask_flatten.append(mask)

            src_flatten = torch.cat(src_flatten, 1)
            mask_flatten = torch.cat(mask_flatten, 1)
            lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
            spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
            level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
            valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

            # encoder
            img_memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios,
                                      lvl_pos_embed_flatten, mask_flatten)

            memory_cache = {
                "text_memory_resized": text_memory_resized,
                "img_memory": img_memory,
                "mask": mask_flatten,
                "text_attention_mask": text_attention_mask,
                "pos_embed": lvl_pos_embed_flatten,
                "query_embed": query_embed,
                "tokenized": tokenized,
                "spatial_shapes": spatial_shapes,
                "level_start_index": level_start_index,
                "valid_ratios": valid_ratios,
            }
            return memory_cache

        else:
            # prepare input for decoder
            bs, _, c = img_memory.shape

            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points

        # decoder
        hs, inter_references = self.decoder(tgt, reference_points, img_memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed,
                                            src_padding_mask=masks)
        # Concatenate decoder output with the text embeddings
        hs_mask = torch.zeros((hs.shape[0], hs.shape[1]), dtype=torch.bool, device=hs.device)
        hs = torch.cat([hs.permute(1, 0, 2), text_memory], dim=0)
        hs_mask = torch.cat([hs_mask, text_attention_mask], dim=1)
        hs = self.img_text_attn(hs, src_key_padding_mask=hs_mask)
        text_memory_dec = [h[-len(text_memory):] for h in hs]
        hs = [h[:query_embed.shape[1]].permute(1, 0, 2) for h in hs]

        inter_references_out = [inter_references for i in range(len(hs))]

        return torch.stack(hs), init_reference_out, torch.stack(inter_references_out), torch.stack(text_memory_dec)


class ImageTextTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward models
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
            self,
            src,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
            self,
            src,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
            self,
            src,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class ImageTextTransformer(nn.Module):
    def __init__(self, layer, num_layers, norm=None, return_intermediate=True):
        super().__init__()
        self.layers = _get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
            self,
            src,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):

        output = src

        all_outputs = []
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
            all_outputs.append(output)

        if self.norm is not None:
            all_outputs = [self.norm(o) for o in all_outputs]

        if self.return_intermediate:
            return all_outputs
        else:
            return all_outputs[-1]


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output
