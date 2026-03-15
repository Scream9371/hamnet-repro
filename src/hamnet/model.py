"""HAM-Net model components used by the reproducibility package."""

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class AttentionBlock(nn.Module):


    def __init__(self, hidden_size: int) -> None:

        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self, states: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if states.dim() == 2:
            states = states.unsqueeze(0)
            mask = mask.unsqueeze(0)
        proj_state = torch.tanh(self.proj(states))
        attn_scores = self.score(proj_state).squeeze(-1)
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(attn_scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        context = torch.bmm(weights.unsqueeze(1), states).squeeze(1)
        return context, weights


class HierarchicalAttention(nn.Module):


    def __init__(self, hidden_size: int, segment_len: int = 32) -> None:

        super().__init__()
        self.segment_len = segment_len
        self.word_att = AttentionBlock(hidden_size)
        self.segment_att = AttentionBlock(hidden_size)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        batch_size, seq_len, hidden = hidden_states.size()
        segment_len = self.segment_len
        n_segments = (seq_len + segment_len - 1) // segment_len
        pad_len = n_segments * segment_len - seq_len
        if pad_len > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))
            attention_mask = F.pad(attention_mask, (0, pad_len))

        states = hidden_states.view(batch_size, n_segments, segment_len, hidden)
        masks = attention_mask.view(batch_size, n_segments, segment_len)

        flat_states = states.view(batch_size * n_segments, segment_len, hidden)
        flat_masks = masks.view(batch_size * n_segments, segment_len)
        word_contexts, word_weights = self.word_att(flat_states, flat_masks)
        word_contexts = word_contexts.view(batch_size, n_segments, hidden)
        word_weights = word_weights.view(batch_size, n_segments, segment_len)

        segment_mask = (masks.sum(dim=-1) > 0).long()
        doc_context, segment_weights = self.segment_att(word_contexts, segment_mask)

        return doc_context, {
            "token_weights": word_weights,
            "segment_weights": segment_weights,
        }


class GraphAttentionLayer(nn.Module):


    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:

        super().__init__()
        self.hidden_size = hidden_size
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.edge_mlp = nn.Linear(hidden_size * 2, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.act = nn.GELU()

    def forward(
        self, node_states: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:

        if edge_index.numel() == 0:
            return self.act(self.norm(node_states))
        src, dst = edge_index
        h = self.proj(node_states)
        edge_feat = torch.cat([h[src], h[dst]], dim=-1)
        scores = F.leaky_relu(self.edge_mlp(edge_feat).squeeze(-1), negative_slope=0.2)
        scores = scores - scores.max()
        attn = torch.exp(scores)
        denom = torch.zeros(
            node_states.size(0), device=node_states.device, dtype=attn.dtype
        ).index_add_(0, dst, attn)
        denom = denom + 1e-6
        weights = (attn / denom[dst]).to(h.dtype)
        messages = h[src] * weights.unsqueeze(-1)
        agg = torch.zeros_like(h).index_add_(0, dst, messages)
        agg = self.dropout(agg)
        out = self.norm(agg + node_states)
        return self.act(out)


class GraphAttentionEncoder(nn.Module):


    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.layers = nn.ModuleList(
            [
                GraphAttentionLayer(hidden_size, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.pool = AttentionBlock(hidden_size)

    def forward(
        self, graphs: Sequence[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:

        graph_vectors = []
        attn_weights: List[torch.Tensor] = []
        valid_mask = []
        device = graphs[0]["node_ids"].device if graphs else torch.device("cpu")
        for graph in graphs:
            node_ids = graph["node_ids"]
            edge_index = graph["edge_index"]
            if node_ids.numel() == 0 or (node_ids != 0).sum() == 0:
                graph_vectors.append(
                    torch.zeros(self.hidden_size, device=device, dtype=torch.float32)
                )
                attn_weights.append(None)
                valid_mask.append(False)
                continue

            node_states = self.embedding(node_ids)
            node_mask = (node_ids != 0).long()
            if node_mask.sum() == 0:
                graph_vectors.append(
                    torch.zeros(self.hidden_size, device=device, dtype=torch.float32)
                )
                attn_weights.append(None)
                valid_mask.append(False)
                continue

            for layer in self.layers:
                node_states = layer(node_states, edge_index)
            graph_vec, weights = self.pool(node_states, node_mask)

            if graph_vec.dim() == 2 and graph_vec.size(0) == 1:
                graph_vec = graph_vec.squeeze(0)
            if weights.dim() == 2 and weights.size(0) == 1:
                weights = weights.squeeze(0)
            graph_vectors.append(graph_vec)
            attn_weights.append(weights)
            valid_mask.append(True)

        vectors = (
            torch.stack(graph_vectors, dim=0)
            if graph_vectors
            else torch.zeros(0, self.hidden_size, device=device)
        )
        mask_tensor = torch.as_tensor(
            valid_mask, device=device, dtype=torch.float32
        ).view(-1)
        return vectors, attn_weights, mask_tensor


class HamNetEncoder(nn.Module):


    def __init__(
        self,
        encoder_name: str,
        node_vocab_size: int,
        segment_len: int = 32,
        graph_hidden: int = 256,
        dropout: float = 0.2,
        freeze_encoder: bool = False,
        unfreeze_last_n: int = 0,
        use_hier_attn: bool = True,
        use_graph: bool = True,
        use_ast_seq: bool = False,
        semantic_chunk_size: int = 0,
        enable_gradient_checkpointing: bool = False,
    ) -> None:

        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.semantic_chunk_size = max(int(semantic_chunk_size), 0)
        if enable_gradient_checkpointing:
            try:
                self.encoder.gradient_checkpointing_enable()
            except Exception:

                pass
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            if unfreeze_last_n and unfreeze_last_n > 0:

                encoder_layers = getattr(
                    getattr(self.encoder, "encoder", None), "layer", None
                )
                if encoder_layers is not None:
                    for layer in encoder_layers[-unfreeze_last_n:]:
                        for param in layer.parameters():
                            param.requires_grad = True

        self.use_hier_attn = use_hier_attn
        self.use_graph = use_graph
        self.use_ast_seq = use_ast_seq
        self.graph_hidden = graph_hidden
        self.fusion_mode = "gate_concat"

        hidden_size = self.encoder.config.hidden_size
        if self.use_hier_attn:
            self.semantic_att = HierarchicalAttention(hidden_size, segment_len)
        else:
            self.semantic_att = None


        self.sem_norm = nn.LayerNorm(hidden_size)
        self.graph_norm = nn.LayerNorm(graph_hidden) if self.use_graph else None
        self.delta_norm = nn.LayerNorm(hidden_size)
        self.ast_seq_norm = nn.LayerNorm(graph_hidden) if self.use_ast_seq else None

        if self.use_graph:
            self.graph_encoder = GraphAttentionEncoder(
                vocab_size=node_vocab_size, hidden_size=graph_hidden
            )
            fusion_dim = hidden_size + graph_hidden


            self.graph_to_sem = nn.Sequential(
                nn.Linear(graph_hidden, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
            )

            self.graph_gate = None
        else:
            self.graph_encoder = None
            fusion_dim = hidden_size

            self.graph_to_sem = None
            self.graph_gate = None

        if self.use_ast_seq:
            self.ast_embedding = nn.Embedding(
                node_vocab_size, graph_hidden, padding_idx=0
            )
            self.ast_conv = nn.Conv1d(
                graph_hidden, graph_hidden, kernel_size=5, padding=2
            )
            self.ast_pool = nn.AdaptiveMaxPool1d(1)
            fusion_dim += graph_hidden
        else:
            self.ast_embedding = None
            self.ast_conv = None
            self.ast_pool = None

        self.fusion_dim = fusion_dim

    def _encode_semantic(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor | None]]:

        total_funcs = int(input_ids.size(0))
        chunk_size = self.semantic_chunk_size
        if chunk_size <= 0 or total_funcs <= chunk_size:
            encoder_outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            hidden_states = encoder_outputs.last_hidden_state
            if (
                hidden_states is not None
                and self.use_hier_attn
                and self.semantic_att is not None
            ):
                return self.semantic_att(hidden_states, attention_mask)
            return hidden_states[:, 0, :], {
                "token_weights": None,
                "segment_weights": None,
            }

        semantic_chunks: List[torch.Tensor] = []
        token_weight_chunks: List[torch.Tensor] = []
        segment_weight_chunks: List[torch.Tensor] = []
        has_token_weights = True
        has_segment_weights = True

        for start in range(0, total_funcs, chunk_size):
            end = min(start + chunk_size, total_funcs)
            chunk_ids = input_ids[start:end]
            chunk_mask = attention_mask[start:end]
            outputs = self.encoder(input_ids=chunk_ids, attention_mask=chunk_mask)
            hidden_states = outputs.last_hidden_state
            if (
                hidden_states is not None
                and self.use_hier_attn
                and self.semantic_att is not None
            ):
                chunk_vec, chunk_attn = self.semantic_att(hidden_states, chunk_mask)
                semantic_chunks.append(chunk_vec)
                token_w = chunk_attn.get("token_weights")
                seg_w = chunk_attn.get("segment_weights")
                if token_w is None:
                    has_token_weights = False
                else:
                    token_weight_chunks.append(token_w)
                if seg_w is None:
                    has_segment_weights = False
                else:
                    segment_weight_chunks.append(seg_w)
            else:
                semantic_chunks.append(hidden_states[:, 0, :])
                has_token_weights = False
                has_segment_weights = False

        semantic_vec = torch.cat(semantic_chunks, dim=0)
        semantic_attn = {
            "token_weights": (
                torch.cat(token_weight_chunks, dim=0)
                if has_token_weights and token_weight_chunks
                else None
            ),
            "segment_weights": (
                torch.cat(segment_weight_chunks, dim=0)
                if has_segment_weights and segment_weight_chunks
                else None
            ),
        }
        return semantic_vec, semantic_attn

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        graphs: Sequence[Dict[str, torch.Tensor]],
        ast_seq: torch.Tensor | None = None,
        ast_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        semantic_vec, semantic_attn = self._encode_semantic(input_ids, attention_mask)

        target_dtype = self.sem_norm.weight.dtype
        if semantic_vec.dtype != target_dtype:
            semantic_vec = semantic_vec.to(target_dtype)
        semantic_vec = self.sem_norm(semantic_vec)

        graph_attn = None
        graph_vec = None
        graph_mask = None
        ast_vec = None
        if (
            self.use_graph
            and self.graph_encoder is not None
            and graphs is not None
        ):
            graph_vec, graph_attn, graph_mask = self.graph_encoder(graphs)
            if graph_vec is not None and self.graph_norm is not None:
                graph_vec = self.graph_norm(graph_vec.to(target_dtype))
            if graph_mask is not None and graph_vec is not None:
                graph_vec = graph_vec * graph_mask.unsqueeze(-1)
            if self.graph_to_sem is not None and graph_vec is not None:

                delta = self.graph_to_sem(graph_vec)
                delta = self.delta_norm(delta)
                semantic_vec = self.sem_norm(semantic_vec + delta)

        if self.use_ast_seq and ast_seq is not None:
            ast_ids = ast_seq.to(semantic_vec.device)
            ast_mask_t = (
                ast_mask.to(semantic_vec.device) if ast_mask is not None else None
            )
            emb = self.ast_embedding(ast_ids)
            emb = emb.transpose(1, 2)
            conv_out = torch.relu(self.ast_conv(emb))
            if ast_mask_t is not None:
                mask = ast_mask_t.unsqueeze(1)
                conv_out = conv_out.masked_fill(mask == 0, float("-inf"))
            pooled = self.ast_pool(conv_out).squeeze(-1)
            pooled = torch.nan_to_num(pooled, neginf=0.0)
            if self.ast_seq_norm is not None:
                pooled = self.ast_seq_norm(pooled.to(target_dtype))
            ast_vec = pooled

        fuse_list = [semantic_vec]
        if graph_vec is not None:
            fuse_list.append(graph_vec.to(target_dtype))
        if ast_vec is not None:
            fuse_list.append(ast_vec.to(target_dtype))
        fused = torch.cat(fuse_list, dim=-1)

        attn_info = {
            "token_weights": semantic_attn["token_weights"],
            "segment_weights": semantic_attn["segment_weights"],
            "graph_node_weights": graph_attn,
            "graph_valid_mask": graph_mask,
        }
        return fused, attn_info


class HamNetMIL(nn.Module):


    def __init__(self, encoder: HamNetEncoder, dropout: float = 0.2) -> None:

        super().__init__()
        self.encoder = encoder
        self.hidden_size = encoder.fusion_dim
        self.attn_fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_vec = nn.Linear(self.hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.adapter = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 1),
        )

    def meta_parameters(self):

        for module in (self.attn_fc, self.attn_vec, self.adapter, self.classifier):
            for param in module.parameters():
                if param.requires_grad:
                    yield param

    def mil_pool(
        self, h_funcs: torch.Tensor, bag_idx: torch.Tensor | None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        if bag_idx is None:
            bag_idx = torch.arange(
                h_funcs.size(0), device=h_funcs.device, dtype=torch.long
            )
        if h_funcs.numel() == 0:
            return h_funcs.new_zeros((0, self.hidden_size)), []

        num_bags = int(bag_idx.max().item()) + 1 if bag_idx.numel() > 0 else 0
        scores = self.attn_vec(torch.tanh(self.attn_fc(h_funcs))).squeeze(-1)


        max_per_bag = scores.new_full((num_bags,), float("-inf"))
        if hasattr(max_per_bag, "scatter_reduce_"):
            max_per_bag.scatter_reduce_(
                0, bag_idx, scores, reduce="amax", include_self=True
            )
        else:
            for b in range(num_bags):
                mask = bag_idx == b
                if mask.any():
                    max_per_bag[b] = scores[mask].max()
        stable_scores = scores - max_per_bag[bag_idx]
        exp_scores = torch.exp(stable_scores)
        denom = exp_scores.new_zeros((num_bags,))
        denom.scatter_add_(0, bag_idx, exp_scores)
        weights = exp_scores / denom.clamp_min(1e-12)[bag_idx]
        weights = torch.nan_to_num(weights, nan=0.0)

        bag_repr = h_funcs.new_zeros(num_bags, self.hidden_size)
        bag_repr.index_add_(0, bag_idx, h_funcs * weights.unsqueeze(-1))

        order = torch.argsort(bag_idx)
        sorted_weights = weights[order]
        counts = torch.bincount(bag_idx, minlength=num_bags)
        attn_weights: List[torch.Tensor] = []
        start = 0
        for cnt in counts.tolist():
            end = start + cnt
            attn_weights.append(sorted_weights[start:end])
            start = end

        return bag_repr, attn_weights

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        graphs: Sequence[Dict[str, torch.Tensor]],
        bag_idx: torch.Tensor | None = None,
        ast_seq: torch.Tensor | None = None,
        ast_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        h_funcs, attn_info = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            graphs=graphs,
            ast_seq=ast_seq,
            ast_mask=ast_mask,
        )
        bag_repr, bag_attn = self.mil_pool(h_funcs, bag_idx)
        adapted = bag_repr + self.adapter(self.dropout(bag_repr))
        logits = self.classifier(self.dropout(adapted)).squeeze(-1)
        attn_info.update({"bag_attn": bag_attn})
        return logits, attn_info


class HAMNetModel(HamNetMIL):


    def __init__(
        self,
        encoder_name: str,
        node_vocab_size: int,
        segment_len: int = 32,
        graph_hidden: int = 256,
        dropout: float = 0.2,
        freeze_encoder: bool = False,
        unfreeze_last_n: int = 0,
        use_hier_attn: bool = True,
        use_graph: bool = True,
        use_ast_seq: bool = False,
        semantic_chunk_size: int = 0,
        enable_gradient_checkpointing: bool = False,
    ) -> None:

        encoder = HamNetEncoder(
            encoder_name=encoder_name,
            node_vocab_size=node_vocab_size,
            segment_len=segment_len,
            graph_hidden=graph_hidden,
            dropout=dropout,
            freeze_encoder=freeze_encoder,
            unfreeze_last_n=unfreeze_last_n,
            use_hier_attn=use_hier_attn,
            use_graph=use_graph,
            use_ast_seq=use_ast_seq,
            semantic_chunk_size=semantic_chunk_size,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
        )
        super().__init__(encoder=encoder, dropout=dropout)
