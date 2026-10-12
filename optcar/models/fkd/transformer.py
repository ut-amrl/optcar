"""Transformer models for FKD.

This module contains transformer-based models for Forward Kinematics Dynamics learning.
"""

import importlib
from typing import Dict, Optional
import torch
from torch import nn
import torch.nn.functional as F

from .config import FKD_model_config
from .backbone import FKDLightningModule
from .transformer_helper import (
    generate_markov_mask,
    LearnedPositionalEncoding,
    MultiLinear,
    MapEncoder_ViT
)


class TransformerModel_decoderonly(FKDLightningModule):
    """Transformer decoder-only model for FKD."""
    
    def __init__(
        self,
        model_config: FKD_model_config,
        learning_config: Optional[object] = None,
        markov: bool = False,
        **kwargs
    ):
        """Initialize transformer decoder-only model.
        
        Args:
            model_config: Model configuration
            markov: Whether to use Markov mask
            **kwargs: Additional arguments
        """
        super().__init__(model_config=model_config, learning_config=learning_config, **kwargs)

        self.model_config = model_config
        self.learning_config = learning_config
        self.embed_dim = self.model_config.embedding_dims
        self.aleatoric_uncertainty = getattr(self.model_config, "aleatoric_uncertainty", False)

        self.include_twist = getattr(self.model_config, "decoder_include_twist", False)
        self.pose_dim = 7
        self.twist_dim = 6
        self._total_output_dim = self.pose_dim + (self.twist_dim if self.include_twist else 0)
        
        ###################
        #### Encoder ######
        ###################
        total_embeddings = 0
        _concat = getattr(model_config, "concat_state_actions", False)
        _pair   = getattr(model_config, "pair_state_actions",   False)
        if _concat and _pair:
            raise ValueError("concat_state_actions and pair_state_actions are mutually exclusive.")

        if _concat:
            # Single MLP that sees all active per-step streams concatenated.
            concat_input_dim = 0
            if "history_states" in model_config.encoder_types:
                concat_input_dim += model_config.input_state_dims
            if "delta_history_states" in model_config.encoder_types:
                concat_input_dim += model_config.input_state_dims
            # history_actions always included (also shared with decoder below)
            concat_input_dim += model_config.action_dims
            if "concat_history" not in model_config.encoder_types:
                raise ValueError(
                    "concat_state_actions=True requires encoder_types to contain "
                    "a 'concat_history' key with the hidden dims, e.g. [64, 128]."
                )
            self.combined_hist_encoder = MultiLinear(
                self.embed_dim,
                concat_input_dim,
                model_config.encoder_types["concat_history"],
                dropout=model_config.multilinear_dropout,
            )
            total_embeddings += model_config.history_horizon_steps  # one token per step
            # action_encoder still needed for the decoder future-action embedding
            self.action_encoder = MultiLinear(
                self.embed_dim,
                model_config.action_dims,
                self.model_config.encoder_types['history_actions'],
                dropout=model_config.multilinear_dropout,
            )
        else:
            # Default / pair: separate per-stream encoders, each contributing H tokens.
            if "history_states" in self.model_config.encoder_types.keys():
                self.state_encoder = MultiLinear(
                    self.embed_dim,
                    model_config.input_state_dims,
                    self.model_config.encoder_types['history_states'],
                    dropout=model_config.multilinear_dropout
                )
                total_embeddings += self.model_config.history_horizon_steps

            if "delta_history_states" in self.model_config.encoder_types.keys():
                self.delta_state_encoder = MultiLinear(
                    self.embed_dim,
                    model_config.input_state_dims,
                    self.model_config.encoder_types["delta_history_states"],
                    dropout=model_config.multilinear_dropout
                )
                total_embeddings += self.model_config.history_horizon_steps

            if "history_actions" in self.model_config.encoder_types.keys():
                self.action_encoder = MultiLinear(
                    self.embed_dim,
                    model_config.action_dims,
                    self.model_config.encoder_types['history_actions'],
                    dropout=model_config.multilinear_dropout
                )
                total_embeddings += self.model_config.history_horizon_steps
        
        if "history_cmd_vel" in self.model_config.encoder_types.keys() and self.model_config.input_cmd_vel:
            self.cmd_vel_encoder = MultiLinear(
                self.embed_dim,
                model_config.action_dims,
                self.model_config.encoder_types["history_cmd_vel"],
                dropout=model_config.multilinear_dropout
            )
            total_embeddings += self.model_config.history_horizon_steps
        
        if "state_wrt_world" in self.model_config.encoder_types.keys():
            self.curr_state_encoder = MultiLinear(
                self.embed_dim,
                10,
                self.model_config.encoder_types['state_wrt_world'],
                dropout=model_config.multilinear_dropout
            )
            total_embeddings += 1
        
        hmap_key = None
        if "hmap" in self.model_config.encoder_types:
            hmap_key = "hmap"
        elif "map" in self.model_config.encoder_types:
            hmap_key = "map"

        self.hmap_key = hmap_key

        if self.hmap_key is not None:
            self.map_encoder = MapEncoder_ViT(
                self.embed_dim,
                patch_size=self.model_config.encoder_types[self.hmap_key]
            )
            with torch.no_grad():
                map_enc = self.map_encoder(
                    torch.zeros(1, 1, 500, 500, dtype=self._model_dtype)
                )
                _, N, _ = map_enc.shape
            total_embeddings += N
        
        self.combined_encoder_pos_enc = LearnedPositionalEncoding(
            self.embed_dim,
            max_len=total_embeddings,
            flip=getattr(model_config, "history_pos_enc_flip", False),
        )

        ###################
        #### Decoder ######
        ###################
        self.future_action_pos_enc = LearnedPositionalEncoding(
            self.embed_dim,
            max_len=model_config.future_horizon_steps
        )
        
        _decoder_ff = getattr(model_config, "decoder_ff_dim", None) or model_config.ff_dim
        _decoder_nhead = getattr(model_config, "decoder_nhead", None) or model_config.nhead
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=_decoder_nhead,
            dim_feedforward=_decoder_ff,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.model_config.num_layers)
        
        # Causal mask for autoregressive decoding
        T = self.model_config.future_horizon_steps
        causal = torch.ones(T, T, dtype=torch.bool).triu(1)  # upper triangle = True
        self.register_buffer("tgt_causal_mask_cpu", causal, persistent=False)
        
        ##################
        #### FC Out ######
        ##################
        if not self.model_config.decoder_dims:
            raise ValueError("decoder_dims must specify at least one hidden layer size.")
        self.fc_out = MultiLinear(
            self._total_output_dim,
            self.embed_dim,
            self.model_config.decoder_dims,
            dropout=model_config.multilinear_dropout
        )
        
        if self.aleatoric_uncertainty:
            if not self.model_config.decoder_logvar_dims:
                raise ValueError("decoder_logvar_dims must specify at least one hidden layer size.")
            self.fc_out_log_var = MultiLinear(
                self._total_output_dim,
                self.embed_dim,
                self.model_config.decoder_logvar_dims,
                dropout=model_config.multilinear_dropout
            )
    
    def gen_embedding(self, x, encoder):
        """Generate embeddings for input tensor.

        Args:
            x: Input tensor of shape (B, H, n)
            encoder: Encoder module

        Returns:
            Embedded tensor of shape (B, H, embed_dim)
        """
        B, H, n = x.shape
        x_flat = x.reshape(B * H, n)
        x_enc = encoder(x_flat).reshape(B, H, self.embed_dim)
        return x_enc

    def build_encoder_memory(self, embeddings_list: list) -> torch.Tensor:
        """Concat per-stream embeddings, apply positional encoding, return memory.

        Subclasses override this to inject a CTX token or additional processing
        between the per-token MLPs and the decoder cross-attention.
        """
        combined = torch.cat(embeddings_list, dim=1)
        combined = self.combined_encoder_pos_enc(combined)
        return combined

    def get_decoder_memory(self, mem: torch.Tensor) -> torch.Tensor:
        """Select which tokens from mem the decoder cross-attends to.

        Subclasses override this to restrict cross-attention to a subset of
        memory tokens (e.g. CTX-only bottleneck).
        """
        return mem

    def forward(self, batch: Dict[str, torch.Tensor]):
        """Forward pass.
        
        Args:
            batch: Dictionary containing:
                - history: History states (B, H_hist, input_state_dims)
                - history_actions: History actions (B, H_hist, action_dims)
                - future_actions: Future actions (B, H_fut, action_dims)
                - quat_wrt_world: Current quaternion (B, 4)
                - vel_wrt_world: Current velocity (B, 3)
                - angvel_wrt_world: Current angular velocity (B, 3)
                - hmap: Heightmap (B, C, H, W) [optional]
                - delta_history: Delta history states (B, H_hist, input_state_dims) [optional]
                - history_cmd_vel: History command velocity (B, H_hist, 2) [optional]
        
        Returns:
            Dictionary containing:
                - future_traj: Predicted future trajectory (B, H_fut, total_decoder_dim)
                - future_traj_log_var: Log variance (if aleatoric_uncertainty) (B, H_fut, total_decoder_dim)
                - decoder_out: Decoder output (B, H_fut, embed_dim)
        """
        # Pre-processed data
        batch_history: torch.Tensor = batch["history"]
        batch_history_actions: torch.Tensor = batch["history_actions"]
        batch_hmap = batch.get("hmap")
        batch_future_actions: torch.Tensor = batch["future_actions"]
        
        batch_quat_wrt_world: torch.Tensor = batch["quat_wrt_world"]
        batch_vel_wrt_world: torch.Tensor = batch["vel_wrt_world"]
        batch_angvel_wrt_world: torch.Tensor = batch["angvel_wrt_world"]
        
        if self.model_config.input_cmd_vel:
            batch_history_cmd_vel: torch.Tensor = batch["history_cmd_vel"]
        
        # Encode per-step history into memory tokens.
        combined__embeddings = []
        _concat = getattr(self.model_config, "concat_state_actions", False)
        _pair   = getattr(self.model_config, "pair_state_actions",   False)

        if _concat:
            # --- concat_state_actions ---
            # Stack active raw inputs along the feature dim and encode as one token.
            raw_parts: list[torch.Tensor] = []
            if "history_states" in self.model_config.encoder_types:
                raw_parts.append(batch_history)
            if "delta_history_states" in self.model_config.encoder_types:
                raw_parts.append(batch["delta_history"])
            raw_parts.append(batch_history_actions)  # always included
            combined_raw = torch.cat(raw_parts, dim=-1)  # (B, H, sum_dims)
            combined__embeddings.append(
                self.gen_embedding(combined_raw, self.combined_hist_encoder)
            )

        elif _pair:
            # --- pair_state_actions ---
            # Encode each stream separately, then interleave into
            # [s0, d0, a0, s1, d1, a1, ...] so per-step tokens are adjacent.
            per_step_encs: list[torch.Tensor] = []
            if "history_states" in self.model_config.encoder_types:
                per_step_encs.append(self.gen_embedding(batch_history, self.state_encoder))
            if "delta_history_states" in self.model_config.encoder_types:
                per_step_encs.append(
                    self.gen_embedding(batch["delta_history"], self.delta_state_encoder)
                )
            per_step_encs.append(self.gen_embedding(batch_history_actions, self.action_encoder))
            k = len(per_step_encs)
            B_, H_, D_ = per_step_encs[0].shape
            interleaved = torch.empty(
                B_, H_ * k, D_,
                device=per_step_encs[0].device,
                dtype=per_step_encs[0].dtype,
            )
            for i, enc in enumerate(per_step_encs):
                interleaved[:, i::k] = enc
            combined__embeddings.append(interleaved)

        else:
            # --- default: separate token streams ---
            if "history_states" in self.model_config.encoder_types:
                combined__embeddings.append(
                    self.gen_embedding(batch_history, self.state_encoder)
                )
            if "delta_history_states" in self.model_config.encoder_types:
                batch_delta_history: torch.Tensor = batch["delta_history"]
                combined__embeddings.append(
                    self.gen_embedding(batch_delta_history, self.delta_state_encoder)
                )
            # history_actions always included (action_encoder shared with decoder)
            combined__embeddings.append(
                self.gen_embedding(batch_history_actions, self.action_encoder)
            )
        
        if "state_wrt_world" in self.model_config.encoder_types:
            state_wrt_world = torch.cat([
                batch_quat_wrt_world,
                batch_vel_wrt_world,
                batch_angvel_wrt_world
            ], dim=-1)
            state_wrt_world_enc = self.curr_state_encoder(state_wrt_world).unsqueeze(1)
            combined__embeddings.append(state_wrt_world_enc)
        
        if self.hmap_key is not None and batch_hmap is not None:
            map_enc = self.map_encoder(batch_hmap)
            combined__embeddings.append(map_enc)
        
        if "history_cmd_vel" in self.model_config.encoder_types and self.model_config.input_cmd_vel:
            history_cmd_vel_enc = self.gen_embedding(batch_history_cmd_vel, self.cmd_vel_encoder)
            combined__embeddings.append(history_cmd_vel_enc)

        # Input-level token masking: zero out random history tokens before build_encoder_memory
        # so both the base decoder cross-attention and the CTX history_encoder see the masked
        # sequence. Single-token streams (state_wrt_world, map patches) are never masked.
        if self.training:
            min_r = getattr(self.model_config, "history_mask_min_ratio", 0.0)
            max_r = getattr(self.model_config, "history_mask_max_ratio", 0.0)
            if max_r > 0.0:
                for idx, emb in enumerate(combined__embeddings):
                    if emb.shape[1] > 1:  # skip single-token streams
                        B_, N_, _ = emb.shape
                        frac = torch.empty(B_, 1, device=emb.device).uniform_(min_r, max_r)
                        keep = (torch.rand(B_, N_, device=emb.device) >= frac).unsqueeze(-1)
                        combined__embeddings[idx] = emb * keep.to(emb.dtype)

        combined_embeddings = self.build_encoder_memory(combined__embeddings)
        decoder_memory = self.get_decoder_memory(combined_embeddings)

        # Decoder
        future_actions_embedding = self.gen_embedding(batch_future_actions, self.action_encoder)
        future_actions_embedding = self.future_action_pos_enc(future_actions_embedding)

        B, H, n = batch_future_actions.shape

        tgt_mask = self.tgt_causal_mask_cpu[:H, :H].to(future_actions_embedding.device)

        decoder_out = self.decoder(
            tgt=future_actions_embedding,
            memory=decoder_memory,
            tgt_mask=tgt_mask,
            tgt_is_causal=False,
        )

        head_out = self.fc_out(decoder_out.view(B * H, self.embed_dim)).reshape(B, H, self._total_output_dim)

        future_traj = head_out
        
        head_log_var = None
        if self.aleatoric_uncertainty:
            head_log_var = self.fc_out_log_var(
                decoder_out.view(B * H, self.embed_dim)
            ).reshape(B, H, self._total_output_dim)

        return {
            "future_traj": future_traj,
            "future_traj_log_var": head_log_var,
            "decoder_out": decoder_out,
            "component_predictions": future_traj,
            "component_log_var": head_log_var,
        }




class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: projects a conditioning vector to
    per-feature (gamma, beta) that affine-transform the input.

        out = gamma * x + beta

    where gamma and beta are predicted from the conditioning vector.
    gamma is initialised near 1 and beta near 0 so FiLM starts as identity.
    """

    def __init__(self, cond_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, feature_dim * 2)
        # Small non-zero weight so CTX has immediate gradient flow into FiLM.
        # Bias keeps gamma ≈ 1, beta ≈ 0 for stable init.
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            self.proj.bias[:feature_dim].fill_(1.0)  # gamma init

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply FiLM.

        Args:
            x: (B, T, D) — sequence to modulate.
            cond: (B, D_cond) — conditioning vector (e.g. CTX token).
        Returns:
            (B, T, D) modulated sequence.
        """
        gb = self.proj(cond)                          # (B, 2*D)
        gamma, beta = gb.chunk(2, dim=-1)             # each (B, D)
        gamma = gamma.unsqueeze(1)                    # (B, 1, D)
        beta = beta.unsqueeze(1)                      # (B, 1, D)
        return gamma * x + beta


class TransformerModel_decoderonly_CTX(TransformerModel_decoderonly):
    """TransformerModel_decoderonly with a bidirectional history self-attention encoder.

    Inherits all encoders, decoder, and fc_out from the parent. The only addition
    is a learnable [CTX] token + a TransformerEncoder that runs bidirectional
    self-attention over [CTX | history tokens] before the decoder cross-attends.

    The CTX token has no positional encoding — it floats above the sequence and
    aggregates global trajectory properties (mass, inertia, etc.) that are
    constant across all future steps.

    When ``ctx_film=True``, the CTX token is used for FiLM (Feature-wise Linear
    Modulation) conditioning on the decoder instead of being included in the
    cross-attention memory.  In this mode (with ``ctx_only=False``), the history
    tokens feed cross-attention and the CTX vector modulates every decoder layer
    output via learned (gamma, beta) projections.
    """

    def __init__(
        self,
        model_config: FKD_model_config,
        learning_config: Optional[object] = None,
        **kwargs,
    ):
        super().__init__(model_config=model_config, learning_config=learning_config, **kwargs)

        ctx_num_layers = getattr(model_config, "ctx_num_layers", 2)
        self._ctx_film = getattr(model_config, "ctx_film", False)

        # Learnable context token — one per batch item after expand().
        self.ctx_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.ctx_token, std=0.1)

        # Bidirectional self-attention encoder over [CTX | history tokens].
        # Uses the same nhead and ff_dim as the decoder for consistency.
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.model_config.nhead,
            dim_feedforward=self.model_config.ff_dim,
            activation="gelu",
            batch_first=True,
            dropout=0.0,
        )
        self.history_encoder = nn.TransformerEncoder(enc_layer, num_layers=ctx_num_layers)

        # FiLM layers — one per decoder layer.  Only allocated when ctx_film=True.
        # film_layers:       applied after the full decoder layer (post-layer)
        # film_xattn_layers: applied to the query between self-attn and cross-attn
        if self._ctx_film:
            self.film_layers = nn.ModuleList([
                FiLMLayer(self.embed_dim, self.embed_dim)
                for _ in range(self.model_config.num_layers)
            ])
            self.film_xattn_layers = nn.ModuleList([
                FiLMLayer(self.embed_dim, self.embed_dim)
                for _ in range(self.model_config.num_layers)
            ])

    def build_encoder_memory(self, embeddings_list: list) -> torch.Tensor:
        """Concat + pos-encode history, prepend CTX, run bidirectional self-attention."""
        # 1. Concat all per-stream embeddings and apply positional encoding (parent logic).
        combined = torch.cat(embeddings_list, dim=1)          # (B, N, d)
        combined = self.combined_encoder_pos_enc(combined)     # (B, N, d)

        # 2. Prepend the CTX token — no positional encoding on it.
        B = combined.shape[0]
        ctx = self.ctx_token.expand(B, 1, -1)                 # (B, 1, d)
        mem = torch.cat([ctx, combined], dim=1)                # (B, N+1, d)

        # 3. Bidirectional self-attention: CTX attends to all history tokens and
        #    vice versa.  After this, mem[:, 0] holds the global context.
        mem = self.history_encoder(mem)                        # (B, N+1, d)
        return mem

    def get_decoder_memory(self, mem: torch.Tensor) -> torch.Tensor:
        """Return the appropriate memory for decoder cross-attention.

        When ctx_film is active the CTX token is consumed by FiLM layers
        instead of cross-attention, so we strip it from the memory and
        return history-only tokens.  The CTX vector is stashed on ``self``
        for use in the overridden forward().
        """
        if self._ctx_film:
            # Stash CTX vector for FiLM conditioning.
            self._ctx_vec = mem[:, 0]        # (B, d) — used in forward()
            if getattr(self.model_config, "ctx_only", False):
                return mem[:, 0:1]           # (B, 1, d) — CTX only, block history from cross-attn
            return mem[:, 1:]                # (B, N, d) — history tokens + FiLM
        if getattr(self.model_config, "ctx_only", False):
            return mem[:, 0:1]               # (B, 1, d) — forced bottleneck
        return mem                           # (B, N+1, d) — full memory (default)

    def _film_decoder_forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        ctx_vec: torch.Tensor,
    ) -> torch.Tensor:
        """Run decoder layers manually with FiLM applied at two points per layer:
          1. film_xattn: conditions the query between self-attn and cross-attn
          2. film:       conditions the output after the full layer (self-attn + cross-attn + FF)
        """
        output = tgt
        for layer, film, xfilm in zip(self.decoder.layers, self.film_layers, self.film_xattn_layers):
            # --- self-attention (causal) ---
            sa, _ = layer.self_attn(output, output, output, attn_mask=tgt_mask, need_weights=False)
            output = layer.norm1(output + layer.dropout1(sa))

            # --- FiLM on query before cross-attention ---
            output = xfilm(output, ctx_vec)

            # --- cross-attention to CTX memory ---
            ca, _ = layer.multihead_attn(output, memory, memory, need_weights=False)
            output = layer.norm2(output + layer.dropout2(ca))

            # --- feed-forward ---
            ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(output))))
            output = layer.norm3(output + layer.dropout3(ff))

            # --- post-layer FiLM ---
            output = film(output, ctx_vec)

        return output

    # ------------------------------------------------------------------
    # forward() — verbatim copy of TransformerModel_decoderonly.forward()
    # with FiLM decoder path spliced in.
    # ------------------------------------------------------------------
    def forward(self, batch: Dict[str, torch.Tensor]):
        """Forward pass.

        Args:
            batch: Dictionary containing:
                - history: History states (B, H_hist, input_state_dims)
                - history_actions: History actions (B, H_hist, action_dims)
                - future_actions: Future actions (B, H_fut, action_dims)
                - quat_wrt_world: Current quaternion (B, 4)
                - vel_wrt_world: Current velocity (B, 3)
                - angvel_wrt_world: Current angular velocity (B, 3)
                - hmap: Heightmap (B, C, H, W) [optional]
                - delta_history: Delta history states (B, H_hist, input_state_dims) [optional]
                - history_cmd_vel: History command velocity (B, H_hist, 2) [optional]

        Returns:
            Dictionary containing:
                - future_traj: Predicted future trajectory (B, H_fut, total_decoder_dim)
                - future_traj_log_var: Log variance (if aleatoric_uncertainty) (B, H_fut, total_decoder_dim)
                - decoder_out: Decoder output (B, H_fut, embed_dim)
        """
        # Pre-processed data
        batch_history: torch.Tensor = batch["history"]
        batch_history_actions: torch.Tensor = batch["history_actions"]
        batch_hmap = batch.get("hmap")
        batch_future_actions: torch.Tensor = batch["future_actions"]

        batch_quat_wrt_world: torch.Tensor = batch["quat_wrt_world"]
        batch_vel_wrt_world: torch.Tensor = batch["vel_wrt_world"]
        batch_angvel_wrt_world: torch.Tensor = batch["angvel_wrt_world"]

        if self.model_config.input_cmd_vel:
            batch_history_cmd_vel: torch.Tensor = batch["history_cmd_vel"]

        # Encode per-step history into memory tokens.
        combined__embeddings = []
        _concat = getattr(self.model_config, "concat_state_actions", False)
        _pair   = getattr(self.model_config, "pair_state_actions",   False)

        if _concat:
            # --- concat_state_actions ---
            # Stack active raw inputs along the feature dim and encode as one token.
            raw_parts: list[torch.Tensor] = []
            if "history_states" in self.model_config.encoder_types:
                raw_parts.append(batch_history)
            if "delta_history_states" in self.model_config.encoder_types:
                raw_parts.append(batch["delta_history"])
            raw_parts.append(batch_history_actions)  # always included
            combined_raw = torch.cat(raw_parts, dim=-1)  # (B, H, sum_dims)
            combined__embeddings.append(
                self.gen_embedding(combined_raw, self.combined_hist_encoder)
            )

        elif _pair:
            # --- pair_state_actions ---
            # Encode each stream separately, then interleave into
            # [s0, d0, a0, s1, d1, a1, ...] so per-step tokens are adjacent.
            per_step_encs: list[torch.Tensor] = []
            if "history_states" in self.model_config.encoder_types:
                per_step_encs.append(self.gen_embedding(batch_history, self.state_encoder))
            if "delta_history_states" in self.model_config.encoder_types:
                per_step_encs.append(
                    self.gen_embedding(batch["delta_history"], self.delta_state_encoder)
                )
            per_step_encs.append(self.gen_embedding(batch_history_actions, self.action_encoder))
            k = len(per_step_encs)
            B_, H_, D_ = per_step_encs[0].shape
            interleaved = torch.empty(
                B_, H_ * k, D_,
                device=per_step_encs[0].device,
                dtype=per_step_encs[0].dtype,
            )
            for i, enc in enumerate(per_step_encs):
                interleaved[:, i::k] = enc
            combined__embeddings.append(interleaved)

        else:
            # --- default: separate token streams ---
            if "history_states" in self.model_config.encoder_types:
                combined__embeddings.append(
                    self.gen_embedding(batch_history, self.state_encoder)
                )
            if "delta_history_states" in self.model_config.encoder_types:
                batch_delta_history: torch.Tensor = batch["delta_history"]
                combined__embeddings.append(
                    self.gen_embedding(batch_delta_history, self.delta_state_encoder)
                )
            # history_actions always included (action_encoder shared with decoder)
            combined__embeddings.append(
                self.gen_embedding(batch_history_actions, self.action_encoder)
            )

        if "state_wrt_world" in self.model_config.encoder_types:
            state_wrt_world = torch.cat([
                batch_quat_wrt_world,
                batch_vel_wrt_world,
                batch_angvel_wrt_world
            ], dim=-1)
            state_wrt_world_enc = self.curr_state_encoder(state_wrt_world).unsqueeze(1)
            combined__embeddings.append(state_wrt_world_enc)

        if self.hmap_key is not None and batch_hmap is not None:
            map_enc = self.map_encoder(batch_hmap)
            combined__embeddings.append(map_enc)

        if "history_cmd_vel" in self.model_config.encoder_types and self.model_config.input_cmd_vel:
            history_cmd_vel_enc = self.gen_embedding(batch_history_cmd_vel, self.cmd_vel_encoder)
            combined__embeddings.append(history_cmd_vel_enc)

        # Input-level token masking: zero out random history tokens before build_encoder_memory
        # so both the base decoder cross-attention and the CTX history_encoder see the masked
        # sequence. Single-token streams (state_wrt_world, map patches) are never masked.
        if self.training:
            min_r = getattr(self.model_config, "history_mask_min_ratio", 0.0)
            max_r = getattr(self.model_config, "history_mask_max_ratio", 0.0)
            if max_r > 0.0:
                for idx, emb in enumerate(combined__embeddings):
                    if emb.shape[1] > 1:  # skip single-token streams
                        B_, N_, _ = emb.shape
                        frac = torch.empty(B_, 1, device=emb.device).uniform_(min_r, max_r)
                        keep = (torch.rand(B_, N_, device=emb.device) >= frac).unsqueeze(-1)
                        combined__embeddings[idx] = emb * keep.to(emb.dtype)

        combined_embeddings = self.build_encoder_memory(combined__embeddings)
        decoder_memory = self.get_decoder_memory(combined_embeddings)

        # Decoder
        future_actions_embedding = self.gen_embedding(batch_future_actions, self.action_encoder)
        future_actions_embedding = self.future_action_pos_enc(future_actions_embedding)

        B, H, n = batch_future_actions.shape

        tgt_mask = self.tgt_causal_mask_cpu[:H, :H].to(future_actions_embedding.device)

        # ---- FiLM path vs. standard path ----
        if self._ctx_film:
            decoder_out = self._film_decoder_forward(
                tgt=future_actions_embedding,
                memory=decoder_memory,
                tgt_mask=tgt_mask,
                ctx_vec=self._ctx_vec,
            )
        else:
            decoder_out = self.decoder(
                tgt=future_actions_embedding,
                memory=decoder_memory,
                tgt_mask=tgt_mask,
                tgt_is_causal=False,
            )

        head_out = self.fc_out(decoder_out.view(B * H, self.embed_dim)).reshape(B, H, self._total_output_dim)

        future_traj = head_out

        head_log_var = None
        if self.aleatoric_uncertainty:
            head_log_var = self.fc_out_log_var(
                decoder_out.view(B * H, self.embed_dim)
            ).reshape(B, H, self._total_output_dim)

        return {
            "future_traj": future_traj,
            "future_traj_log_var": head_log_var,
            "decoder_out": decoder_out,
            "component_predictions": future_traj,
            "component_log_var": head_log_var,
        }
