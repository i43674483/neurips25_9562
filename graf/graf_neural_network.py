import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoder(nn.Module):
    """Frequency-based positional encoder using sinusoidal functions."""

    def __init__(self, **config):
        super().__init__()
        self.config = config
        self._build_embedding_functions()

    def _build_embedding_functions(self):
        functions = []
        input_dim = self.config["input_dims"]
        total_output_dim = 0

        if self.config.get("include_input", False):
            functions.append(lambda x: x)
            total_output_dim += input_dim

        num_freqs = self.config["num_freqs"]
        max_log_freq = self.config["max_freq_log2"]
        use_log_sampling = self.config.get("log_sampling", True)

        if use_log_sampling:
            frequencies = 2.0 ** torch.linspace(0.0, max_log_freq, steps=num_freqs)
        else:
            frequencies = torch.linspace(2.0 ** 0.0, 2.0 ** max_log_freq, steps=num_freqs)

        for freq in frequencies:
            for periodic_fn in self.config["periodic_fns"]:
                functions.append(lambda x, fn=periodic_fn, f=freq: fn(x * f))
                total_output_dim += input_dim

        self._embedding_functions = functions
        self.output_dim = total_output_dim

    def forward(self, x):
        return torch.cat([fn(x) for fn in self._embedding_functions], dim=-1)


def build_encoder():
    return PositionalEncoder(
        input_dims=3,
        include_input=True,
        max_freq_log2=9,
        num_freqs=10,
        log_sampling=True,
        periodic_fns=[torch.sin, torch.cos],
    )


class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_prob):
        super().__init__()
        # MLPBlock(input_dim=64, hidden_dim=256, dropout_prob=0.1)

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.activation_fn = nn.ReLU()

    def forward(self, inputs):
        out = self.linear1(inputs)
        out = self.activation_fn(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.dropout(out)

        return out


class SubtractiveAttentionBlock(nn.Module):
    def __init__(self, feature_dim, dropout_prob):
        super().__init__()
        # Query, Key, Value projection layers without bias
        self.to_query = nn.Linear(feature_dim, feature_dim, bias=False)
        self.to_key = nn.Linear(feature_dim, feature_dim, bias=False)
        self.to_value = nn.Linear(feature_dim, feature_dim, bias=False)

        # Positional embedding MLP
        self.position_mlp = nn.Sequential(
            nn.Linear(4, feature_dim // 8),  # 4D relative position vector
            nn.ReLU(),
            nn.Linear(feature_dim // 8, feature_dim)
        )

        # Attention scoring MLP
        self.score_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 8),
            nn.ReLU(),
            nn.Linear(feature_dim // 8, feature_dim)
        )

        self.output_layer = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, query, key, rel_pos, attn_mask=None):
        """
        Args:
            query:      Tensor of shape [B, H, C]         - query vectors
            key:        Tensor of shape [B, H, N, C]      - key/value vectors
            rel_pos:    Tensor of shape [B, H, N, 4]      - relative positional encodings
            attn_mask:  Optional mask [B, H, N, 1]
        Returns:
            Tensor of shape [B, H, C] after attention
        """
        q_proj = self.to_query(query)                     # [B, H, C]
        k_proj = self.to_key(key)                         # [B, H, N, C]
        v_proj = self.to_value(key)                       # [B, H, N, C]

        pos_encoded = self.position_mlp(rel_pos)          # [B, H, N, C]

        # Subtractive attention formulation: k - q + pos
        q_exp = q_proj.unsqueeze(2)                       # [B, H, 1, C]
        attn_input = k_proj - q_exp + pos_encoded         # [B, H, N, C]

        attn_weights = self.score_mlp(attn_input)         # [B, H, N, C]
        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask == 0, -1e9)

        attn_weights = F.softmax(attn_weights, dim=-2)    # softmax across the N dimension
        attn_weights = self.dropout(attn_weights)

        weighted_values = (v_proj + pos_encoded) * attn_weights  # [B, H, N, C]
        aggregated = weighted_values.sum(dim=2)           # [B, H, C]

        output = self.output_layer(aggregated)
        output = self.dropout(output)

        return output


class CrossFeatureTransformer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout_ff, dropout_attn):
        super().__init__()
        """
        Example usage:
        transformer_block = CrossFeatureTransformer(
            embed_dim=64,
            hidden_dim=256,         # 64 * 4
            dropout_ff=0.1,
            dropout_attn=0.1
        )
        """
        self.norm_attn = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm_ff = nn.LayerNorm(embed_dim, eps=1e-6)

        self.feedforward = MLPBlock(embed_dim, hidden_dim, dropout_ff)
        self.attention = SubtractiveAttentionBlock(embed_dim, dropout_attn)

    def forward(self, query_input, key_input, relative_pos, visibility_mask=None):
        """
        Args:
            query_input:     Tensor [B, H, C] — e.g., primary query features
            key_input:       Tensor [B, H, N, C] — e.g., contextual features
            relative_pos:    Tensor [B, H, N, 4] — positional deltas
            visibility_mask: Optional mask [B, H, N, 1]
        Returns:
            Tensor [B, H, C]
        """

        # Attention block with residual
        residual = query_input
        x = self.norm_attn(query_input)
        x = self.attention(x, key_input, relative_pos, visibility_mask)
        x = x + residual

        # Feedforward block with residual
        residual = x
        x = self.norm_ff(x)
        x = self.feedforward(x)
        x = x + residual

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, feature_dim, num_heads, dropout_rate, mode="qk", pos_input_dim=None):
        super().__init__()
        """
        MultiHeadAttention(
            feature_dim=64,
            num_heads=4,
            dropout_rate=0.1,
            mode="qk" or "pos",
            pos_input_dim=4 if mode == "pos"
        )
        """

        self.mode = mode
        self.num_heads = num_heads

        if mode == "qk":
            self.query_proj = nn.Linear(feature_dim, feature_dim, bias=False)
            self.key_proj = nn.Linear(feature_dim, feature_dim, bias=False)

        if mode == "pos":
            assert pos_input_dim is not None, "pos_input_dim must be set for pos mode"
            self.positional_encoder = nn.Sequential(
                nn.Linear(pos_input_dim, pos_input_dim),
                nn.ReLU(),
                nn.Linear(pos_input_dim, feature_dim // 8)
            )
            self.pos_score_proj = nn.Linear(feature_dim // 8, num_heads)

        self.value_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs, pos_info=None, return_attention=False):
        """
        inputs:     Tensor [B, L, C]
        pos_info:   Optional Tensor [B, L, D] (required for "pos" mode)
        return_attention: If True, returns attention weights
        """
        B, L, _ = inputs.shape

        v = self.value_proj(inputs).view(B, L, self.num_heads, -1).permute(0, 2, 1, 3)  # [B, H, L, D]

        if self.mode == "qk":
            q = self.query_proj(inputs).view(B, L, self.num_heads, -1).permute(0, 2, 1, 3)  # [B, H, L, D]
            k = self.key_proj(inputs).view(B, L, self.num_heads, -1).permute(0, 2, 1, 3)  # [B, H, L, D]
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.shape[-1])  # [B, H, L, L]
            attn_weights = torch.softmax(attn_scores, dim=-1)

        elif self.mode == "pos":
            pos_encoded = self.positional_encoder(pos_info)  # [B, L, D]
            pos_diff = pos_encoded[:, :, None, :] - pos_encoded[:, None, :, :]  # [B, L, L, D]
            attn_scores = self.pos_score_proj(pos_diff).permute(0, 3, 1, 2)  # [B, H, L, L]
            attn_weights = torch.softmax(attn_scores, dim=-1)

        attn_weights = self.dropout(attn_weights)

        attended = torch.matmul(attn_weights, v).permute(0, 2, 1, 3).contiguous()  # [B, L, H, D]
        output = attended.view(B, L, -1)  # [B, L, C]
        output = self.dropout(self.output_proj(output))

        if return_attention:
            return output, attn_weights
        else:
            return output


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout_ff, num_heads, dropout_attn, mode="qk", pos_input_dim=None):
        super().__init__()
        """
        Example:
        transformer_layer = AttentionBlock(
            embed_dim=64,
            hidden_dim=256,
            dropout_ff=0.1,
            num_heads=4,
            dropout_attn=0.1,
            mode="qk",
            pos_input_dim=None
        )
        """

        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)

        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout_attn, mode, pos_input_dim)
        self.feed_forward = MLPBlock(embed_dim, hidden_dim, dropout_ff)

    def forward(self, features, position_info=None, return_weights=False):
        """
        Args:
            features: Tensor [B, L, C]
            position_info: Optional tensor [B, L, D] for positional encoding
            return_weights: bool, whether to return attention map

        Returns:
            Tensor [B, L, C], optionally attention scores
        """

        residual = features
        x = self.norm1(features)
        attn_out = self.self_attention(x, position_info, return_weights)

        if return_weights:
            x, attn_scores = attn_out
        else:
            x = attn_out

        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = x + residual

        if return_weights:
            return x, attn_scores.mean(dim=1)[:, 0]  # shape: [B]
        else:
            return x


class ResNetFeature_FC(nn.Module):

    def __init__(self, out_dimension):
        super(ResNetFeature_FC, self).__init__()
        
        # First convolution layer
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
        
        # Second convolution layer
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1))
        
        # Flatten layer is implicit before the fully connected layers
        # Assume we do not change the spatial dimensions in the conv layers
        
        # Calculate the flattened size after convolutions
        self.flattened_size = 24 * 92 * 128
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, out_dimension)  # Final layer with output size 64

    def forward(self, x):
        # Apply convolution layers
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, self.flattened_size)
        
        # Apply fully connected layers
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        # No activation after the last layer if it's a regression task
        # If it's a classification task, you might want to add a softmax or sigmoid, depending on the number of classes
        
        return x



class RadianceFieldTransformer(nn.Module):
    def __init__(self, args, in_channels=32, pos_dim=3, view_dim=3, tx_dim=3, return_attention=False):
        super().__init__()

        self.feature_projector = ResNetFeature_FC(args.netwidth)

        self.view_attention_blocks = nn.ModuleList()
        self.ray_attention_blocks = nn.ModuleList()
        self.feature_mixers = nn.ModuleList()

        for i in range(args.trans_depth):
            self.view_attention_blocks.append(CrossFeatureTransformer(
                embed_dim=args.netwidth,
                hidden_dim=args.netwidth * 4,
                dropout_ff=0.1,
                dropout_attn=0.1
            ))

            self.ray_attention_blocks.append(AttentionBlock(
                embed_dim=args.netwidth,
                hidden_dim=args.netwidth * 4,
                dropout_ff=0.1,
                num_heads=4,
                dropout_attn=0.1
            ))

            if i % 2 == 0:
                self.feature_mixers.append(nn.Sequential(
                    nn.Linear(args.netwidth + pos_dim + view_dim + tx_dim, args.netwidth),
                    nn.ReLU(),
                    nn.Linear(args.netwidth, args.netwidth)
                ))
            else:
                self.feature_mixers.append(nn.Identity())

        self.return_attention = return_attention

        self.encoder_position = build_encoder()
        self.encoder_view = build_encoder()
        self.encoder_tx = build_encoder()

        self.output_norm = nn.LayerNorm(args.netwidth)
        self.output_layer = nn.Linear(args.netwidth, 1)
        
        self.pos_dim = pos_dim
        self.view_dim = view_dim
        self.tx_dim = tx_dim

    def forward(self, rgb_features, ray_offsets, mask, positions, directions, origins):
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)
        view_encoded = self.encoder_view(directions.float().reshape(-1, 3))
        tx_encoded = self.encoder_tx(origins.float().reshape(-1, 3))
        pos_encoded = self.encoder_position(positions.reshape(-1, positions.shape[-1]).float())
        pos_encoded = pos_encoded.view(*positions.shape[:-1], -1)

        view_encoded = view_encoded.unsqueeze(1).expand_as(pos_encoded)
        tx_encoded = tx_encoded.unsqueeze(1).expand_as(pos_encoded)

        composite_input = torch.cat([pos_encoded, view_encoded, tx_encoded], dim=-1)
        pos_part, view_part, tx_part = torch.split(composite_input, [self.pos_dim, self.view_dim, self.tx_dim], dim=-1)

        res_feat = self.feature_projector(rgb_features)  # [N_view, C]
        res_feat = res_feat.repeat(mask.size(0), mask.size(1), 1, 1)

        q = res_feat.max(dim=2)[0]  # Initial ray representation

        for i, (view_attn, mixer, ray_attn) in enumerate(zip(
            self.view_attention_blocks, self.feature_mixers, self.ray_attention_blocks
        )):
            q = view_attn(q, res_feat, ray_offsets, mask)
            if i % 2 == 0:
                q = torch.cat([q, pos_part, view_part, tx_part], dim=-1)
                q = mixer(q)
            q = ray_attn(q, return_weights=self.return_attention)
            if self.return_attention:
                q, attn_map = q

        output_features = self.output_norm(q)
        output = self.output_layer(output_features.mean(dim=1))

        if self.return_attention:
            return torch.cat([output, attn_map], dim=1)
        return output

