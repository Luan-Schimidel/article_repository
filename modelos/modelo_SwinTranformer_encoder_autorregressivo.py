"""
SwinLSTM para previsão espaço-temporal de campos 2D (ex.: altura significativa
de onda - SWH), com correções em relação à versão inicial:

  1. Decoder autoregressivo real: cada passo futuro (t+1, t+2, ..., t+N) é
     gerado a partir do passo anterior, com estado (h, c) propagado — não é
     mais a mesma previsão repetida N vezes.

  2. Atenção manual (sem nn.MultiheadAttention): implementa QKV própria,
     soma a shifted-window mask com o broadcasting correto
     (B_ // nW, nW, heads, N, N) e inclui relative position bias aprendido,
     como no Swin Transformer original.

  3. Skip connection estilo U-Net: o hidden state do encoder (antes do
     patch merging) é preservado e reinjetado no decoder, evitando a perda
     de detalhe fino da simples interpolação bilinear.

  Bônus: input_channels > 1 já é suportado nativamente, então é direto
  incluir variáveis exógenas (ex.: vento u/v da ERA5) como canais extras
  de entrada, mesmo prevendo só 1 canal de saída (SWH).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. WINDOW PARTITION / REVERSE
# ============================================================

def window_partition(x, window_size):
    """
    x: (B, H, W, C) -> (B * num_windows, Ws*Ws, C)
    """
    B, H, W, C = x.shape
    Ws = window_size

    assert H % Ws == 0 and W % Ws == 0, (
        f"H={H}, W={W} precisam ser múltiplos de window_size={Ws}"
    )

    x = x.view(B, H // Ws, Ws, W // Ws, Ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = x.view(-1, Ws * Ws, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    windows: (B*num_windows, Ws*Ws, C) -> (B, H, W, C)
    """
    Ws = window_size
    nh, nw = H // Ws, W // Ws
    B = windows.shape[0] // (nh * nw)

    x = windows.view(B, nh, nw, Ws, Ws, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


# ============================================================
# 2. SHIFTED WINDOW MASK
# ============================================================

def create_shifted_window_mask(H, W, window_size, shift_size, device):
    if shift_size == 0:
        return None

    Ws, Ss = window_size, shift_size

    img_mask = torch.zeros((1, H, W, 1), device=device)

    h_slices = (slice(0, -Ws), slice(-Ws, -Ss), slice(-Ss, None))
    w_slices = (slice(0, -Ws), slice(-Ws, -Ss), slice(-Ss, None))

    count = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = count
            count += 1

    mask_windows = window_partition(img_mask, Ws).squeeze(-1)  # (nW, Ws*Ws)

    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float("-inf"))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)

    return attn_mask  # (nW, N, N)


# ============================================================
# 3. RELATIVE POSITION INDEX (bias por janela, como no Swin original)
# ============================================================

def get_relative_position_index(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
    coords_flatten = torch.flatten(coords, 1)  # (2, Ws*Ws)

    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (N, N, 2)

    relative_coords[:, :, 0] += window_size - 1
    relative_coords[:, :, 1] += window_size - 1
    relative_coords[:, :, 0] *= 2 * window_size - 1

    relative_position_index = relative_coords.sum(-1)  # (N, N)
    return relative_position_index


# ============================================================
# 4. PATCH EMBEDDING / EXPANSION
# ============================================================

class PatchEmbedding(nn.Module):
    """(B, C_in, H, W) -> (B, H/P, W/P, embed_dim)"""

    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x


class PatchExpansion(nn.Module):
    """(B, H/P, W/P, C_in) -> (B, C_out, H, W) — upsampling aprendido, não interpolação fixa."""

    def __init__(self, in_dim, out_channels, patch_size):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_dim, out_channels, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # (B, C, Hp, Wp)
        x = self.deconv(x)
        return x


# ============================================================
# 5. WINDOW ATTENTION (manual, com máscara e relative position bias)
# ============================================================

class WindowAttention(nn.Module):

    def __init__(self, dim, num_heads, window_size, dropout=0.0):
        super().__init__()

        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)

        num_relative_positions = (2 * window_size - 1) ** 2
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_relative_positions, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.register_buffer(
            "relative_position_index",
            get_relative_position_index(window_size),
            persistent=False,
        )

    def forward(self, windows, attn_mask=None):
        """
        windows: (B_, N, C)  onde B_ = B * num_windows, N = window_size**2
        attn_mask: (nW, N, N) ou None
        """
        B_, N, C = windows.shape
        residual = windows

        x = self.norm(windows)

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B_, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B_, heads, N, N)

        # relative position bias
        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)  # (N, N, heads)
        bias = bias.permute(2, 0, 1).contiguous()  # (heads, N, N)
        attn = attn + bias.unsqueeze(0)

        # shifted-window mask, com broadcasting correto por grupo de janelas
        if attn_mask is not None:
            nW = attn_mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(B_, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        out = self.proj_dropout(out)

        return residual + out


# ============================================================
# 6. SWIN BLOCK
# ============================================================

class SwinBlock(nn.Module):

    def __init__(self, dim, num_heads, window_size, shift_size=0, dropout=0.0):
        super().__init__()

        self.window_size = window_size
        self.shift_size = shift_size

        self.attention = WindowAttention(dim, num_heads, window_size, dropout)

        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x, attn_mask=None):
        """x: (B, H, W, C)"""
        B, H, W, C = x.shape

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows = window_partition(x, self.window_size)
        windows = self.attention(windows, attn_mask)
        x = window_reverse(windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        x = residual + x

        return x


# ============================================================
# 7. SWIN LSTM CELL
# ============================================================

class SwinLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_heads=4, window_size=4, dropout=0.0):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.shift_size = window_size // 2

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.swin_block_1 = SwinBlock(hidden_dim, num_heads, window_size, shift_size=0, dropout=dropout)
        self.swin_block_2 = SwinBlock(hidden_dim, num_heads, window_size, shift_size=self.shift_size, dropout=dropout)

        self.gates = nn.Linear(hidden_dim * 2, hidden_dim * 4)

        self._mask_cache = {}

    def _get_mask(self, H, W, device):
        key = (H, W, device)
        if key not in self._mask_cache:
            mask = create_shifted_window_mask(H, W, self.window_size, self.shift_size, device)
            self._mask_cache[key] = mask
        return self._mask_cache[key]

    def forward(self, x, h_prev, c_prev):
        """
        x, h_prev, c_prev: (B, H, W, dim)
        """
        x = self.input_projection(x)
        x = x + h_prev
        H, W = x.shape[1], x.shape[2]
        mask = self._get_mask(H, W, x.device)

        x = self.swin_block_1(x)
        x = self.swin_block_2(x, mask)

        combined = torch.cat([x, h_prev], dim=-1)
        gates = self.gates(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=-1)

        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


# ============================================================
# 8. SWIN LSTM — ENCODER + DECODER AUTOREGRESSIVO COM SKIP CONNECTION
# ============================================================

class Autorregressive_SwinLSTM(nn.Module):

    def __init__(
        self,
        input_channels=1,
        output_channels=1,
        future_steps = 1,
        patch_size=4,
        embed_dim=32,
        hidden_dim=32,
        num_heads=4,
        window_size=4,
        dropout=0.0,
        teacher_forcing_ratio=0.0,
    ):
        """
        input_channels:  nº de canais de entrada (ex.: SWH=1, ou SWH+vento_u+vento_v=3)
        output_channels: nº de canais previstos (normalmente 1: SWH)
        teacher_forcing_ratio: prob. de usar o frame real (em vez da própria
            previsão) como entrada do próximo passo do decoder, durante
            treino. 0.0 = sempre autoregressivo (closed-loop).
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.future_steps = future_steps

        # ---- embeddings (entrada multicanal e saída) ----
        self.patch_embed_input = PatchEmbedding(input_channels, embed_dim, patch_size)
        self.patch_embed_output = PatchEmbedding(output_channels, hidden_dim, patch_size)

        # ---- encoder ----
        self.encoder_cell_1 = SwinLSTMCell(embed_dim, hidden_dim, num_heads, window_size, dropout)
        self.encoder_cell_2 = SwinLSTMCell(embed_dim, hidden_dim, num_heads, window_size, dropout)


        # ---- decoder (parâmetros próprios, não compartilha pesos com o encoder) ----
        self.decoder_cell = SwinLSTMCell(hidden_dim, hidden_dim, num_heads, window_size, dropout)

        # ---- fusão do skip connection (concat h_decoder + skip_encoder -> hidden_dim) ----
        self.skip_fuse = nn.Linear(hidden_dim * 2, hidden_dim)

        # ---- projeção de volta à resolução original ----
        self.patch_expand = PatchExpansion(hidden_dim, output_channels, patch_size)

    def forward(self, x, y_true=None):
        """
        x: (B, T, C_in, H, W)                       — histórico observado
        y_true: (B, future_steps, C_out, H, W) opc.  — usado só p/ teacher forcing no treino

        retorna: (B, future_steps, C_out, H, W)
        """
        B, T, C, H, W = x.shape
        device = x.device

        x0 = self.patch_embed_input(x[:, 0])

        # criando os vetores de célula e de estado h,c com as dimensoes (B,Hp,Wp,embed_dimension)
        Hp, Wp = x0.shape[1], x0.shape[2]

        h = torch.zeros(B, Hp, Wp, self.hidden_dim, device=device)
        c = torch.zeros_like(h)

        # ================= ENCODER =================
        for t in range(T):
            xt = self.patch_embed_input(x[:, t])
            h, c = self.encoder_cell_1(xt, h, c)

        for t in range(T):
            xt = self.patch_embed_input(x[:, t])
            h, c = self.encoder_cell_2(xt, h, c)
        skip = h  # preserva o último hidden state do encoder para o U-Net skip

        # ================= DECODER AUTOREGRESSIVO =================
        # primeira entrada do decoder: último frame observado (nos canais de saída)
        last_channels = min(C, self.output_channels)
        prev_frame = x[:, -1, :last_channels]
        if last_channels < self.output_channels:
            pad = torch.zeros(
                B, self.output_channels - last_channels, H, W, device=device
            )
            prev_frame = torch.cat([prev_frame, pad], dim=1)

        predictions = []

        for step in range(self.future_steps):
            xt = self.patch_embed_output(prev_frame)
            h, c = self.decoder_cell(xt, h, c)

            fused = self.skip_fuse(torch.cat([h, skip], dim=-1))
            frame_pred = self.patch_expand(fused)  # (B, C_out, H, W)

            predictions.append(frame_pred)

            # próxima entrada: própria previsão (closed-loop) ou ground truth
            # (teacher forcing), se disponível durante treino
            use_teacher = (
                self.training
                and y_true is not None
                and torch.rand(1).item() < self.teacher_forcing_ratio
            )
            prev_frame = y_true[:, step] if use_teacher else frame_pred

        predictions = torch.stack(predictions, dim=1)  # (B, future_steps, C_out, H, W)
        return predictions

