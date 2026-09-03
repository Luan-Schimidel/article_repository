import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. WINDOW PARTITION / REVERSE
# ============================================================

def window_partition(x, window_size):
    """
    x: (B, H, W, C)

    retorna:
    windows: (B * num_windows, Ws*Ws, C)
    """

    B, H, W, C = x.shape
    Ws = window_size

    assert H % Ws == 0 and W % Ws == 0, (
        f"H={H}, W={W} precisam ser múltiplos "
        f"de window_size={Ws}"
    )

    x = x.view(
        B,
        H // Ws,
        Ws,
        W // Ws,
        Ws,
        C
    )

    x = x.permute(
        0, 1, 3, 2, 4, 5
    ).contiguous()

    windows = x.view(
        -1,
        Ws * Ws,
        C
    )

    return windows


def window_reverse(windows, window_size, H, W):
    """
    windows:
        (B*num_windows, Ws*Ws, C)

    retorna:
        (B, H, W, C)
    """

    Ws = window_size

    nh = H // Ws
    nw = W // Ws

    B = windows.shape[0] // (nh * nw)

    x = windows.view(
        B,
        nh,
        nw,
        Ws,
        Ws,
        -1
    )

    x = x.permute(
        0,
        1,
        3,
        2,
        4,
        5
    ).contiguous()

    x = x.view(
        B,
        H,
        W,
        -1
    )

    return x


# ============================================================
# 2. SHIFTED WINDOW MASK
# ============================================================

def create_shifted_window_mask(
    H,
    W,
    window_size,
    shift_size,
    device
):

    if shift_size == 0:
        return None

    Ws = window_size
    Ss = shift_size

    img_mask = torch.zeros(
        (1, H, W, 1),
        device=device
    )

    h_slices = (
        slice(0, -Ws),
        slice(-Ws, -Ss),
        slice(-Ss, None)
    )

    w_slices = (
        slice(0, -Ws),
        slice(-Ws, -Ss),
        slice(-Ss, None)
    )

    count = 0

    for h in h_slices:
        for w in w_slices:

            img_mask[:, h, w, :] = count
            count += 1

    mask_windows = window_partition(
        img_mask,
        Ws
    ).squeeze(-1)

    attn_mask = (
        mask_windows.unsqueeze(1)
        -
        mask_windows.unsqueeze(2)
    )

    attn_mask = attn_mask.masked_fill(
        attn_mask != 0,
        float("-inf")
    )

    attn_mask = attn_mask.masked_fill(
        attn_mask == 0,
        0.0
    )

    return attn_mask


# ============================================================
# 3. RELATIVE POSITION INDEX
# ============================================================

def get_relative_position_index(window_size):

    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)

    coords = torch.stack(
        torch.meshgrid(
            coords_h,
            coords_w,
            indexing="ij"
        )
    )

    coords_flatten = torch.flatten(
        coords,
        1
    )

    relative_coords = (
        coords_flatten[:, :, None]
        -
        coords_flatten[:, None, :]
    )

    relative_coords = relative_coords.permute(
        1,
        2,
        0
    ).contiguous()

    relative_coords[:, :, 0] += window_size - 1
    relative_coords[:, :, 1] += window_size - 1

    relative_coords[:, :, 0] *= (
        2 * window_size - 1
    )

    relative_position_index = (
        relative_coords.sum(-1)
    )

    return relative_position_index


# ============================================================
# 4. PATCH EMBEDDING
# ============================================================

class PatchEmbedding(nn.Module):

    """
    (B, C_in, H, W)
        ->
    (B, H/P, W/P, embed_dim)
    """

    def __init__(
        self,
        in_channels,
        embed_dim,
        patch_size
    ):

        super().__init__()

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):

        x = self.proj(x)

        x = x.permute(
            0,
            2,
            3,
            1
        )

        return x


# ============================================================
# 5. PATCH EXPANSION
# ============================================================

class PatchExpansion(nn.Module):

    """
    (B, Hp, Wp, C)
        ->
    (B, C_out, H, W)
    """

    def __init__(
        self,
        in_dim,
        out_channels,
        patch_size
    ):

        super().__init__()

        self.deconv = nn.ConvTranspose2d(
            in_dim,
            out_channels,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):

        x = x.permute(
            0,
            3,
            1,
            2
        )

        x = self.deconv(x)

        return x


# ============================================================
# 6. WINDOW ATTENTION
# ============================================================

class WindowAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        dropout=0.0
    ):

        super().__init__()

        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.scale = self.head_dim ** -0.5

        self.window_size = window_size

        self.norm = nn.LayerNorm(dim)

        self.qkv = nn.Linear(
            dim,
            dim * 3,
            bias=True
        )

        self.attn_dropout = nn.Dropout(dropout)

        self.proj = nn.Linear(
            dim,
            dim
        )

        self.proj_dropout = nn.Dropout(dropout)

        num_relative_positions = (
            2 * window_size - 1
        ) ** 2

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                num_relative_positions,
                num_heads
            )
        )

        nn.init.trunc_normal_(
            self.relative_position_bias_table,
            std=0.02
        )

        self.register_buffer(
            "relative_position_index",
            get_relative_position_index(
                window_size
            ),
            persistent=False
        )

    def forward(
        self,
        windows,
        attn_mask=None
    ):

        B_, N, C = windows.shape

        residual = windows

        x = self.norm(windows)

        qkv = self.qkv(x).reshape(
            B_,
            N,
            3,
            self.num_heads,
            self.head_dim
        )

        qkv = qkv.permute(
            2,
            0,
            3,
            1,
            4
        )

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (
            q @ k.transpose(-2, -1)
        ) * self.scale

        # Relative position bias

        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ]

        bias = bias.view(
            N,
            N,
            -1
        )

        bias = bias.permute(
            2,
            0,
            1
        ).contiguous()

        attn = attn + bias.unsqueeze(0)

        # Shifted-window mask

        if attn_mask is not None:

            nW = attn_mask.shape[0]

            attn = attn.view(
                B_ // nW,
                nW,
                self.num_heads,
                N,
                N
            )

            attn = (
                attn
                +
                attn_mask.unsqueeze(1).unsqueeze(0)
            )

            attn = attn.view(
                B_,
                self.num_heads,
                N,
                N
            )

        attn = attn.softmax(
            dim=-1
        )

        attn = self.attn_dropout(
            attn
        )

        out = (
            attn @ v
        ).transpose(
            1,
            2
        ).reshape(
            B_,
            N,
            C
        )

        out = self.proj(out)

        out = self.proj_dropout(out)

        return residual + out


# ============================================================
# 7. SWIN BLOCK
# ============================================================

class SwinBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        shift_size=0,
        dropout=0.0
    ):

        super().__init__()

        self.window_size = window_size
        self.shift_size = shift_size

        self.attention = WindowAttention(
            dim,
            num_heads,
            window_size,
            dropout
        )

        self.norm = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(

            nn.Linear(
                dim,
                4 * dim
            ),

            nn.GELU(),

            nn.Linear(
                4 * dim,
                dim
            )
        )

    def forward(
        self,
        x,
        attn_mask=None
    ):

        """
        x:
            (B, H, W, C)
        """

        B, H, W, C = x.shape

        # Shift

        if self.shift_size > 0:

            x = torch.roll(
                x,
                shifts=(
                    -self.shift_size,
                    -self.shift_size
                ),
                dims=(1, 2)
            )

        # Window partition

        windows = window_partition(
            x,
            self.window_size
        )

        # Attention

        windows = self.attention(
            windows,
            attn_mask
        )

        # Reverse

        x = window_reverse(
            windows,
            self.window_size,
            H,
            W
        )

        # Reverse shift

        if self.shift_size > 0:

            x = torch.roll(
                x,
                shifts=(
                    self.shift_size,
                    self.shift_size
                ),
                dims=(1, 2)
            )

        # MLP

        residual = x

        x = self.norm(x)

        x = self.mlp(x)

        x = residual + x

        return x


# ============================================================
# 8. SWIN LSTM CELL
# ============================================================

class SwinLSTMCell(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_heads=4,
        window_size=4,
        dropout=0.0
    ):

        super().__init__()

        self.hidden_dim = hidden_dim

        self.window_size = window_size

        self.shift_size = window_size // 2

        # Entrada -> dimensão escondida

        self.input_projection = nn.Linear(
            input_dim,
            hidden_dim
        )

        # Swin

        self.swin_block_1 = SwinBlock(
            hidden_dim,
            num_heads,
            window_size,
            shift_size=0,
            dropout=dropout
        )

        self.swin_block_2 = SwinBlock(
            hidden_dim,
            num_heads,
            window_size,
            shift_size=self.shift_size,
            dropout=dropout
        )

        # LSTM gates

        self.gates = nn.Linear(
            hidden_dim * 2,
            hidden_dim * 4
        )

        self._mask_cache = {}

    def _get_mask(
        self,
        H,
        W,
        device
    ):

        key = (
            H,
            W,
            device
        )

        if key not in self._mask_cache:

            mask = create_shifted_window_mask(
                H,
                W,
                self.window_size,
                self.shift_size,
                device
            )

            self._mask_cache[key] = mask

        return self._mask_cache[key]

    def forward(
        self,
        x,
        h_prev,
        c_prev
    ):

        """
        x:
            (B, H, W, input_dim)

        h_prev:
            (B, H, W, hidden_dim)

        c_prev:
            (B, H, W, hidden_dim)
        """

        x = self.input_projection(x)

        # Combinação entrada + estado anterior

        x = x + h_prev

        H, W = x.shape[1], x.shape[2]

        mask = self._get_mask(
            H,
            W,
            x.device
        )

        # Swin

        x = self.swin_block_1(x)

        x = self.swin_block_2(
            x,
            mask
        )

        # LSTM gates

        combined = torch.cat(
            [x, h_prev],
            dim=-1
        )

        gates = self.gates(
            combined
        )

        i, f, o, g = torch.chunk(
            gates,
            4,
            dim=-1
        )

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)

        g = torch.tanh(g)

        # Cell state

        c_next = (
            f * c_prev
            +
            i * g
        )

        # Hidden state

        h_next = (
            o *
            torch.tanh(c_next)
        )

        return h_next, c_next


# ============================================================
# 9. SINGLE-SHOT SWIN DECODER
# ============================================================

class SingleShotSwinDecoder(nn.Module):

    """
    Decoder NÃO autoregressivo.

    Entrada:
        h:
            (B, Hp, Wp, hidden_dim)

    Saída:
        predictions:
            (B, future_steps, C_out, H, W)
    """

    def __init__(
        self,
        hidden_dim,
        output_channels,
        future_steps,
        patch_size,
        num_heads,
        window_size,
        dropout=0.0
    ):

        super().__init__()

        self.hidden_dim = hidden_dim

        self.future_steps = future_steps

        self.output_channels = output_channels

        self.patch_size = patch_size

        # ----------------------------------------------------
        # Projeção do estado final para todos os futuros
        # ----------------------------------------------------

        self.future_projection = nn.Linear(
            hidden_dim,
            future_steps * hidden_dim
        )

        # ----------------------------------------------------
        # Embedding temporal
        #
        # Cada futuro possui uma representação diferente:
        #
        # t+1 -> E[0]
        # t+2 -> E[1]
        # ...
        # t+K -> E[K-1]
        # ----------------------------------------------------

        self.temporal_embedding = nn.Parameter(
            torch.zeros(
                1,
                future_steps,
                1,
                1,
                hidden_dim
            )
        )

        nn.init.trunc_normal_(
            self.temporal_embedding,
            std=0.02
        )

        # ----------------------------------------------------
        # Swin blocks
        # ----------------------------------------------------

        self.swin_block_1 = SwinBlock(
            hidden_dim,
            num_heads,
            window_size,
            shift_size=0,
            dropout=dropout
        )

        self.swin_block_2 = SwinBlock(
            hidden_dim,
            num_heads,
            window_size,
            shift_size=window_size // 2,
            dropout=dropout
        )

        # ----------------------------------------------------
        # Saída
        # ----------------------------------------------------

        self.patch_expand = PatchExpansion(
            hidden_dim,
            output_channels,
            patch_size
        )

    def forward(
        self,
        h,
        skip=None
    ):

        """
        h:
            (B, Hp, Wp, hidden_dim)

        skip:
            (B, Hp, Wp, hidden_dim)

        retorna:
            (B, K, C_out, H, W)
        """

        B, Hp, Wp, D = h.shape

        # ----------------------------------------------------
        # 1. Gerar todas as representações futuras
        # ----------------------------------------------------

        future = self.future_projection(h)

        future = future.view(
            B,
            Hp,
            Wp,
            self.future_steps,
            D
        )

        future = future.permute(
            0,
            3,
            1,
            2,
            4
        )

        # Agora:

        # future:
        # (B, K, Hp, Wp, D)

        # ----------------------------------------------------
        # 2. Adicionar informação temporal
        # ----------------------------------------------------

        future = (
            future
            +
            self.temporal_embedding
        )

        # ----------------------------------------------------
        # 3. Skip connection
        #
        # O skip é o estado final do encoder.
        # Ele é usado para todos os horizontes,
        # mas NÃO existe realimentação temporal.
        # ----------------------------------------------------

        if skip is not None:

            skip = skip.unsqueeze(1)

            future = future + skip

        # ----------------------------------------------------
        # 4. Processar os K futuros simultaneamente
        #
        # Transformamos:
        #
        # (B, K, Hp, Wp, D)
        #
        # em:
        #
        # (B*K, Hp, Wp, D)
        #
        # Portanto os K futuros são processados em
        # uma única chamada.
        # ----------------------------------------------------

        future = future.reshape(
            B * self.future_steps,
            Hp,
            Wp,
            D
        )

        # ----------------------------------------------------
        # 5. Máscara espacial do Swin
        # ----------------------------------------------------

        mask = create_shifted_window_mask(
            Hp,
            Wp,
            self.swin_block_2.window_size,
            self.swin_block_2.shift_size,
            future.device
        )

        # ----------------------------------------------------
        # 6. Swin
        # ----------------------------------------------------

        future = self.swin_block_1(
            future
        )

        future = self.swin_block_2(
            future,
            mask
        )

        # ----------------------------------------------------
        # 7. Patch expansion
        # ----------------------------------------------------

        frames = self.patch_expand(
            future
        )

        # frames:

        # (B*K, C_out, H, W)

        # ----------------------------------------------------
        # 8. Recuperar dimensão temporal
        # ----------------------------------------------------

        frames = frames.view(
            B,
            self.future_steps,
            self.output_channels,
            frames.shape[-2],
            frames.shape[-1]
        )

        return frames


# ============================================================
# 10. SWIN LSTM
#     ENCODER + SINGLE-SHOT DECODER
# ============================================================

class SwinLSTM(nn.Module):

    def __init__(
        self,
        input_channels=1,
        output_channels=1,
        patch_size=4,
        embed_dim=32,
        hidden_dim=32,
        num_heads=4,
        window_size=4,
        future_steps=5,
        dropout=0.0
    ):

        super().__init__()

        self.hidden_dim = hidden_dim

        self.future_steps = future_steps

        # ====================================================
        # PATCH EMBEDDING
        # ====================================================

        self.patch_embed_input = PatchEmbedding(
            input_channels,
            embed_dim,
            patch_size
        )

        # ====================================================
        # ENCODER
        # ====================================================

        self.encoder_cell = SwinLSTMCell(
            embed_dim,
            hidden_dim,
            num_heads,
            window_size,
            dropout
        )

        # ====================================================
        # DECODER SINGLE-SHOT
        # ====================================================

        self.decoder = SingleShotSwinDecoder(
            hidden_dim=hidden_dim,
            output_channels=output_channels,
            future_steps=future_steps,
            patch_size=patch_size,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout
        )
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. WINDOW PARTITION / REVERSE
# ============================================================

def window_partition(x, window_size):
    """
    x: (B, H, W, C)

    retorna:
    windows: (B * num_windows, Ws*Ws, C)
    """

    B, H, W, C = x.shape
    Ws = window_size

    assert H % Ws == 0 and W % Ws == 0, (
        f"H={H}, W={W} precisam ser múltiplos "
        f"de window_size={Ws}"
    )

    x = x.view(
        B,
        H // Ws,
        Ws,
        W // Ws,
        Ws,
        C
    )

    x = x.permute(
        0, 1, 3, 2, 4, 5
    ).contiguous()

    windows = x.view(
        -1,
        Ws * Ws,
        C
    )

    return windows


def window_reverse(windows, window_size, H, W):
    """
    windows:
        (B*num_windows, Ws*Ws, C)

    retorna:
        (B, H, W, C)
    """

    Ws = window_size

    nh = H // Ws
    nw = W // Ws

    B = windows.shape[0] // (nh * nw)

    x = windows.view(
        B,
        nh,
        nw,
        Ws,
        Ws,
        -1
    )

    x = x.permute(
        0,
        1,
        3,
        2,
        4,
        5
    ).contiguous()

    x = x.view(
        B,
        H,
        W,
        -1
    )

    return x


# ============================================================
# 2. SHIFTED WINDOW MASK
# ============================================================

def create_shifted_window_mask(
    H,
    W,
    window_size,
    shift_size,
    device
):

    if shift_size == 0:
        return None

    Ws = window_size
    Ss = shift_size

    img_mask = torch.zeros(
        (1, H, W, 1),
        device=device
    )

    h_slices = (
        slice(0, -Ws),
        slice(-Ws, -Ss),
        slice(-Ss, None)
    )

    w_slices = (
        slice(0, -Ws),
        slice(-Ws, -Ss),
        slice(-Ss, None)
    )

    count = 0

    for h in h_slices:
        for w in w_slices:

            img_mask[:, h, w, :] = count
            count += 1

    mask_windows = window_partition(
        img_mask,
        Ws
    ).squeeze(-1)

    attn_mask = (
        mask_windows.unsqueeze(1)
        -
        mask_windows.unsqueeze(2)
    )

    attn_mask = attn_mask.masked_fill(
        attn_mask != 0,
        float("-inf")
    )

    attn_mask = attn_mask.masked_fill(
        attn_mask == 0,
        0.0
    )

    return attn_mask


# ============================================================
# 3. RELATIVE POSITION INDEX
# ============================================================

def get_relative_position_index(window_size):

    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)

    coords = torch.stack(
        torch.meshgrid(
            coords_h,
            coords_w,
            indexing="ij"
        )
    )

    coords_flatten = torch.flatten(
        coords,
        1
    )

    relative_coords = (
        coords_flatten[:, :, None]
        -
        coords_flatten[:, None, :]
    )

    relative_coords = relative_coords.permute(
        1,
        2,
        0
    ).contiguous()

    relative_coords[:, :, 0] += window_size - 1
    relative_coords[:, :, 1] += window_size - 1

    relative_coords[:, :, 0] *= (
        2 * window_size - 1
    )

    relative_position_index = (
        relative_coords.sum(-1)
    )

    return relative_position_index


# ============================================================
# 4. PATCH EMBEDDING
# ============================================================

class PatchEmbedding(nn.Module):

    """
    (B, C_in, H, W)
        ->
    (B, H/P, W/P, embed_dim)
    """

    def __init__(
        self,
        in_channels,
        embed_dim,
        patch_size
    ):

        super().__init__()

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):

        x = self.proj(x)

        x = x.permute(
            0,
            2,
            3,
            1
        )

        return x


# ============================================================
# 5. PATCH EXPANSION
# ============================================================

class PatchExpansion(nn.Module):

    """
    (B, Hp, Wp, C)
        ->
    (B, C_out, H, W)
    """

    def __init__(
        self,
        in_dim,
        out_channels,
        patch_size
    ):

        super().__init__()

        self.deconv = nn.ConvTranspose2d(
            in_dim,
            out_channels,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):

        x = x.permute(
            0,
            3,
            1,
            2
        )

        x = self.deconv(x)

        return x


# ============================================================
# 6. WINDOW ATTENTION
# ============================================================

class WindowAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        dropout=0.0
    ):

        super().__init__()

        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.scale = self.head_dim ** -0.5

        self.window_size = window_size

        self.norm = nn.LayerNorm(dim)

        self.qkv = nn.Linear(
            dim,
            dim * 3,
            bias=True
        )

        self.attn_dropout = nn.Dropout(dropout)

        self.proj = nn.Linear(
            dim,
            dim
        )

        self.proj_dropout = nn.Dropout(dropout)

        num_relative_positions = (
            2 * window_size - 1
        ) ** 2

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                num_relative_positions,
                num_heads
            )
        )

        nn.init.trunc_normal_(
            self.relative_position_bias_table,
            std=0.02
        )

        self.register_buffer(
            "relative_position_index",
            get_relative_position_index(
                window_size
            ),
            persistent=False
        )

    def forward(
        self,
        windows,
        attn_mask=None
    ):

        B_, N, C = windows.shape

        residual = windows

        x = self.norm(windows)

        qkv = self.qkv(x).reshape(
            B_,
            N,
            3,
            self.num_heads,
            self.head_dim
        )

        qkv = qkv.permute(
            2,
            0,
            3,
            1,
            4
        )

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (
            q @ k.transpose(-2, -1)
        ) * self.scale

        # Relative position bias

        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ]

        bias = bias.view(
            N,
            N,
            -1
        )

        bias = bias.permute(
            2,
            0,
            1
        ).contiguous()

        attn = attn + bias.unsqueeze(0)

        # Shifted-window mask

        if attn_mask is not None:

            nW = attn_mask.shape[0]

            attn = attn.view(
                B_ // nW,
                nW,
                self.num_heads,
                N,
                N
            )

            attn = (
                attn
                +
                attn_mask.unsqueeze(1).unsqueeze(0)
            )

            attn = attn.view(
                B_,
                self.num_heads,
                N,
                N
            )

        attn = attn.softmax(
            dim=-1
        )

        attn = self.attn_dropout(
            attn
        )

        out = (
            attn @ v
        ).transpose(
            1,
            2
        ).reshape(
            B_,
            N,
            C
        )

        out = self.proj(out)

        out = self.proj_dropout(out)

        return residual + out


# ============================================================
# 7. SWIN BLOCK
# ============================================================

class SwinBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        shift_size=0,
        dropout=0.0
    ):

        super().__init__()

        self.window_size = window_size
        self.shift_size = shift_size

        self.attention = WindowAttention(
            dim,
            num_heads,
            window_size,
            dropout
        )

        self.norm = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(

            nn.Linear(
                dim,
                4 * dim
            ),

            nn.GELU(),

            nn.Linear(
                4 * dim,
                dim
            )
        )

    def forward(
        self,
        x,
        attn_mask=None
    ):

        """
        x:
            (B, H, W, C)
        """

        B, H, W, C = x.shape

        # Shift

        if self.shift_size > 0:

            x = torch.roll(
                x,
                shifts=(
                    -self.shift_size,
                    -self.shift_size
                ),
                dims=(1, 2)
            )

        # Window partition

        windows = window_partition(
            x,
            self.window_size
        )

        # Attention

        windows = self.attention(
            windows,
            attn_mask
        )

        # Reverse

        x = window_reverse(
            windows,
            self.window_size,
            H,
            W
        )

        # Reverse shift

        if self.shift_size > 0:

            x = torch.roll(
                x,
                shifts=(
                    self.shift_size,
                    self.shift_size
                ),
                dims=(1, 2)
            )

        # MLP

        residual = x

        x = self.norm(x)

        x = self.mlp(x)

        x = residual + x

        return x


# ============================================================
# 8. SWIN LSTM CELL
# ============================================================

class SwinLSTMCell(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_heads=4,
        window_size=4,
        dropout=0.0
    ):

        super().__init__()

        self.hidden_dim = hidden_dim

        self.window_size = window_size

        self.shift_size = window_size // 2

        # Entrada -> dimensão escondida

        self.input_projection = nn.Linear(
            input_dim,
            hidden_dim
        )

        # Swin

        self.swin_block_1 = SwinBlock(
            hidden_dim,
            num_heads,
            window_size,
            shift_size=0,
            dropout=dropout
        )

        self.swin_block_2 = SwinBlock(
            hidden_dim,
            num_heads,
            window_size,
            shift_size=self.shift_size,
            dropout=dropout
        )

        # LSTM gates

        self.gates = nn.Linear(
            hidden_dim * 2,
            hidden_dim * 4
        )

        self._mask_cache = {}

    def _get_mask(
        self,
        H,
        W,
        device
    ):

        key = (
            H,
            W,
            device
        )

        if key not in self._mask_cache:

            mask = create_shifted_window_mask(
                H,
                W,
                self.window_size,
                self.shift_size,
                device
            )

            self._mask_cache[key] = mask

        return self._mask_cache[key]

    def forward(
        self,
        x,
        h_prev,
        c_prev
    ):

        """
        x:
            (B, H, W, input_dim)

        h_prev:
            (B, H, W, hidden_dim)

        c_prev:
            (B, H, W, hidden_dim)
        """

        x = self.input_projection(x)

        # Combinação entrada + estado anterior

        x = x + h_prev

        H, W = x.shape[1], x.shape[2]

        mask = self._get_mask(
            H,
            W,
            x.device
        )

        # Swin

        x = self.swin_block_1(x)

        x = self.swin_block_2(
            x,
            mask
        )

        # LSTM gates

        combined = torch.cat(
            [x, h_prev],
            dim=-1
        )

        gates = self.gates(
            combined
        )

        i, f, o, g = torch.chunk(
            gates,
            4,
            dim=-1
        )

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)

        g = torch.tanh(g)

        # Cell state

        c_next = (
            f * c_prev
            +
            i * g
        )

        # Hidden state

        h_next = (
            o *
            torch.tanh(c_next)
        )

        return h_next, c_next


# ============================================================
# 9. SINGLE-SHOT SWIN DECODER
# ============================================================

class SingleShotSwinDecoder(nn.Module):

    """
    Decoder NÃO autoregressivo.

    Entrada:
        h:
            (B, Hp, Wp, hidden_dim)

    Saída:
        predictions:
            (B, future_steps, C_out, H, W)
    """

    def __init__(
        self,
        hidden_dim,
        output_channels,
        future_steps,
        patch_size,
        num_heads,
        window_size,
        dropout=0.0
    ):

        super().__init__()

        self.hidden_dim = hidden_dim

        self.future_steps = future_steps

        self.output_channels = output_channels

        self.patch_size = patch_size

        # ----------------------------------------------------
        # Projeção do estado final para todos os futuros
        # ----------------------------------------------------

        self.future_projection = nn.Linear(
            hidden_dim,
            future_steps * hidden_dim
        )

        # ----------------------------------------------------
        # Embedding temporal
        #
        # Cada futuro possui uma representação diferente:
        #
        # t+1 -> E[0]
        # t+2 -> E[1]
        # ...
        # t+K -> E[K-1]
        # ----------------------------------------------------

        self.temporal_embedding = nn.Parameter(
            torch.zeros(
                1,
                future_steps,
                1,
                1,
                hidden_dim
            )
        )

        nn.init.trunc_normal_(
            self.temporal_embedding,
            std=0.02
        )

        # ----------------------------------------------------
        # Swin blocks
        # ----------------------------------------------------

        self.swin_block_1 = SwinBlock(
            hidden_dim,
            num_heads,
            window_size,
            shift_size=0,
            dropout=dropout
        )

        self.swin_block_2 = SwinBlock(
            hidden_dim,
            num_heads,
            window_size,
            shift_size=window_size // 2,
            dropout=dropout
        )

        # ----------------------------------------------------
        # Saída
        # ----------------------------------------------------

        self.patch_expand = PatchExpansion(
            hidden_dim,
            output_channels,
            patch_size
        )

    def forward(
        self,
        h,
        skip=None
    ):

        """
        h:
            (B, Hp, Wp, hidden_dim)

        skip:
            (B, Hp, Wp, hidden_dim)

        retorna:
            (B, K, C_out, H, W)
        """

        B, Hp, Wp, D = h.shape

        # ----------------------------------------------------
        # 1. Gerar todas as representações futuras
        # ----------------------------------------------------

        future = self.future_projection(h)

        future = future.view(
            B,
            Hp,
            Wp,
            self.future_steps,
            D
        )

        future = future.permute(
            0,
            3,
            1,
            2,
            4
        )

        # Agora:

        # future:
        # (B, K, Hp, Wp, D)

        # ----------------------------------------------------
        # 2. Adicionar informação temporal
        # ----------------------------------------------------

        future = (
            future
            +
            self.temporal_embedding
        )

        # ----------------------------------------------------
        # 3. Skip connection
        #
        # O skip é o estado final do encoder.
        # Ele é usado para todos os horizontes,
        # mas NÃO existe realimentação temporal.
        # ----------------------------------------------------

        if skip is not None:

            skip = skip.unsqueeze(1)

            future = future + skip

        # ----------------------------------------------------
        # 4. Processar os K futuros simultaneamente
        #
        # Transformamos:
        #
        # (B, K, Hp, Wp, D)
        #
        # em:
        #
        # (B*K, Hp, Wp, D)
        #
        # Portanto os K futuros são processados em
        # uma única chamada.
        # ----------------------------------------------------

        future = future.reshape(
            B * self.future_steps,
            Hp,
            Wp,
            D
        )

        # ----------------------------------------------------
        # 5. Máscara espacial do Swin
        # ----------------------------------------------------

        mask = create_shifted_window_mask(
            Hp,
            Wp,
            self.swin_block_2.window_size,
            self.swin_block_2.shift_size,
            future.device
        )

        # ----------------------------------------------------
        # 6. Swin
        # ----------------------------------------------------

        future = self.swin_block_1(
            future
        )

        future = self.swin_block_2(
            future,
            mask
        )

        # ----------------------------------------------------
        # 7. Patch expansion
        # ----------------------------------------------------

        frames = self.patch_expand(
            future
        )

        # frames:

        # (B*K, C_out, H, W)

        # ----------------------------------------------------
        # 8. Recuperar dimensão temporal
        # ----------------------------------------------------

        frames = frames.view(
            B,
            self.future_steps,
            self.output_channels,
            frames.shape[-2],
            frames.shape[-1]
        )

        return frames


# ============================================================
# 10. SWIN LSTM
#     ENCODER + SINGLE-SHOT DECODER
# ============================================================

class SwinLSTM(nn.Module):

    def __init__(
        self,
        input_channels=1,
        output_channels=1,
        patch_size=4,
        embed_dim=32,
        hidden_dim=32,
        num_heads=4,
        window_size=4,
        future_steps=5,
        dropout=0.0
    ):

        super().__init__()

        self.hidden_dim = hidden_dim

        self.future_steps = future_steps

        # ====================================================
        # PATCH EMBEDDING
        # ====================================================

        self.patch_embed_input = PatchEmbedding(
            input_channels,
            embed_dim,
            patch_size
        )

        # ====================================================
        # ENCODER
        # ====================================================

        self.encoder_cell = SwinLSTMCell(
            embed_dim,
            hidden_dim,
            num_heads,
            window_size,
            dropout
        )

        # ====================================================
        # DECODER SINGLE-SHOT
        # ====================================================

        self.decoder = SingleShotSwinDecoder(
            hidden_dim=hidden_dim,
            output_channels=output_channels,
            future_steps=future_steps,
            patch_size=patch_size,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout
        )

    def forward(
        self,
        x
    ):

        """
        x:
            (B, T, C_in, H, W)

        retorna:
            (B, future_steps, C_out, H, W)
        """

        B, T, C, H, W = x.shape

        device = x.device

        # ====================================================
        # Inicialização
        # ====================================================

        x0 = self.patch_embed_input(
            x[:, 0]
        )

        Hp = x0.shape[1]
        Wp = x0.shape[2]

        h = torch.zeros(
            B,
            Hp,
            Wp,
            self.hidden_dim,
            device=device
        )

        c = torch.zeros_like(h)

        # ====================================================
        # ENCODER TEMPORAL
        # ====================================================

        for t in range(T):

            xt = self.patch_embed_input(
                x[:, t]
            )

            h, c = self.encoder_cell(
                xt,
                h,
                c
            )

        # ====================================================
        # ESTADO FINAL DO ENCODER
        # ====================================================

        skip = h

        # ====================================================
        # DECODER SINGLE-SHOT
        # ====================================================

        predictions = self.decoder(
            h,
            skip
        )

        return predictions
    def forward(
        self,
        x
    ):

        """
        x:
            (B, T, C_in, H, W)

        retorna:
            (B, future_steps, C_out, H, W)
        """

        B, T, C, H, W = x.shape

        device = x.device

        # ====================================================
        # Inicialização
        # ====================================================

        x0 = self.patch_embed_input(
            x[:, 0]
        )

        Hp = x0.shape[1]
        Wp = x0.shape[2]

        h = torch.zeros(
            B,
            Hp,
            Wp,
            self.hidden_dim,
            device=device
        )

        c = torch.zeros_like(h)

        # ====================================================
        # ENCODER TEMPORAL
        # ====================================================

        for t in range(T):

            xt = self.patch_embed_input(
                x[:, t]
            )

            h, c = self.encoder_cell(
                xt,
                h,
                c
            )

        # ====================================================
        # ESTADO FINAL DO ENCODER
        # ====================================================

        skip = h

        # ====================================================
        # DECODER SINGLE-SHOT
        # ====================================================

        predictions = self.decoder(
            h,
            skip
        )

        return predictions
