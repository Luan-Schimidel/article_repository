import torch
import torch.nn as nn


# ============================================================
# ConvLSTM Cell
# ============================================================

class ConvLSTMCell(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        bias=True,
        dropout=0.0
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.kernel_size = kernel_size

        padding = (
            kernel_size[0] // 2,
            kernel_size[1] // 2
        )

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

        # Uma única convolução produz os 4 gates:
        #
        # input gate
        # forget gate
        # output gate
        # candidate cell
        #
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )

    def forward(self, x, h_prev, c_prev):

        # Dropout na entrada
        if self.dropout is not None:
            x = self.dropout(x)

        # Concatena entrada e estado oculto
        combined = torch.cat([x, h_prev], dim=1)

        # Convolução
        gates = self.conv(combined)

        # Divide nos quatro gates
        i, f, o, g = torch.chunk(  gates, chunks=4, dim=1 )

        # Ativações
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)

        g = torch.tanh(g)

        # Atualiza estado da célula
        c_next = f * c_prev + i * g

        # Atualiza estado oculto
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


# ============================================================
# ConvLSTM
# ============================================================

class ConvLSTM(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        return_sequences=True,
        dropout=0.0
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.return_sequences = return_sequences

        self.cell = ConvLSTMCell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            dropout=dropout
        )

    def forward(self, x, initial_state=None):

        # x:
        # (B, T, C, H, W)

        B, T, C, H, W = x.shape

        # ----------------------------------------------------
        # Estado inicial
        # ----------------------------------------------------

        if initial_state is None:

            h = torch.zeros( B, self.hidden_dim,  H,   W,  device=x.device,  dtype=x.dtype )

            c = torch.zeros( B, self.hidden_dim,  H,   W,   device=x.device, dtype=x.dtype )

        else:

            h, c = initial_state

        outputs = []

        # ----------------------------------------------------
        # Percorre o tempo
        # ----------------------------------------------------

        for t in range(T):

            x_t = x[:, t]

            h, c = self.cell( x_t, h, c )

            if self.return_sequences:
                outputs.append(h)

        # ----------------------------------------------------
        # Retorna sequência
        # ----------------------------------------------------

        if self.return_sequences:

            # (B,T,C,H,W)
            output = torch.stack( outputs,   dim=1   )

            return output, h, c

        # ----------------------------------------------------
        # Retorna somente último estado
        # ----------------------------------------------------

        return h, h, c


# ============================================================
# Modelo completo
# ============================================================

class ConvLSTMEncoderDecoder(nn.Module):

    def __init__(
        self,
        input_channels=1,
        t_out=5,
        filters_encoder_1=32,
        filters_encoder_2=64,
        filters_bottleneck=32,
        filters_decoder=32,
        dropout=0.2,
        output_channels=1
    ):
        super().__init__()

        self.t_out = t_out

        # ====================================================
        # ENCODER
        # ====================================================

        self.encoder_lstm1 = ConvLSTM(
            input_dim=input_channels,
            hidden_dim=filters_encoder_1,
            kernel_size=(5, 5),
            return_sequences=True,
            dropout=dropout
        )

        self.encoder_bn1 = nn.BatchNorm2d(
            filters_encoder_1
        )

        self.encoder_relu1 = nn.ReLU()

        # ----------------------------------------------------

        self.encoder_lstm2 = ConvLSTM(
            input_dim=filters_encoder_1,
            hidden_dim=filters_encoder_2,
            kernel_size=(5, 5),
            return_sequences=True,
            dropout=dropout
        )

        self.encoder_bn2 = nn.BatchNorm2d(
            filters_encoder_2
        )

        self.encoder_relu2 = nn.ReLU()

        # ----------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------

        self.encoder_bottleneck = ConvLSTM(
            input_dim=filters_encoder_2,
            hidden_dim=filters_bottleneck,
            kernel_size=(3, 3),
            return_sequences=False,
            dropout=dropout
        )

        # ====================================================
        # DECODER
        # ====================================================

        self.decoder_lstm1 = ConvLSTM(
            input_dim=filters_bottleneck,
            hidden_dim=filters_decoder,
            kernel_size=(3, 3),
            return_sequences=True,
            dropout=dropout
        )

        self.decoder_bn1 = nn.BatchNorm2d(
            filters_decoder
        )

        self.decoder_relu1 = nn.ReLU()

        # ----------------------------------------------------

        self.decoder_lstm2 = ConvLSTM(
            input_dim=filters_decoder,
            hidden_dim=filters_decoder,
            kernel_size=(3, 3),
            return_sequences=True,
            dropout=dropout
        )

        self.decoder_bn2 = nn.BatchNorm2d(
            filters_decoder
        )

        self.decoder_relu2 = nn.ReLU()

        # ====================================================
        # OUTPUT
        # ====================================================

        # Equivalente a:
        #
        # Conv3D(
        #     filters=1,
        #     kernel_size=(1,3,3),
        #     padding='same'
        # )
        #
        self.output_conv = nn.Conv3d(
            in_channels=filters_decoder,
            out_channels=output_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1)
        )

    # ========================================================
    # BatchNorm aplicado temporalmente
    # ========================================================

    def apply_batchnorm(self, x, bn):

        # x:
        # (B,T,C,H,W)

        B, T, C, H, W = x.shape

        # BatchNorm2d espera:
        # (B,C,H,W)

        x = x.reshape(B * T, C, H, W  )

        x = bn(x)

        x = x.reshape( B, T,  C, H,  W )

        return x

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, x):

        # ====================================================
        # INPUT
        # ====================================================
        #
        # PyTorch:
        # (B,T,C,H,W)
        #

        # ====================================================
        # ENCODER - LAYER 1
        # ====================================================

        x, _, _ = self.encoder_lstm1(x)

        x = self.apply_batchnorm(x, self.encoder_bn1  )

        x = self.encoder_relu1(x)

        # ====================================================
        # ENCODER - LAYER 2
        # ====================================================

        x, _, _ = self.encoder_lstm2(x)

        x = self.apply_batchnorm( x, self.encoder_bn2 )

        x = self.encoder_relu2(x)

        # ====================================================
        # BOTTLENECK
        # ====================================================

        _, state_h, state_c = self.encoder_bottleneck(x)

        # state_h:
        #
        # (B,32,H,W)
        #
        # state_c:
        #
        # (B,32,H,W)

        # ====================================================
        # DECODER INPUT
        # ====================================================

        B, C, H, W = state_h.shape

        decoder_input = torch.zeros(
            B,
            self.t_out,
            C,
            H,
            W,
            device=x.device,
            dtype=x.dtype
        )

        # ====================================================
        # DECODER - LAYER 1
        # ====================================================

        x, _, _ = self.decoder_lstm1(
            decoder_input,
            initial_state=(state_h, state_c)
        )

        x = self.apply_batchnorm(
            x,
            self.decoder_bn1
        )

        x = self.decoder_relu1(x)

        # ====================================================
        # DECODER - LAYER 2
        # ====================================================

        x, _, _ = self.decoder_lstm2(x)

        x = self.apply_batchnorm(x, self.decoder_bn2 )

        x = self.decoder_relu2(x)

        # ====================================================
        # OUTPUT
        # ====================================================

        # Conv3D espera:
        #
        # (B,C,T,H,W)
        #
        x = x.permute( 0, 2, 1, 3, 4 )

        x = self.output_conv(x)

        # Retorna para:
        #
        # (B,T,C,H,W)

        x = x.permute(0, 2, 1, 3, 4 )

        return x


