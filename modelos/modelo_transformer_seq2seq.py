import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)
np.random.seed(42)

def patchify(x: torch.Tensor, P: int = PATCH_SIZE) -> torch.Tensor:
    """
    x: (B, T , C, H, W)
    Returns: (B,T, N, C*P*P)  where N = (H//P) * (W//P)
    """
    B, T ,C, H, W = x.shape
    nh, nw = H // P, W // P
    x = x.reshape(B,T ,C, nh, P, nw, P)
    x = x.permute(0,1, 3, 5, 2, 4, 6).contiguous()  # (B, nh, nw, C, P, P)
    x = x.reshape(B, T ,nh * nw, C * P * P)
    return x


def unpatchify(
    patches: torch.Tensor,
    H: int,
    W: int,
    C_out: int = 1,
    P: int = PATCH_SIZE
    ) -> torch.Tensor:

    """
    patches: (B, T, N, C_out*P*P)

    Returns:
        x: (B, T, C_out, H, W)
    """

    B, T, N, D = patches.shape

    nh, nw = H // P, W // P



    assert N == nh * nw, \
        f"N={N}, mas esperado {nh*nw} patches"

    assert D == C_out * P * P, \
        f"Última dimensão={D}, mas esperado {C_out*P*P}"

    x = patches.reshape(B, T, nh, nw, C_out, P, P )

    x = x.permute( 0, 1, 4, 2, 5, 3, 6 ).contiguous()

    x = x.reshape( B, T, C_out, H, W )

    return x


class PatchTransformerAR(nn.Module):

    def __init__(
        self,
        n_channels_input=1,
        n_channels_output=1,
        input_timestep=5,
        output_timestep=5,
        spatial_shape=(130, 130),
        patch_size=10,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.0,
    ):

        super().__init__()

        self.T_in = input_timestep
        self.T_out = output_timestep

        self.H, self.W = spatial_shape
        self.P = patch_size

        self.n_channels_input = n_channels_input
        self.n_channels_output = n_channels_output
        self.d_model = d_model

        # Número de patches espaciais
        self.N = (self.H // self.P) * (self.W // self.P)


        # =========================================================
        # ENCODER PATCH EMBEDDING
        # =========================================================

        self.encoder_patch_embedding = nn.Linear(
            n_channels_input * patch_size * patch_size,
            d_model
        )


        # =========================================================
        # DECODER PATCH EMBEDDING
        # =========================================================

        self.decoder_patch_embedding = nn.Linear(
            n_channels_output * patch_size * patch_size,
            d_model
        )


        # =========================================================
        # POSITIONAL EMBEDDING
        # =========================================================

        self.encoder_pos_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.T_in,
                self.N,
                d_model
            )
        )

        self.decoder_pos_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.T_out,
                self.N,
                d_model
            )
        )


        # =========================================================
        # TRANSFORMER ENCODER
        # =========================================================

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )


        # =========================================================
        # TRANSFORMER DECODER
        # =========================================================

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )


        # =========================================================
        # OUTPUT HEAD
        # =========================================================

        self.output_head = nn.Linear(
            d_model,
            n_channels_output * patch_size * patch_size
        )


        # =========================================================
        # Inicialização
        # =========================================================

        nn.init.normal_(
            self.encoder_pos_embedding,
            mean=0.0,
            std=0.02
        )

        nn.init.normal_(
            self.decoder_pos_embedding,
            mean=0.0,
            std=0.02
        )


    # =============================================================
    # MÁSCARA CAUSAL
    # =============================================================

    def causal_mask(self, T, device):

        mask = torch.triu(
            torch.ones(
                T,
                T,
                device=device
            ),
            diagonal=1
        )

        mask = mask.masked_fill(
            mask == 1,
            float("-inf")
        )

        mask = mask.masked_fill(
            mask == 0,
            0.0
        )

        # Expandir para os patches espaciais

        mask = mask.repeat_interleave(
            self.N,
            dim=0
        )

        mask = mask.repeat_interleave(
            self.N,
            dim=1
        )

        return mask


    # =============================================================
    # ENCODER
    # =============================================================

    def encode(self, x):

        """
        x:
            (B, T_in, C, H, W)

        retorna:
            memory: (B, T_in*N, D)
        """

        B = x.shape[0]

        # ---------------------------------------------------------
        # 1. Patchify
        # ---------------------------------------------------------

        tokens = patchify(
            x,
            self.P
        )

        # (B,T_in,N,C*P²)


        # ---------------------------------------------------------
        # 2. Embedding dos patches
        # ---------------------------------------------------------

        tokens = self.encoder_patch_embedding(
            tokens
        )

        # (B,T_in,N,D)


        # ---------------------------------------------------------
        # 3. Soma do embedding posicional
        # ---------------------------------------------------------

        tokens = (
            tokens
            + self.encoder_pos_embedding
        )


        # ---------------------------------------------------------
        # 4. Achata tempo e espaço em uma única sequência
        # ---------------------------------------------------------

        tokens = tokens.reshape(
            B,
            self.T_in * self.N,
            self.d_model
        )


        # ---------------------------------------------------------
        # 5. Transformer Encoder
        # ---------------------------------------------------------

        memory = self.encoder(
            tokens
        )

        return memory


    # =============================================================
    # DECODER
    # =============================================================

    def decode(self, decoder_input, memory):

        """
        decoder_input:
            (B, T_dec, C, H, W)

        memory:
            (B, T_in*N, D)

        retorna:
            output: (B, T_dec, C, H, W)
        """

        B, T_dec, _, _, _ = decoder_input.shape


        # ---------------------------------------------------------
        # 1. Patchify
        # ---------------------------------------------------------

        tokens = patchify(
            decoder_input,
            self.P
        )

        # (B,T_dec,N,C*P²)


        # ---------------------------------------------------------
        # 2. Embedding dos patches
        # ---------------------------------------------------------

        tokens = self.decoder_patch_embedding(
            tokens
        )


        # ---------------------------------------------------------
        # 3. Soma do embedding posicional
        #    (apenas os T_dec primeiros passos)
        # ---------------------------------------------------------

        tokens = (
            tokens
            + self.decoder_pos_embedding[:, :T_dec]
        )


        # ---------------------------------------------------------
        # 4. Achata tempo e espaço em uma única sequência
        # ---------------------------------------------------------

        tokens = tokens.reshape(
            B,
            T_dec * self.N,
            self.d_model
        )


        # ---------------------------------------------------------
        # 5. Máscara causal (espaço-temporal)
        # ---------------------------------------------------------

        mask = self.causal_mask(
            T_dec,
            decoder_input.device
        )


        # ---------------------------------------------------------
        # 6. Transformer Decoder (self-attn causal + cross-attn memory)
        # ---------------------------------------------------------

        output_tokens = self.decoder(
            tgt=tokens,
            memory=memory,
            tgt_mask=mask
        )


        # ---------------------------------------------------------
        # 7. Projeção de volta para o espaço de patches
        # ---------------------------------------------------------

        out_patches = self.output_head(
            output_tokens
        )

        out_patches = out_patches.reshape(
            B,
            T_dec,
            self.N,
            -1
        )


        # ---------------------------------------------------------
        # 8. Unpatchify -> imagem
        # ---------------------------------------------------------

        output = unpatchify(
            out_patches,
            self.H,
            self.W,
            self.n_channels_output,
            self.P
        )

        return output


    # =============================================================
    # FORWARD (AUTORREGRESSIVO / SCHEDULED SAMPLING)
    # =============================================================

    def forward(self, x, y=None, teacher_forcing_ratio=0.0):

        """
        x:
            (B, T_in, C, H, W)

        y (opcional, apenas treino):
            (B, T_out, C, H, W) — sequência alvo real

        teacher_forcing_ratio:
            probabilidade, em cada passo, de usar o frame REAL (y)
            em vez da própria previsão do modelo como próximo input
            do decoder.

            - 0.0  -> autorregressivo puro (usa sempre a própria previsão)
            - 1.0  -> teacher forcing puro (usa sempre y, se disponível)
            - se y is None, ratio é ignorado e o modelo é sempre
              autorregressivo (comportamento de inferência)

        retorna:
            (B, T_out, C, H, W)
        """

        # ---------------------------------------------------------
        # 1. Encoder
        # ---------------------------------------------------------

        memory = self.encode(x)


        # ---------------------------------------------------------
        # 2. Primeiro input do decoder
        # ---------------------------------------------------------

        decoder_input = x[:, -1:]

        # Começa com o último campo conhecido:
        #
        # [t4]


        predictions = []


        # ---------------------------------------------------------
        # 3. Geração passo a passo
        #    (autorregressiva / scheduled sampling)
        # ---------------------------------------------------------

        for t in range(self.T_out):

            output = self.decode(
                decoder_input,
                memory
            )

            # Pega somente a última previsão

            next_prediction = output[:, -1:]

            predictions.append(
                next_prediction
            )


            # -----------------------------------------------------
            # Decide o que entra no próximo passo:
            #   - frame REAL (teacher forcing), se y disponível
            #     e o sorteio favorecer
            #   - própria previsão do modelo, caso contrário
            # -----------------------------------------------------

            use_teacher_forcing = (
                y is not None
                and torch.rand(1).item() < teacher_forcing_ratio
            )

            if use_teacher_forcing:

                next_input = y[:, t:t + 1]

            else:

                next_input = next_prediction


            decoder_input = torch.cat(
                [
                    decoder_input,
                    next_input
                ],
                dim=1
            )


        # ---------------------------------------------------------
        # 4. Concatenar previsões
        # ---------------------------------------------------------

        predictions = torch.cat(
            predictions,
            dim=1
        )

        return predictions
