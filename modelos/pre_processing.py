import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn

class Wave_Dataset_encoder_decoder(Dataset):

    def __init__(
        self,
        dataset: np.ndarray,
        input_timestep: int = 5,
        output_timestep: int = 5,
    ):

        # dataset:
        # (T, H, W)

        self.data = torch.from_numpy(dataset).float().unsqueeze(1)

        # (T, 1, H, W)

        self.input_timestep = input_timestep
        self.output_timestep = output_timestep


    def __len__(self):

        return ( len(self.data) - self.input_timestep  - self.output_timestep  + 1  )


    def __getitem__(self, idx):

        # --------------------------------------------------
        # Encoder
        # --------------------------------------------------

        encoder_start = idx

        encoder_end = ( idx + self.input_timestep )

        x_encoder = self.data[ encoder_start:encoder_end ]

        # (5, 1, H, W)


        # --------------------------------------------------
        # Target / Output
        # --------------------------------------------------

        output_start = encoder_end

        output_end = ( output_start  + self.output_timestep )

        y = self.data[ output_start:output_end ]

        # (5, 1, H, W)


        # --------------------------------------------------
        # Decoder input
        # --------------------------------------------------

        # Primeiro timestep do decoder:
        # START

        start_token = torch.zeros_like( y[:1])

        # Depois entram os valores reais anteriores
        # durante teacher forcing

        decoder_input = torch.cat( [ start_token,  y[:-1] ], dim=0 )

        # (5, 1, H, W)


        return x_encoder, decoder_input, y




class Wave_Dataset_NanMask(Dataset):

    def __init__(
        self,
        dataset,
        input_timestep=10,
        output_timestep=10
    ):

        # -----------------------------------------
        # Máscara terra/oceano
        # -----------------------------------------

        self.mask = torch.from_numpy((~np.isnan(dataset[0]))[0,:,:].astype(np.float32))


        # -----------------------------------------
        # Substituir NaN
        # -----------------------------------------

        dataset = np.nan_to_num( dataset,   nan=0.0 )


        # -----------------------------------------
        # Tensor
        # -----------------------------------------

        self.data = torch.from_numpy( dataset )#.float().unsqueeze(1)

        # (T,1,H,W)

        self.input_timestep = input_timestep
        self.output_timestep = output_timestep


    def __len__(self):

        return (
            len(self.data)
            - self.input_timestep
            - self.output_timestep
            + 1
        )


    def __getitem__(self, idx):

        idx_input = (
            idx + self.input_timestep
        )

        idx_output = (
            idx_input + self.output_timestep
        )

        x_encoder = self.data[
            idx:idx_input
        ]

        y = self.data[
            idx_input:idx_output,2:,:,: # indices 0 e 1 são os valores de vento. As variaveis de ondas (hs,mwp,etc) são a partir do indice 2
        ]   

        return x_encoder, y



class MaskedMSELoss(nn.Module):

    def __init__(self, mask):
        super().__init__()

        mask = np.where(mask==1,mask,0)
        mask = torch.from_numpy(mask).float()
        self.register_buffer(
            "mask",
            mask.float()
        )

    def forward(self, pred, target):

        # pred:
        # (B, T, C, H, W)

        # target:
        # (B, T, C, H, W)

        mask = self.mask

        # (H, W)
        # ↓
        # (1, 1, 1, H, W)

        mask = mask[None, None, None, :, :]


        squared_error = (pred - target) ** 2

        masked_error = squared_error * mask

        loss = (
            masked_error.sum()
            / mask.sum().clamp_min(1.0)
        )

        return loss
