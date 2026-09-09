import numpy as np
#import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from modelos.pre_processing import *
from modelos.modelo_convlstm_torch import ConvLSTMEncoderDecoder
from modelos.modelo_SwinTranformer_one_shot import SwinLSTM
from modelos.modelo_SwinTranformer_encoder_autorregressivo import Autorregressive_SwinLSTM

torch.manual_seed(42)
np.random.seed(42)

H, W = 128, 128
PATCH_SIZE = 10
T_in  = 1
T_out = 2
BATCH_SIZE = 2
input_C  = 4
output_C = 1

def load_dataset(dataset):

    dataset = np.load(dataset)
    dataset = dataset[:, 0:128, 0:128]
    dataset_max = np.nanmax(dataset)
    dataset_min = np.nanmin(dataset)
    dataset_mean = np.nanmean(dataset)
    dataset_std  = np.nanstd(dataset)
    return dataset, dataset_mean, dataset_std


hs, hs_mean, hs_std    = load_dataset('dados_teste/hs.npy')
#u10, u10_mean, u10_std = load_dataset('dados_teste/u10.npy')
#v10, v10_mean, v10_std = load_dataset('dados_teste/v10.npy')

print(f"Dataset shape : {hs.shape}")

print(f"Value range   : {np.nanmax(hs):.0f} – {np.nanmin(hs):.0f} meters")
print(f"Global mean   : {np.nanmean(hs):.0f} meters")
print(f"Global std    : {np.nanstd(hs):.0f} meters")


# Normalise

hs_norm = (hs - hs_mean) / hs_std
#u10_norm = (u10 - u10_mean) / u10_std
#v10_norm = (v10 - u10_mean) / v10_std

hs_norm  = np.expand_dims(hs_norm, axis=1)
#u10_norm = np.expand_dims(u10_norm, axis=1)
#v10_norm = np.expand_dims(v10_norm, axis=1)

#dataset =  np.concat([v10_norm,u10_norm,hs_norm,],axis = 1)

dataset = np.concat([hs_norm,hs_norm,hs_norm],axis=1)

n_train = int(0.70*hs_norm.shape[0])
n_val = int(n_train*0.10)


# primeiro 70% dos dados
train_ds  = Wave_Dataset_NanMask(dataset[:n_train],input_timestep = T_in, output_timestep = T_out)
# validando com 10% dos dados
val_ds    = Wave_Dataset_NanMask(dataset[n_train:n_train+n_val],input_timestep = T_in, output_timestep = T_out)

# Testando com 20% dos dados
test_ds   = Wave_Dataset_NanMask(dataset[n_train+n_val:],input_timestep = T_in, output_timestep = T_out)
print('train size :', n_train )
print('val size :', n_val )


train_loader = DataLoader(train_ds,batch_size = BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_ds,batch_size = BATCH_SIZE, shuffle = True)
test_loader = DataLoader(test_ds,batch_size = BATCH_SIZE)


encoder_batch, y = next(iter(train_loader))

print(f"Batch shapes       : encoder {tuple(encoder_batch.shape)}, target {tuple(y.shape)}")




def train_one_epoch(model, loader, optimiser,criterion ,mask = None ,ratio = 0.0):
    model.train()
    total_loss = 0.0
    for x_encoder, y in loader:
        optimiser.zero_grad()
        if mask is not None:
            B,T,C,H,W = x_encoder.shape
            mask_x = mask.to(x_encoder.device)
            mask_x = mask_x.unsqueeze(0).unsqueeze(0)
            mask_x = mask_x.expand(B, T, 1, H, W)
            x_encoder = torch.cat([x_encoder, mask_x], dim=2)
        pred = model(x_encoder)#,teacher_forcing_ratio = ratio)
        loss = criterion(pred, y)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * x_encoder.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion ,mask = None, ratio = 0.0):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_encoder, y in loader:
            if mask is not None:
                B,T,C,H,W = x_encoder.shape
                mask_x = mask.to(x_encoder.device)
                mask_x = mask_x.unsqueeze(0).unsqueeze(0)
                mask_x = mask_x.expand(B, T, 1, H, W)
                x_encoder = torch.cat([x_encoder, mask_x], dim=2)
            pred = model(x_encoder)
            loss = criterion(pred, y)
            total_loss += loss.item() * x_encoder.size(0)
    return total_loss / len(loader.dataset)




def teste(model, loader, mask=None, device="cuda"):

    model.eval()

    resultados = []
    targets = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)
            B,T,C,H,W = x.shape
            # Adiciona máscara como canal
            if mask is not None:

                mask_x = mask.to(device)

                # mask: [H, W]
                # -> [1, 1, 1, H, W]
                mask_x = mask_x.unsqueeze(0).unsqueeze(0).unsqueeze(0)

                # Expande para [B, T, 1, H, W]
                mask_x = mask_x.expand(B, T, 1, H, W )

                x = torch.cat([x, mask_x], dim=2)

            result = model(x)

            resultados.append(result.cpu())
            targets.append(y.cpu())

    resultados = torch.cat(resultados, dim=0)
    targets = torch.cat(targets, dim=0)

    return resultados.numpy(), targets.numpy()      



NUM_EPOCHS = 2
LR = 1e-3



#model = ConvLSTMEncoderDecoder(input_channels = 4, t_out = T_out,  filters_encoder_1 = 32 ,
#                               filters_encoder_2=64, filters_bottleneck=32 ,
#                               filters_decoder=32, dropout=0.2 ,output_channels=1)




#model = SwinLSTM( input_channels=4, output_channels=1,  patch_size=4, embed_dim=32,
#        hidden_dim=32, num_heads=4,  window_size=4, future_steps=T_out,  dropout=0.0  )



model = Autorregressive_SwinLSTM( input_channels= input_C, output_channels= output_C, future_steps = T_out, patch_size=4,
        embed_dim=64,   hidden_dim=64, num_heads=4, window_size=4,  dropout=0.0,
		teacher_forcing_ratio=0.0)


n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {n_params:,}")


optimiser = torch.optim.Adam(model.parameters(), lr=LR)
criterion = MaskedMSELoss(train_ds.mask)



train_losses = []
val_losses   = []


###### TREINANDO O MODELO 

for epoch in range(1, NUM_EPOCHS + 1):

    #ratio = max(0.0, 1.0 - epoch / NUM_EPOCHS)
    ratio = 0.0
    train_loss = train_one_epoch(model, train_loader, optimiser, criterion, mask = train_ds.mask , ratio = ratio)
    train_losses.append(train_loss)

    val_loss = evaluate(model, val_loader, criterion, mask = train_ds.mask)
    val_losses.append(val_loss)


    print(f"Epoch {epoch:2d}/{NUM_EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}")



pred, real = teste(model, test_loader, train_ds.mask,device = 'cpu')

#result = np.array(result).reshape(-1,T,output_C, H,W)

###### avaliando conjunto de teste



