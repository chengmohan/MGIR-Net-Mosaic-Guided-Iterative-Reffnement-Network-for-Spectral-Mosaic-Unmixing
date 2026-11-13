import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from model.model import AutoEncoder
import torch
import torch.utils
import torch.utils.data
from torch import nn
from time import time
from tqdm import tqdm
from model.Rank_net import SSAF
from model.UNmamba import UNMambaLinear
from utils.loss import My_Loss
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from utils.FCLSU import FCLSU
from utils.common_utils import *
from utils.IPPI_re import IPPI_reconstruction, tv_loss
from utils.loadhsi import loadhsi
from utils.hyperVca import hyperVca
import torch.nn.functional as F
from utils.result_em import result_em
import scipy.io as scio
from utils.IPPI_re import IPPI_reconstruction, tv_loss
import scipy
from scipy.signal import convolve2d
from utility import SparseKLloss, NonZeroClipper, SumToOneLoss, load_HSI, hyperVca, MyTrainData, robust_vca_fusion
import pandas as pd
import sys
from models import *
sys.path.append('/home/Chengmohan/文档/jiehun/HySUPP-main')
sys.path.append('/home/Chengmohan/文档/jiehun/UNMamba-main/utils')
sys.path.append('/home/Chengmohan/文档/jiehun/Hyperspectral-Unmixing-Models-main/CyCUNet')

from utils.CMH import Endmember_extract
from utility import SparseKLloss, NonZeroClipper, SumToOneLoss
import plots
import mymertics
import math
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
# import time
dtype = torch.cuda.FloatTensor
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('training on', device)

cases = ['ex2', 'ridge', 'houston','OURS']
case = cases[1]
# rCol = 100
# nCol = 100
# epochs = 2000
# z_dim = 4
epochs = 1000
pattern25 = np.arange(25).reshape(5,5)
INPUT = 'noise'  # 'meshgrid'
pad = 'reflection'
need_bias = True
OPT_OVER = 'net'  # 'net,input'

#
reg_noise_std = 0.0  # 1/100# 1./30. # set to 1./20. for sigma=50
LR1 = 0.001

OPTIMIZER1 = 'adam'  # 'RMSprop'#'adam' # 'LBFGS'

num_iter1 = 3000

def build_kernel(device, dtype):
    k1d = torch.tensor(
        [1, 2, 3, 4, 5, 4, 3, 2, 1],
        dtype=torch.float32,
        device=device
    )

    k2d = torch.outer(k1d, k1d)
    k2d = k2d / 25   # ? ÕýÈ·¹éÒ»»¯

    return k2d.unsqueeze(0).unsqueeze(0)  # (1,1,9,9)

def residual_re(x, y, mask, kernel, alpha=0.1):
    """
    x:    (1, 25, H, W)  current reconstruction
    y:    (1, 25, H, W)  observed (or PPID reference)
    mask: (1, 25, H, W)  sparse sampling mask (0/1)
    """

    # -------------------------
    # 1. residual
    # -------------------------
    r = y - x
    r = r * mask   # ? sparse residual

    B = x.shape[1]
    pad = kernel.shape[-1] // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode='circular')
    y_pad = F.pad(y, (pad, pad, pad, pad), mode='circular')
    m_pad = F.pad(mask, (pad, pad, pad, pad), mode='circular')
    # -------------------------
    # 2. normalized diffusion (VERY IMPORTANT)
    # -------------------------
    r = (y_pad - x_pad) * m_pad

    # =====================================================
    # 2. normalized masked diffusion
    # =====================================================
    num = F.conv2d(
        r,
        kernel.repeat(B, 1, 1, 1),
        groups=B
    )


    # -------------------------
    # 3. stable update (small step!)
    # -------------------------
    x_new = x + alpha * num

    return x_new

def msfa_downsample(x):

    """
    x:
        shape = (1,25,H,W)

    return:
        shape = (1,25,H/5,W/5)
    """

    B, C, H, W = x.shape

    assert C == 25
    assert H % 5 == 0
    assert W % 5 == 0

    out = []

    for band in range(25):

        r = band // 5
        c = band % 5

        sampled = x[:, band, r::5, c::5]

        out.append(sampled)

    out = torch.stack(out, dim=1)

    return out

def compute_sad_matrix(E_true, E_est):
    """
    ¼ÆËãËùÓÐ¶ËÔªÖ®¼äµÄSAD¾ØÕó (p x p)
    """
    p = E_true.shape[1]
    sad_matrix = np.zeros((p, p))

    for i in range(p):
        for j in range(p):
            x = E_true[:, i]
            y = E_est[:, j]

            cos_theta = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)
            cos_theta = np.clip(cos_theta, -1, 1)  # ·ÀÖ¹ÊýÖµÒç³ö

            sad_matrix[i, j] = np.arccos(cos_theta)

    return sad_matrix


class SADLoss(nn.Module):

    def __init__(self, eps=1e-8, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, y_true, y_pred):
        """
        y_true: (25, 3, 1, 1)
        y_pred: (25, 3, 1, 1)
        """

        # =====================================================
        # 1. reshape -> (bands, K)
        # =====================================================

        y_true = y_true.view(y_true.shape[0], y_true.shape[1])  # (25,3)
        y_pred = y_pred.view(y_pred.shape[0], y_pred.shape[1])  # (25,3)

        # =====================================================
        # 2. transpose -> (K, bands)
        # Ã¿¸ö¶ËÔªÒ»ÐÐ
        # =====================================================

        y_true = y_true.t()  # (3,25)
        y_pred = y_pred.t()  # (3,25)

        # =====================================================
        # 3. L2 normalize per endmember
        # =====================================================

        y_true = y_true / (torch.norm(y_true, dim=-1, keepdim=True) + self.eps)
        y_pred = y_pred / (torch.norm(y_pred, dim=-1, keepdim=True) + self.eps)

        # =====================================================
        # 4. cosine similarity
        # =====================================================

        cos = torch.sum(y_true * y_pred, dim=-1)  # (3,)

        # =====================================================
        # 5. stability clamp
        # =====================================================

        cos = torch.clamp(cos, -1.0 + self.eps, 1.0 - self.eps)

        # =====================================================
        # 6. SAD surrogate loss
        # =====================================================

        loss = 1.0 - cos  # (3,)

        # =====================================================
        # 7. reduction
        # =====================================================

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def scaling_smooth_loss(psi):
    """
    spatial smoothness constraint for scaling coefficients

    Args:
        psi:
            scaling coefficients
            [N,K,H,W]

    Returns:
        smooth loss
    """

    # horizontal gradient
    dx = psi[:, :, :, 1:] - psi[:, :, :, :-1]


    # vertical gradient
    dy = psi[:, :, 1:, :] - psi[:, :, :-1, :]


    loss_x = torch.mean(dx ** 2)
    loss_y = torch.mean(dy ** 2)


    return loss_x + loss_y

def scale_regularization(FACTOR):
    """
    FACTOR: scaling factor tensor
             shape: [B, C, H, W]
             B: batch
             C: endmembers
             H,W: spatial size

    L_scale = ||Psi - 1||^2
    """

    loss_scale = torch.mean((FACTOR - 1.0) ** 2)

    return loss_scale

def mySAD(a, b):
    a = a / (np.linalg.norm(a, axis=0, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=0, keepdims=True) + 1e-8)
    return np.arccos(np.clip(np.sum(a * b, axis=0), -1, 1))

def mean_sad(E_true, E_est):
    """
    ¼ÆËã×îÓÅÆ¥ÅäºóµÄÆ½¾ùSAD
    """
    sad_matrix = compute_sad_matrix(E_true, E_est)

    # ÐÙÑÀÀûËã·¨£¨×îÐ¡Æ¥Åä£©
    row_ind, col_ind = linear_sum_assignment(sad_matrix)

    matched_sad = sad_matrix[row_ind, col_ind]

    return np.mean(matched_sad), matched_sad

def add_noise_snr(msfa, snr_db):
    # 1. ¼ÆËãÐÅºÅ¹¦ÂÊ
    signal_power = np.mean(msfa ** 2)

    # 2. ¼ÆËãÔëÉù¹¦ÂÊ
    noise_power = signal_power / (10 ** (snr_db / 10))

    # 3. ±ê×¼²î
    sigma = np.sqrt(noise_power)

    # 4. ¼ÓÔëÉù
    noise = np.random.normal(0, sigma, msfa.shape).astype(np.float32)
    noisy = msfa + noise

    return np.clip(noisy, 0, 1)

def l21_spectral(X, eps=1e-8):
    # X: (1, p, h, w)
    X = X.squeeze(0)              # (p, h, w)
    X = X.view(X.shape[0], -1)    # (p, h*w)
    return torch.sum(torch.sqrt(torch.sum(X**2, dim=1) + eps))

class CAE_AbEst(nn.Module):
    def __init__(self,input_depth, rmax, H, W):
        super(CAE_AbEst, self).__init__()
        # encoding layers
        self.conv1 = nn.Sequential(
            UnmixArch(
                input_depth, rmax,
                # num_channels_down = [8, 16, 32, 64, 128],
                # num_channels_up   = [8, 16, 32, 64, 128],
                # num_channels_skip = [4, 4, 4, 4, 4],
                num_channels_down=[256],
                num_channels_up=[256],
                num_channels_skip=[4],
                filter_size_up=3, filter_size_down=3, filter_skip_size=1,
                upsample_mode='bilinear',  # downsample_mode='avg',
                need1x1_up=True,
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
        )
        # self.softmax = nn.Softmax(dim=1)
        self.raw_factor = nn.Parameter(torch.zeros(1,rmax, H, W))

    def forward(self, x, y):
        x = self.conv1(x)
        factor = torch.exp(self.raw_factor)
        out = y * factor
        x2 = x * out
        x2 = torch.sum(x2, dim=1)
        x2 = x2.unsqueeze(0)
        # x = self.softmax(x1)
        return x, x2

def weighted_bilinear_interpolation(raw, kernel):
    """
    raw:  (H, W)  µ¥¸öbandµÄÏ¡ÊèÍ¼
    mask: (H, W)  0/1 ÑÚÄ£
    kernel: ¾í»ýºË
    """
    H, W = raw.shape
    pad_H, pad_W = H % 5, W % 5
    pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2
    d_h = pad_h - pad_H
    d_w = pad_w - pad_W
    raw_padded = np.pad(raw, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    raw_padded[0:3, :] = raw_padded[5:8, :]
    raw_padded[:, 0:3] = raw_padded[:, 5:8]
    raw_padded[H + 4:H + 8, :] = raw_padded[H - 1:H + 3, :]
    raw_padded[:, W + 4:W + 8] = raw_padded[:, W - 1:W + 3]
    numerator = convolve2d(raw_padded,kernel, mode='valid')
    return numerator

class StableSADLoss(nn.Module):

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_true, y_pred):

        # y_true: (N, K, B)
        # y_pred: (1, K, B)

        # L2 normalize along spectral dimension
        y_true = F.normalize(y_true, p=2, dim=-1)
        y_pred = F.normalize(y_pred, p=2, dim=-1)

        # broadcasting automatically
        cos = torch.sum(y_true * y_pred, dim=-1)   # (N, K)

        cos = torch.clamp(cos, -1 + 1e-6, 1 - 1e-6)

        loss = 1 - cos

        return loss.mean()

class StableSADLoss2(nn.Module):

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_true, y_pred):
        # y_true: (N, 1, B)
        # y_pred: (1, 1, B)

        # L2 normalize along spectral dimension
        y_true = F.normalize(y_true, p=2, dim=-1)
        y_pred = F.normalize(y_pred, p=2, dim=-1)

        # broadcast automatically
        cos = torch.sum(y_true * y_pred, dim=-1)   # (N,1)

        cos = torch.clamp(cos, -1 + 1e-6, 1 - 1e-6)

        loss = 1 - cos

        return loss.mean()

def reconstruction_SADloss(output, target):
    _, band, h, w = output.shape
    output = torch.reshape(output, (band, h * w))
    target = torch.reshape(target, (band, h * w))
    # abundance_loss = torch.acos(torch.cosine_similarity(output, target, dim=0))
    cos_sim = torch.cosine_similarity(output, target, dim=0, eps=1e-8)
    cos_sim = torch.clamp(cos_sim, -1 + 1e-7, 1 - 1e-7)
    abundance_loss = torch.acos(cos_sim)
    k = abundance_loss.detach().cpu().numpy()
    # k = abundance_loss.detach().cpu().numpy()
    abundance_loss = torch.mean(abundance_loss)

    return abundance_loss

def l05_norm(A, eps=1e-8):
    return torch.sum((A.abs() + eps).pow(0.5))

def append_metrics_to_excel(row_dict, excel_path):
    df_new = pd.DataFrame([row_dict])

    if os.path.exists(excel_path):
        df_old = pd.read_excel(excel_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_excel(excel_path, index=False)

def generate_msfa(origin, pattern):
    """
    origin:  (B, H, W) ¸ß¹âÆ×Êý¾Ý
    pattern: (ph, pw)  Ä£°å£¬Ã¿¸öÖµ±íÊ¾²¨¶Î±àºÅ

    return:
        msfa_img   -> (H, W) ÂíÈü¿ËÍ¼Ïñ
        msfa_cube  -> (B, H, W) Ï¡ÊèÁ¢·½Ìå
        mask_index -> (H, W) ²¨¶ÎË÷ÒýÄ£°å
    """

    B, H, W = origin.shape
    ph, pw = pattern.shape

    # -------------------------
    # 1?? Éú³ÉÍêÕû³ß´ç mask
    # -------------------------
    tile_h = int(np.ceil(H / ph))
    tile_w = int(np.ceil(W / pw))

    mask_index = np.tile(pattern, (tile_h, tile_w))
    mask_index = mask_index[:H, :W]

    # -------------------------
    # 2?? Éú³É MSFA ÂíÈü¿ËÍ¼Ïñ
    # -------------------------
    rows = np.arange(H)[:, None]
    cols = np.arange(W)

    msfa_img = origin[mask_index, rows, cols]

    # -------------------------
    # 3?? Éú³ÉÏ¡Êè MSFA Á¢·½Ìå
    # -------------------------
    msfa_cube = np.zeros_like(origin)

    for b in range(B):
        msfa_cube[b][mask_index == b] = origin[b][mask_index == b]

    return msfa_img, msfa_cube, mask_index

def get_kernel(num_band):
    if num_band == 16:
        v = np.array([1,2,3,4,3,2,1], dtype=np.float32)
        F = np.outer(v, v)
        F = F / 16
    elif num_band == 25:
        v = np.array([1,2,3,4,5,4,3,2,1], dtype=np.float32)
        F = np.outer(v, v)
        F = F / 25
    else:
        raise ValueError("num_band must be 16 or 25")
    return F

def reconstruct_msfa(raw_cube, num_band):
    """
    raw_cube:  (B, H, W)
    mask_cube: (B, H, W)
    """
    B, H, W = raw_cube.shape
    kernel = get_kernel(num_band)

    recon = np.zeros_like(raw_cube, dtype=np.float32)

    for b in range(B):
        recon[b,:, :] = weighted_bilinear_interpolation(
            raw_cube[b, :, :],
            kernel
        )
    return recon


def sparse_entropy_loss(a, eps=1e-8, reduction='mean'):
    """
    Ï¡Êè½»²æìØ£¨¸ºìØ£©ËðÊ§

    ²ÎÊý:
        a: softmax Êä³öµÄ·á¶È (B, N, H, W) »ò (B, N)
        eps: ·ÀÖ¹ log(0)
        reduction: 'mean' | 'sum' | 'none'

    ·µ»Ø:
        loss
    """
    # ·ÀÖ¹ÊýÖµÎÊÌâ
    a_clamped = a.clamp(min=eps)

    # ¼ÆËã a * log(a)
    loss = a_clamped * torch.log(a_clamped)

    # °´Í¨µÀ£¨·á¶ÈÎ¬£©ÇóºÍ
    loss = loss.sum(dim=1)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def generate_msfa_torch(origin, pattern):
    """
    origin : (1, B, H, W)
    pattern: (ph, pw)  band index template

    return:
        msfa_img : (1, 1, H, W)
    """

    device = origin.device
    N, B, H, W = origin.shape
    ph, pw = pattern.shape

    # 1?? Éú³ÉÍêÕû³ß´ç mask_index
    tile_h = math.ceil(H / ph)
    tile_w = math.ceil(W / pw)
    mask_index = np.tile(pattern, (tile_h, tile_w))[:H, :W]
    mask_index = torch.from_numpy(mask_index).to(device)
    # mask_index = pattern.repeat(tile_h, tile_w)[:H, :W].to(device)

    # 2?? one-hot mask  (B, H, W)
    mask_onehot = F.one_hot(mask_index, num_classes=B)
    mask_onehot = mask_onehot.permute(2, 0, 1).float()  # (B,H,W)
    mask_onehot = mask_onehot.unsqueeze(0)              # (1,B,H,W)

    # 3?? Éú³É MSFA Í¼Ïñ
    msfa_img = (origin * mask_onehot).sum(dim=1, keepdim=True)

    return msfa_img

def my_load(path):

    data = scio.loadmat(path)

    Y = data['data']
    Y = np.array(Y, dtype=np.float32)
    rnum, cnum = data['mosaic_16'].shape
    if Y.shape[0]>Y.shape[1]:
        Y = Y.T
    band = Y.shape[0]
    Y = np.reshape(Y, [band, cnum, rnum])
    Y = Y.transpose(0, 2, 1)

    patch = 5
    # ¼ÆËã¿ÉÒÔÕû³ý5µÄ³ß´ç
    r_crop = (rnum // patch) * patch
    c_crop = (cnum // patch) * patch
    full_mask = np.zeros((25, r_crop, c_crop), dtype=np.float32)
    for i in range(5):  # Python Ë÷Òý´Ó 0 ¿ªÊ¼
        for j in range(5):
            full_mask[i * 5 + j, i::5, j::5] = 1
    Y = Y[:, :r_crop, :c_crop]
    max_value = np.max(Y.flatten())
    # max_value = 1
    Y = Y / max_value
    hsi_mean = np.mean(Y, axis=(1,2))
    hsi_mean = np.repeat(hsi_mean, data['E_true'].shape[0]).reshape(-1,data['E_true'].shape[0]).T
    Undip_Y = Y
    pca_a = Y.reshape(25, -1)
    band = Y.shape[0]
    msfa25_img, msfa25_cube, mask25_index = generate_msfa(Y, pattern25)
    # Y = reconstruct_msfa(msfa25_cube, 25)
    msfa25_img = msfa25_img[:r_crop, :c_crop]
    mean_mosaic = np.zeros(25)
    for i in range(5):
        for j in range(5):
            mean_mosaic[i * 5 + j] = msfa25_img[i::5, j::5].mean()
    pca_c = np.zeros((band,r_crop//5,c_crop//5))
    for i in range(5):
        for j in range(5):
            pca_c[i*5+j,:,:] = msfa25_img[i::5,j::5]
    pca_c = pca_c.reshape(band,-1)
    msfa25_img = msfa25_img.reshape(r_crop // 5, 5, c_crop // 5, 5)
    print(np.median(msfa25_img))
    msfa25_img = msfa25_img.transpose(0, 2, 1, 3)
    msfa25_img = msfa25_img.reshape(-1, 5, 5)
    pca_b = Y.reshape(25, -1)
    Y = Y.reshape(band, r_crop // 5, 5, c_crop // 5, 5)
    Y = Y.transpose(1, 3, 0, 2, 4)
    Y = Y.reshape(-1, band, 5, 5)

    A_true = data['A_true']
    A_true = np.array(A_true, dtype=np.float32)
    if A_true.shape[0]>A_true.shape[1]:
        A_true = A_true.T
    P = A_true.shape[0]
    M = data['E_true']
    M= np.array(M, dtype=np.float32)
    if M.shape[0]<M.shape[1]:
        M = M.T
    A_true = np.reshape(A_true, (P, cnum, rnum))
    A_true = A_true.transpose(0, 2, 1)
    A_true = A_true[:, :r_crop, :c_crop]
    A_true = np.reshape(A_true, (P, r_crop*c_crop))

    msfa25_img = msfa25_img
    block_num = r_crop // 5
    return Y,A_true, P, M, r_crop, c_crop, msfa25_img, mean_mosaic,pca_a,pca_b,pca_c, max_value, Undip_Y,hsi_mean, block_num

def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and m.weight is not None:
        if classname.find('Conv') != -1:
            nn.init.kaiming_uniform_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias.data, 0)

def append_row(seed, sad,excel_path):
    new_df = pd.DataFrame([[seed, sad]], columns=["seed", "SAD"])

    if not os.path.exists(excel_path):
        new_df.to_excel(excel_path, index=False)
    else:
        with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            old_df = pd.read_excel(excel_path)
            startrow = len(old_df) + 1

            new_df.to_excel(writer, index=False, header=False, startrow=startrow)


def set_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def spectral_chaos_batch(blocks):
    """
    blocks:
        shape = (N,25,5,5)

    return:
        chaos: (N,)
    """

    N, B, H, W = blocks.shape

    # -> (N,25 pixels,25 bands)
    x = blocks.reshape(N, B, -1).transpose(0, 2, 1)

    # normalize spectra
    norm = np.linalg.norm(x, axis=2, keepdims=True) + 1e-12
    x = x / norm

    # cosine similarity matrix
    # (N,25,25)
    cos_mat = np.matmul(x, x.transpose(0, 2, 1))

    cos_mat = np.clip(cos_mat, -1, 1)

    # spectral angle matrix
    angle_mat = np.arccos(cos_mat)

    # upper triangle indices
    idx = np.triu_indices(25, k=1)

    # mean SAM
    chaos = angle_mat[:, idx[0], idx[1]].mean(axis=1)

    return chaos

def train(lr=0.05, path = '' ,base = '',sad_scale = 0.0, seeds = 3, iters = 20,EI_path = ''):
    dataset = base
    set_seed(seeds)
    # dataset = 'simple'
    # base = 'simple'
    # EI = EI_path
    # CyCU params
    if dataset == 'Samson':
        LR, EPOCH, batch_size = 2e-3, 480, 1
        beta, delta, gamma = 0.5, 1e-3, 1e-7
        sparse_decay, weight_decay_param = 5e-6, 1e-4
        lambda_new = 0.07
    if dataset == 'Jasper':
        LR, EPOCH, batch_size = 8e-3, 420, 1
        beta, delta, gamma = 0.5, 1e-2, 1e-7
        sparse_decay, weight_decay_param = 0, 0
        lambda_new = 0.007
    if dataset == 'Apex':
        LR, EPOCH, batch_size = 8e-3, 420, 1
        beta, delta, gamma = 0.5, 1e-2, 1e-7
        sparse_decay, weight_decay_param = 0, 0
        lambda_new = 0.007
    if dataset == 'Huston':
        LR, EPOCH, batch_size = 2e-3, 480, 1
        beta, delta, gamma = 0.5, 1e-3, 1e-7
        sparse_decay, weight_decay_param = 5e-6, 1e-4
        lambda_new = 0.07
    if dataset == 'Urban':
        LR, EPOCH, batch_size = 0.002, 300, 1
        beta, delta, gamma = 0.5, 0.001, 1e-7
        sparse_decay, weight_decay_param = 0, 1e-4
        lambda_new = 0.1
    if dataset == 'Checkerboard':
        LR, EPOCH, batch_size = 0.002, 300, 1
        beta, delta, gamma = 0.5, 0.001, 1e-7
        sparse_decay, weight_decay_param = 0, 1e-4
    if dataset == 'dc':
        LR, EPOCH, batch_size = 0.002, 300, 1
        beta, delta, gamma = 0.5, 0.001, 1e-7
        sparse_decay, weight_decay_param = 0, 1e-4
    if dataset == 'sim':
        LR, EPOCH, batch_size = 0.001, 300, 1
        beta, delta, gamma = 0.5, 0.001, 1e-7
        sparse_decay, weight_decay_param = 0, 1e-3
    if dataset == 'berlin':
        LR, EPOCH, batch_size = 2e-3, 480, 1
        beta, delta, gamma = 0.5, 1e-3, 1e-7
        sparse_decay, weight_decay_param = 5e-6, 1e-4
        lambda_new = 0.07
    if dataset == 'simple':
        LR, EPOCH, batch_size = 2e-3, 30, 1
        beta, delta, gamma = 0.5, 1e-3, 1e-7
        sparse_decay, weight_decay_param = 5e-6, 1e-4
        lambda_new = 0.07
    if dataset == 'complex':
        LR, EPOCH, batch_size = 2e-3, 30, 1
        beta, delta, gamma = 0.5, 1e-3, 1e-7
        sparse_decay, weight_decay_param = 5e-6, 1e-4
        lambda_new = 0.07
    INPUT = 'noise'
    temp_scale = sad_scale
    Y, A_true, P, M_true,rCol, nCol, msfa25_img, mean_mosaic,pca_a,pca_b,pca_c,max_value,Undip_Y,hsi_mean,block_num = my_load(path)
    # fcls_y = (Y.reshape(block_num,block_num,25,5,5).transpose(0,3,1,4,2).reshape(-1,25))

    hsi_mean = torch.tensor(hsi_mean, dtype=torch.float32).to(device)
    Undip_Y = torch.tensor(Undip_Y, dtype=torch.float32).unsqueeze(0).to(device)
    endmember_num = np.mean(Y, axis=(2, 3))
    endmember_num = np.expand_dims(endmember_num, axis=1)
    # P = 4
    # endmember_num = np.zeros((Y.shape[0],25,1))
    # for vc in range(Y.shape[0]):
    #     temp_vc = Y[vc,:,:,:]
    #     temp_vc = temp_vc.reshape(25,25)
    #     endmember_num[vc,:,:], _, snrEstimate = hyperVca(temp_vc, 1)
    # endmember_num = endmember_num.transpose(0,2,1)
    endmember_num = torch.tensor(endmember_num, dtype=torch.float32).to(device)
    net_input1 = get_noise(P, INPUT,
                           (rCol, nCol)).type(dtype).detach()
    mse = torch.nn.MSELoss().type(dtype)
    msfa25_img = torch.tensor(msfa25_img).to(device).unsqueeze(-1)
    full_msfa = msfa25_img.reshape(rCol//5,nCol//5,5,5,1)
    full_msfa = full_msfa.permute(0,2,1,3,4)
    full_msfa = full_msfa.reshape(rCol,nCol).unsqueeze(0)
    full_msfa = full_msfa.repeat(25, 1, 1).unsqueeze(0).to(device)
    mask25 = np.zeros((5, 5, 25), dtype=np.float32)
    for i in range(5):  # Python Ë÷Òý´Ó 0 ¿ªÊ¼
        for j in range(5):
            mask25[i::5, j::5, i * 5 + j] = 1
    mask25 = torch.tensor(mask25, dtype=torch.float32).unsqueeze(0).to(device)
    full_mask = np.zeros((rCol, nCol, 25), dtype=np.float32)
    for i in range(5):  # Python Ë÷Òý´Ó 0 ¿ªÊ¼
        for j in range(5):
            full_mask[i::5, j::5, i * 5 + j] = 1
    full_mask = torch.tensor(full_mask, dtype=torch.float32).unsqueeze(0).to(device).permute(0,3,1,2)
    SAD = StableSADLoss()
    SAD2 = StableSADLoss2()
    z_dim = P

    # print('SNR:', snrEstimate)
    # msfa25_cube = torch.tensor(msfa25_cube).unsqueeze(0).to(device)
    mean_mosaic = torch.tensor(mean_mosaic.astype('float32')).to(device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    batchs, Channel, r_patch,c_patch = Y.shape
    batch_size = 1
    # Y = Y.reshape(batchs*Channel, r_patch, c_patch)
    #
    # Y = torch.tensor(Y).unsqueeze(0)
    # train_cube = Y
    # train_cube = torch.utils.data.TensorDataset(train_cube)
    # train_cube = torch.utils.data.DataLoader(train_cube, batch_size=batch_size, shuffle=True)
    RANK = 1
    # set_seed(seeds)
    model = SSAF(RANK, Channel, rCol, nCol, z_dim, batchs).to(device)
    # with torch.no_grad():
    #     model.end_weights.data.copy_(endmember_num)
    # CyCu = AutoEncoder(z_dim, Channel).cuda()
    # nn.init.kaiming_normal_(CyCu.encoder[0].weight.data)
    # nn.init.kaiming_normal_(CyCu.encoder[4].weight.data)
    # nn.init.kaiming_normal_(CyCu.encoder[7].weight.data)
    # CyCu.apply(weights_init2)
    criterionSumToOne = SumToOneLoss(gamma).cuda()
    criterionSparse = SparseKLloss(sparse_decay).cuda()

    # abundance = CAE_AbEst(z_dim, z_dim,rCol, nCol).to(device)

    # model = SSAF(P, Channel, rCol, nCol, z_dim, M0,batchs).to(device)
    model.apply(weights_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    tic = time()
    chaos = spectral_chaos_batch(Y)
    threshold = np.percentile(chaos, 50)

    # ¡À¡ê¨¢????¨¨?¡§?¨¦
    mask = chaos <= threshold
    remove_Y = Y[~mask, :, :, :]
    crop_Y = Y[mask, :, :, :]
    scio.savemat('crop_Y.mat', {'crop_Y': crop_Y})
    scio.savemat('remove_Y.mat', {'remove_Y': remove_Y})
    remain = Y[mask, :, :, :].transpose(0, 2, 3, 1)
    all_ems = []
    Endmember = remain.reshape(-1, 25)
    EM_hat, _, _ = hyperVca(Endmember.T, P)
    temp_EM = EM_hat
    # FCLS_a = FCLSU(EM_hat, fcls_y.T, 0.01)
    # FCLS_a = FCLS_a.clone().detach()
    # FCLS_a = torch.reshape(FCLS_a, (P, rCol, nCol)).unsqueeze(0).to(device)
    sub_patch = remain.shape[0] // 2
    if base == 'Huston':
        n1 = 17
        n2 = 13
        # n1 = 21
        # n2 = 21
    if base == 'Apex':
        n1 = 22
        n2 = 11
    if base in ['Jasper', 'simple', 'complex']:
        # n1 = 20
        # n2 = 10
        n1 = 20
        n2 = 10
    if base == 'Urban':
        n1 = 1861
        n2 = 1
    if base == 'Checkerboard':
        n1 = 14
        n2 = 56
        # n1 = 61
        # n2 = 61
    remain = remain.reshape(n1, n2, 5, 5, Channel).transpose(0, 2, 1, 3, 4)
    remain = remain.reshape(n1 * 5, n2 * 5, Channel)
    crop_data = remain.reshape(-1, Channel)
    remain = torch.tensor(remain).to(device)
    x = remain.permute(2, 0, 1).unsqueeze(0)
    E_VCA_init = EM_hat
    if base not in ['k']:
        estimation_my = E_VCA_init
        CyCu = AutoEncoder(z_dim, Channel).cuda()
        nn.init.kaiming_normal_(CyCu.encoder[0].weight.data)
        nn.init.kaiming_normal_(CyCu.encoder[4].weight.data)
        nn.init.kaiming_normal_(CyCu.encoder[7].weight.data)
        E_VCA_init = torch.from_numpy(E_VCA_init).unsqueeze(2).unsqueeze(3).float()
        E_VCA_init = E_VCA_init.to(device)
        model_dict = CyCu.state_dict()
        model_dict['decoder1.0.weight'] = E_VCA_init
        model_dict['decoder2.0.weight'] = E_VCA_init
        CyCu.load_state_dict(model_dict)

        loss_func = nn.MSELoss(size_average=True, reduce=True, reduction='mean')
        optimizer3 = torch.optim.Adam(CyCu.parameters(), lr=LR, weight_decay=weight_decay_param)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=20, gamma=0.8)
        apply_clamp_inst1 = NonZeroClipper()
        # gg = gg.reshape(rCol // 5, nCol // 5, 5, 5, Channel).transpose(0, 2, 1, 3, 4)
        # gg = gg.reshape(rCol, nCol, Channel)
        # gg = torch.tensor(gg).to(device)
        # x = gg.permute(2,0,1).unsqueeze(0)
        for epoch in range(EPOCH):
            CyCu.train().cuda()
            # x = x.cuda()
            abu_est1, re_result1, abu_est2, re_result2, end1 = CyCu(x)
            if epoch == EPOCH - 1:
                temp_result = torch.clamp(re_result1.detach(), 0, 1)
            loss_sumtoone = criterionSumToOne(abu_est1) + criterionSumToOne(abu_est2)
            # End = end1.squeeze(-1).squeeze(-1)
            # O = x.mean(dim=(2, 3))  # (1, 25)
            # O = O.view(25, 1)
            # misi = torch.norm(End - O, 'fro') ** 2

            loss_re = beta * loss_func(re_result1, x) + (1 - beta) * loss_func(x, re_result2)
            loss_abu = delta * loss_func(abu_est1, abu_est2)
            loss_sad = 0.5 * reconstruction_SADloss(x, re_result1) + 0.5 * reconstruction_SADloss(x, re_result2)
            mean_data = x.mean(dim=(2, 3), keepdim=True).permute(1, 0, 2, 3)
            loss_enm = loss_func(mean_data, end1)
            loss_sparse = criterionSparse(abu_est1) + criterionSparse(abu_est2)
            # vol
            em_bar = end1.mean(dim=1, keepdim=True)  # (25,1,1,1)
            diff = end1 - em_bar  # broadcast
            loss_minvol = (diff ** 2).mean()
            criterion_sad = SADLoss()
            loss_em_sad = criterion_sad(end1.mean(dim=1, keepdim=True), mean_data)
            if dataset in ['simple']:

                total_loss = loss_re + loss_abu + loss_sumtoone + loss_sad + loss_sparse + 0.01 * loss_enm
            elif dataset in ['complex']:
                total_loss = loss_re + loss_abu + loss_sumtoone + loss_sad + loss_sparse + 0.001 * loss_enm
            elif dataset in ['Urban']:
                total_loss = loss_re + loss_abu + loss_sumtoone + loss_sad + loss_sparse + 0.1 * loss_enm
            elif dataset in ['Checkerboard']:
                total_loss = loss_re + loss_abu + loss_sumtoone + loss_sad + loss_sparse + 0.001 * loss_enm
                # total_loss = loss_re + loss_abu + loss_sumtoone + loss_sad + loss_sparse
            else:
                # loss_sad = 0.5 * reconstruction_SADloss(x, re_result1) + 0.5 * reconstruction_SADloss(x, re_result2)
                # loss_sparse = criterionSparse(abu_est1) + criterionSparse(abu_est2)
                total_loss = loss_re + loss_abu + loss_sumtoone + loss_sparse + loss_sad + loss_enm
            # print(total_loss.cpu().data.numpy())
            optimizer3.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(CyCu.parameters(), max_norm=10, norm_type=1)
            optimizer3.step()

            CyCu.decoder1.apply(apply_clamp_inst1)
            CyCu.decoder2.apply(apply_clamp_inst1)

        if epoch % 100 == 0:
            print('Epoch:', epoch, '| train loss: %.4f' % total_loss.cpu().data.numpy(),
                  '| abu loss: %.4f' % loss_abu.cpu().data.numpy(),
                  '| sumtoone loss: %.4f' % loss_sumtoone.cpu().data.numpy(),
                  '| re loss: %.4f' % loss_re.cpu().data.numpy())

        scheduler.step()

        with torch.no_grad():
            CyCu.eval()
            abu_est1, re_result1, abu_est2, re_result2, end1 = CyCu(x)
            endmember_hat = CyCu.state_dict()["decoder1.0.weight"]
    else:
        # endmember_hat = E_VCA_init
        E_VCA_init = torch.from_numpy(E_VCA_init).unsqueeze(2).unsqueeze(3).float().cuda()
        endmember_hat = E_VCA_init

    # EM_hat = E_VCA_init
    EM_hat = endmember_hat
    # RE = re_result1
    RE = Undip_Y
    EM_hat3 = EM_hat.squeeze(-1).squeeze(-1)
    EM_hat2 = EM_hat.repeat(1, 1, rCol, nCol)
    # EM_hat2 = EM_hat.squeeze(-1).squeeze(-1).unsqueeze(0)

    abundance = UNMambaLinear(num_band=Channel, d_model=16,
                              num_endm=z_dim, num_queries_times=50, ds=16,
                              dropout=5e-2, hsi=RE, train_mean=True, row=rCol,
                              col=nCol, M0=EM_hat3
                              ).to(device)

    my_loss = My_Loss(weight_mse=1,
                      weight_sad=1,
                      weight_endm=0)
    optimizerA = torch.optim.Adam(abundance.parameters(), lr=3e-3, weight_decay=0)
    schedulerA = torch.optim.lr_scheduler.StepLR(optimizerA, step_size=45, gamma=0.9)

    kernel = build_kernel(device, dtype)
    for E_step in range(iters):

        if E_step == 0:
            RE = RE
        else:
            # LPF




            RE = residual_re(RE, Undip_Y, full_mask, kernel, alpha=1)
            Y = RE.cpu().numpy()

            Y = Y.reshape(25, 305 // 5, 5, 305 // 5, 5)
            Y = Y.transpose(1, 3, 0, 2, 4)
            Y = Y.reshape(-1, 25, 5, 5)
            chaos = spectral_chaos_batch(Y)
            threshold = np.percentile(chaos, 50)
            mask = chaos <= threshold
            crop_Y = Y[mask, :, :, :]
            remain = Y[mask, :, :, :].transpose(0, 2, 3, 1)
            all_ems = []
            Endmember = remain.reshape(-1, 25)
            EM_hat, _, _ = hyperVca(Endmember.T, P)
            temp_EM = EM_hat
            # FCLS_a = FCLSU(EM_hat, fcls_y.T, 0.01)
            # FCLS_a = FCLS_a.clone().detach()
            # FCLS_a = torch.reshape(FCLS_a, (P, rCol, nCol)).unsqueeze(0).to(device)
            sub_patch = remain.shape[0] // 2
            if base == 'Huston':
                n1 = 17
                n2 = 13
                # n1 = 21
                # n2 = 21
            if base == 'Apex':
                n1 = 22
                n2 = 11
            if base in ['Jasper', 'simple', 'complex']:
                # n1 = 20
                # n2 = 10
                n1 = 20
                n2 = 10
            if base == 'Urban':
                n1 = 1861
                n2 = 1
            if base == 'Checkerboard':
                n1 = 14
                n2 = 56
                # n1 = 61
                # n2 = 61
            remain = remain.reshape(n1, n2, 5, 5, Channel).transpose(0, 2, 1, 3, 4)
            remain = remain.reshape(n1 * 5, n2 * 5, Channel)
            crop_data = remain.reshape(-1, Channel)
            remain = torch.tensor(remain).to(device)
            x = remain.permute(2, 0, 1).unsqueeze(0)
            E_VCA_init = EM_hat
            if base not in ['k']:
                estimation_my = E_VCA_init
                CyCu = AutoEncoder(z_dim, Channel).cuda()
                nn.init.kaiming_normal_(CyCu.encoder[0].weight.data)
                nn.init.kaiming_normal_(CyCu.encoder[4].weight.data)
                nn.init.kaiming_normal_(CyCu.encoder[7].weight.data)
                E_VCA_init = torch.from_numpy(E_VCA_init).unsqueeze(2).unsqueeze(3).float()
                E_VCA_init = E_VCA_init.to(device)
                model_dict = CyCu.state_dict()
                model_dict['decoder1.0.weight'] = E_VCA_init
                model_dict['decoder2.0.weight'] = E_VCA_init
                CyCu.load_state_dict(model_dict)

                loss_func = nn.MSELoss(size_average=True, reduce=True, reduction='mean')
                optimizer3 = torch.optim.Adam(CyCu.parameters(), lr=LR, weight_decay=weight_decay_param)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=20, gamma=0.8)
                apply_clamp_inst1 = NonZeroClipper()
                # gg = gg.reshape(rCol // 5, nCol // 5, 5, 5, Channel).transpose(0, 2, 1, 3, 4)
                # gg = gg.reshape(rCol, nCol, Channel)
                # gg = torch.tensor(gg).to(device)
                # x = gg.permute(2,0,1).unsqueeze(0)
                for epoch in range(EPOCH):
                    CyCu.train().cuda()
                    # x = x.cuda()
                    abu_est1, re_result1, abu_est2, re_result2, end1 = CyCu(x)
                    if epoch == EPOCH - 1:
                        temp_result = torch.clamp(re_result1.detach(), 0, 1)
                    loss_sumtoone = criterionSumToOne(abu_est1) + criterionSumToOne(abu_est2)
                    # End = end1.squeeze(-1).squeeze(-1)
                    # O = x.mean(dim=(2, 3))  # (1, 25)
                    # O = O.view(25, 1)
                    # misi = torch.norm(End - O, 'fro') ** 2

                    loss_re = beta * loss_func(re_result1, x) + (1 - beta) * loss_func(x, re_result2)
                    loss_abu = delta * loss_func(abu_est1, abu_est2)
                    loss_sad = 0.5 * reconstruction_SADloss(x, re_result1) + 0.5 * reconstruction_SADloss(x, re_result2)
                    mean_data = x.mean(dim=(2, 3), keepdim=True).permute(1, 0, 2, 3)
                    loss_enm = loss_func(mean_data, end1)
                    loss_sparse = criterionSparse(abu_est1) + criterionSparse(abu_est2)
                    # vol
                    em_bar = end1.mean(dim=1, keepdim=True)  # (25,1,1,1)
                    diff = end1 - em_bar  # broadcast
                    loss_minvol = (diff ** 2).mean()
                    criterion_sad = SADLoss()
                    loss_em_sad = criterion_sad(end1.mean(dim=1, keepdim=True), mean_data)
                    if dataset in ['simple']:

                        total_loss = loss_re + loss_abu + loss_sumtoone + loss_sad + loss_sparse + 0.01 * loss_enm
                    elif dataset in ['complex']:
                        total_loss = loss_re + loss_abu + loss_sumtoone + loss_sad + loss_sparse + 0.001 * loss_enm
                    elif dataset in ['Urban']:
                        total_loss = loss_re + loss_abu + loss_sumtoone + loss_sad + loss_sparse + 0.1 * loss_enm
                    elif dataset in ['Checkerboard']:
                        total_loss = loss_re + loss_abu + loss_sumtoone + loss_sad + loss_sparse + 0.001 * loss_enm
                        # total_loss = loss_re + loss_abu + loss_sumtoone + loss_sad + loss_sparse
                    else:
                        # loss_sad = 0.5 * reconstruction_SADloss(x, re_result1) + 0.5 * reconstruction_SADloss(x, re_result2)
                        # loss_sparse = criterionSparse(abu_est1) + criterionSparse(abu_est2)
                        total_loss = loss_re + loss_abu + loss_sumtoone + loss_sparse + loss_sad + loss_enm
                    # print(total_loss.cpu().data.numpy())
                    optimizer3.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(CyCu.parameters(), max_norm=10, norm_type=1)
                    optimizer3.step()

                    CyCu.decoder1.apply(apply_clamp_inst1)
                    CyCu.decoder2.apply(apply_clamp_inst1)

                if epoch % 100 == 0:
                    print('Epoch:', epoch, '| train loss: %.4f' % total_loss.cpu().data.numpy(),
                          '| abu loss: %.4f' % loss_abu.cpu().data.numpy(),
                          '| sumtoone loss: %.4f' % loss_sumtoone.cpu().data.numpy(),
                          '| re loss: %.4f' % loss_re.cpu().data.numpy())

                scheduler.step()

                with torch.no_grad():
                    CyCu.eval()
                    abu_est1, re_result1, abu_est2, re_result2, end1 = CyCu(x)
                    endmember_hat = CyCu.state_dict()["decoder1.0.weight"]
            else:
                # endmember_hat = E_VCA_init
                E_VCA_init = torch.from_numpy(E_VCA_init).unsqueeze(2).unsqueeze(3).float().cuda()
                endmember_hat = E_VCA_init
            EM_hat = endmember_hat
            EM_hat2 = EM_hat.repeat(1, 1, rCol, nCol)
            # RE = (1 - full_mask) * RE + full_mask * (Undip_Y - RE)
            # RE = (1 - full_mask) * RE + full_mask * (full_msfa + RE)
            # RE = (1-full_mask) * RE + full_mask * full_msfa
            # RE = RE.mean(dim = (0,1))
            # sparse_mask = full_mask * full_msfa
            # RE = IPPI_reconstruction(
            #     RE,
            #     sparse_mask,
            #     epoch_num=1
            # )
            # fcls_y = RE.detach().cpu().numpy().reshape(25,-1)
            # FCLS_a = FCLSU(temp_EM , fcls_y, 0.01)
            # FCLS_a = FCLS_a.clone().detach()
            # FCLS_a = torch.reshape(FCLS_a, (P, rCol, nCol)).unsqueeze(0).to(device)

        abundance.train()
        # if E_step == 1:
        #     abundance.y.requires_grad = True
        for epoch in range(0, 500):
            loss_print = 0

            temp_i = 0

            temp_i += 1
            pred_linear, abu_est, pred_endm, factor = abundance(RE,EM_hat2)

            loss = my_loss(RE, pred_linear, pred_endm, 1, pred_aban=abu_est)
            # loss_a = (abu_est - FCLS_a).square().sum() / FCLS_a.shape[2] / FCLS_a.shape[3]
            factor_smooth = scaling_smooth_loss(factor)
            scale_one = scale_regularization(factor)
            # loss = loss
            # loss = loss + 10 * loss_a + factor_smooth
            # loss = loss + 0.001 * loss_a + 0.01 * factor_smooth
            # loss = loss + 0.01 * factor_smooth + scale_one
            loss = loss + 0.05 * scale_one + 0.01 * factor_smooth
            optimizerA.zero_grad()
            loss.backward()
            optimizerA.step()
            loss_print += loss
            loss_print = loss_print / (temp_i + 1)

            print("Epoch: %d/%d" % (epoch + 1, 500),
                  "| lr: %5f" % (optimizer.param_groups[0]['lr']),
                  "| loss: %.4f" % loss_print.cpu().data.numpy()
                  )

            schedulerA.step()


        abundance.eval()

        re_linear, abu_est, endm_, factor = abundance(RE,EM_hat2)
        # RE = (1 - 0.8) * RE + 0.2 * re_linear.detach()
        RE = re_linear.detach()

    A_hat = abu_est.detach().reshape(z_dim,-1)

    # A_hat = F.softmax(A_hat, dim=0)
    if dataset in ['simple','complex']:
        A_hat[A_hat < 0.1] = 0
        A_hat = A_hat / (A_hat.sum(dim=0, keepdim=True))
    # A_hat[A_hat < 0.05] = 0
    # A_hat[A_hat < 0.1] = 0
    EM_hat = endm_.detach().cpu().numpy()
    # result = x.mean(dim=(-2, -1))
    # EM_hat = EM_hat.mean(dim=(-2, -1), keepdim=True)
    EM_hat = EM_hat.mean(axis=(-2, -1), keepdims=True)
    # EM_hat = EM_hat.detach().cpu().numpy()

    EM_hat =EM_hat.reshape(25,P) * max_value
    a5 = EM_hat
    # A_hat = FCLSU(EM_hat.T,y,0.01)
    # scio.savemat('my_data.mat', {'a': y, 'b': EM_hat})
    toc = time()
    elapsed_time = tic - toc
    print(f"ÔËÐÐÊ±¼ä: {elapsed_time:.4f} Ãë")
    E_VCA_init = E_VCA_init * max_value

    return A_hat, EM_hat, A_true, M_true, P, rCol, nCol,E_VCA_init, factor




if __name__ == '__main__':
    # folder = '/home/Chengmohan/文档/jiehun/database/paperdata'
    # Endmember_folder = '/home/Chengmohan/文档/jiehun/Snapshot_Image_Demosaicing_Unmixing_Completion-main/Results'
    # folder = '/home/Chengmohan/文档/jiehun/database/IPPI_old'
    # folder = '/home/Chengmohan/文档/jiehun/result/cmh_results/final/change2/FIX_final/Our/mypaper_experiment/data/noisy/simple'
    # folder = '/home/Chengmohan/文档/jiehun/result/cmh_results/final/change2/FIX_final/Our/mypaper_experiment/data/noisy/simple'
    # folder = '/home/Chengmohan/文档/jiehun/database/OUR'
    # folder = '/home/Chengmohan/文档/jiehun/database/DC2_noisy/IPPI'
    folder = '/home/Chengmohan/文档/jiehun/result/cmh_results/final/change2/FIX_final/Our/mypaper_experiment/data/Urban6'
    mat_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith('.mat')
    ]
    picture_num = len(mat_files)
    # save_path = '/home/Chengmohan/文档/jiehun/result/cmh_results/final/change2/FIX_final/Our'
    # save_path = '/home/Chengmohan/文档/jiehun/result/cmh_results/final/change2/FIX_final/reject_resub/simple_noisy'
    save_path = '/home/Chengmohan/文档/jiehun/result/cmh_results/final/change2/FIX_final/Our/mypaper_experiment/revise_visualize/ablitaton'
    for cmh in mat_files:
        # if cmh in ['/home/Chengmohan/文档/jiehun/database/IPPI_old/simple.mat','/home/Chengmohan/文档/jiehun/database/IPPI_old/complex.mat']:
        #     continue
        # if cmh != '/home/Chengmohan/文档/jiehun/result/cmh_results/final/change2/FIX_final/Our/mypaper_experiment/data/noisy/simple/simple_20.mat':
        #     continue
        if cmh != '/home/Chengmohan/文档/jiehun/result/cmh_results/final/change2/FIX_final/Our/mypaper_experiment/data/Urban6/Urban.mat':
            continue
        # if cmh != '/home/Chengmohan/文档/jiehun/result/cmh_results/final/change2/FIX_final/Our/mypaper_experiment/data/noisy/simple/simple_20.mat':
        #     continue
        # seeds = [1202, 6585, 7413, 1938, 8477, 1834, 6210, 9679, 8116, 7974]
        it = [1,2,3,4,5,6,7,8,9,10]

        # seeds = [5932,9642,4267,6718,7027,6479,811,350,1546,473] # simple
        # seeds = [574,7047,1989,9303,5408,2372,4879,7478,4394,8836]  # complex
        # seeds = [2566,1775,6161,90,3337,3011,4088,280,5073,1106]  # Apex
        # seeds = [18,2970,6386,3644,5634,2963,496,7397,5494,3812]  # Huston
        # seeds = [8881,6830,9434,4037,9472,7595,6749,2225,3015,9835]  # Jasper
        # seeds = [7488,9414,5397,9692,4756,3949,9069,339,1349,89]  # Urban
        print(cmh)
        filename = os.path.basename(cmh)  # "example.mat"
        # È¥µô .mat ºó×º````
        filename_no_ext = os.path.splitext(filename)[0]  # "example"
        # filename_no_ext = 'simple'
        if filename_no_ext == 'Apex':
            # seeds = [2566, 1775, 6161, 90, 3337, 3011, 4088, 280, 5073, 1106]  # Apex
            seeds = [3280, 7643, 1065, 2019, 2533, 1019, 562, 6057, 3007, 5647, 5982, 5068, 5989, 2443, 7050, 9484,
                     6852, 6344, 3675, 8162]
        elif filename_no_ext == 'Urban':
            # seeds = [7488, 9414, 5397, 9692, 4756, 3949, 9069, 339, 1349, 89]
            # seeds = [361,2136,2760,3753,2774,9543,255,907,4687,3980]
            seeds = [9348, 6629, 4534, 9439, 6210, 1686, 3479, 3633, 4602, 9633]
            # seeds = [1628,1671,6600,7122,4687,4448,1595,3980,2291,1105,361,2136,2760,3753,6994,2268,2774,9543,255,907]
        elif filename_no_ext == 'Jasper':
            # seeds = [8881, 6830, 9434, 4037, 9472, 7595, 6749, 2225, 3015, 9835]
            seeds = [3145,3717,7595,6749,2225,3015,9835,9945,7047,1989]
            # seeds = [3145,3717,5071,4879,6532,6402,6830,9434,4037,9472,7595,6749,2225,3015,9835,9945,7047,1989,9303,5408]
        elif filename_no_ext == 'Huston':
            # seeds = [18, 2970, 6386, 3644, 5634, 2963, 496, 7397, 5494, 3812]
            seeds = [6447,6634,2960,1144,8864,6528,9489,298,3372,8848,3581,3256,7344,9669,9595,892,3705,1782,8088,1506]
            # seeds = [8156,765,549,9264,9779,6444,1003,865,6558,8921]
        elif filename_no_ext == 'simple':
            # seeds = [5932,9642,4267,6718,7027,6479,811,350,1546,473]
            seeds = [2136,9905,3753,2268,2274,9543,255,907,6600,3942]
            # seeds = [6751, 1279, 6057, 3007, 5647, 5982, 5068, 6884, 6989, 1059, 682, 8667, 7999, 3076, 9559, 5414,
            #          2019, 2533, 1019, 562]
        elif filename_no_ext == 'complex':
            # seeds = [9945,7047,1989,9303,5408,2372,4879,7478,4394,8836]
            seeds = [775, 7968, 6567, 3807, 4650, 5854, 4234, 2813, 9394, 8123, 2889, 9147, 8251, 2880, 8029, 9832,
                     4942,
                     195, 9757, 795]
        elif filename_no_ext == 'Checkerboard':
            # seeds = [9945,7047,1989,9303,5408,2372,4879,7478,4394,8836]
            seeds = [775, 7968, 6567, 3807, 4650, 5854, 4234, 2813, 9394, 8123, 2889, 9147, 8251, 2880, 8029, 9832,
                     4942,
                     195, 9757, 795]
        else:
            break
        # seeds = [9106,6600,9757,6567,2868,9867,9798,6143,6623,255]
        # seeds = [2679,9106,2151,9798,6143,6623,9641,7056,9543,255,907,6600,9757,795,6567,2868,7824,1092,9867,3362]
        # seeds = [2679,2151,6143,7059,907,9757,2868,7824,1092,3362]
        # seeds = [2136,9905,3753,2268,2774,9543,255,907,6600,3942]
        # for k in range(10):
        for k in range(2,3):
            temp_save_path = os.path.join(save_path, filename_no_ext,str(k))
            os.makedirs(temp_save_path, exist_ok=True)
            tv_path = os.path.join(temp_save_path, 'TV.xlsx')
            for j in range(10):
                seed = seeds[j]
                # seed = 1741
                # seed = random.randint(1, 10000)
                print("Random Seed: ", seed)
                # È¡×îºóÒ»²¿·ÖÎÄ¼þÃû


                # # # xlsx
                excel_path = filename_no_ext + '.xlsx'
                mat_path = filename_no_ext + '.mat'
                # Æ´½ÓÂ·¾¶
                temp_file_name = filename_no_ext + str(j)
                full_path = os.path.join(temp_save_path, temp_file_name)
                os.makedirs(full_path, exist_ok=True)
                #
                excel_path_seed = full_path + '/' + 'sedd.xlsx'
                excel_path = os.path.join(full_path, excel_path)
                mat_path = os.path.join(full_path, mat_path)
                scales = 0.3
                EI_path = full_path + '/' + 'EI.xlsx'

                out_LR_np, estimation_E,true_A,true_E, rmax, nr1,nc1,E_VCA_init,factor= train(path=cmh,base = filename_no_ext, sad_scale = scales, seeds = seed, iters = it[k], EI_path = EI_path)
                factor = factor.detach().cpu().numpy().reshape(rmax,nr1,nc1)
                E_VCA_init = E_VCA_init.detach().cpu().numpy().squeeze(-1).squeeze(-1)
                out_LR_np = out_LR_np.cpu().numpy()
                sad, sad_mean, sad_std = mymertics.my_sad(estimation_E, true_E)
                # Abudances
                # RMSE p x N
                estimation_A = out_LR_np
                # true_A = A_true_np.reshape(rmax, nr1 * nc1)
                total_rmse, individual_rmse, rmse_mean, rmse_std = mymertics.my_rmse(estimation_A, true_A)
                # mer p x N
                mer_value = mymertics.mer(true_A, estimation_A)
                # a mae p x N
                mae_value = mymertics.abundance_mae(true_A, estimation_A)
                # aad
                angles, mean_aad, std_aad = mymertics.aad(true_A, estimation_A)
                # aid
                divergence, mean_aid, std_aid = mymertics.aid(true_A, estimation_A)
                row = {
                    'sad': sad,
                    'sad_mean': sad_mean,
                    'sad_std': sad_std,
                    'total_rmse': total_rmse,
                    'individual_rmse': individual_rmse,
                    'rmse_mean': rmse_mean,
                    'rmse_std': rmse_std,
                    'mer': mer_value,
                    'mae': mae_value,
                    'angles': angles,
                    'mean_aad': mean_aad,
                    'std_aad': std_aad,
                    'divergence': divergence,
                    'mean_aid': mean_aid,
                    'std_aid': std_aid
                }
                append_metrics_to_excel(row, excel_path)
                mat_dict = {
                    'A_true': true_A,
                    'E_true': true_E,
                    'E_E': estimation_E,
                    'E_A': estimation_A,
                    'initital':E_VCA_init,
                    'factor':factor
                }
                scio.savemat(mat_path,mat_dict)
                estimation_A = estimation_A.reshape(rmax,nr1,nc1)
                factor = factor.reshape(rmax,nr1,nc1)
                true_A = true_A.reshape(rmax,nr1,nc1)
                plots.plot_scaling_each_endmember(factor,
                                     save_dir=full_path)
                plots.plot_abundance(true_A.transpose((1,2,0)), estimation_A.transpose((1, 2, 0)),
                                     rmax,
                                     save_dir=full_path)
                plots.plot_endmembers(true_E, estimation_E,rmax,
                                      save_dir=full_path)
                mean_sad_value, sad_list = mean_sad(true_E, estimation_E)
                append_row(seed, mean_sad_value,excel_path_seed)
                # append_row(seed,TV,tv_path)
