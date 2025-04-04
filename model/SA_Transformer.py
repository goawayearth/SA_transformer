import torch
import math
from torch import nn


# 图片分割为窗口
def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    # 不能整除会报错
    assert (H % window_size == 0 and W % window_size == 0), "Invalid window_size."

    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


# 恢复图片形状
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, -1, H, W)
    return x


# 自注意力机制
class Attention(nn.Module):
    '''
    input: [B, N, C]
    '''

    def __init__(self, num_heads, hidden_size, dropout):
        super(Attention, self).__init__()

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.head_size = int(self.hidden_size / self.num_heads)
        self.all_head_size = self.num_heads * self.head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        self.out = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.proj_dropout = nn.Dropout(self.dropout)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        # x: [B, N, C] -> [B, N, num_heads, head_size]
        #              -> [B, num_heads, N, head_size]
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 计算查询、键、值的线性变换
    # 计算注意力分数，softmax归一化
    def forward(self, hidden_state):
        # 线性
        q_expend = self.query(hidden_state)
        k_expend = self.key(hidden_state)
        v_expend = self.value(hidden_state)

        q = self.transpose_for_scores(q_expend)
        k = self.transpose_for_scores(k_expend)
        v = self.transpose_for_scores(v_expend)

        attn = q @ (k.transpose(-1, -2))
        attn = attn / math.sqrt(self.head_size)
        attn_prob = self.softmax(attn)
        attn_prob = self.attn_dropout(attn_prob)

        context = attn_prob @ v
        context = context.permute(0, 2, 1, 3).contiguous()  # [B, N, num_heads, head_size]
        new_context_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_context_shape)  # [B, N, C]
        opt = self.out(context)
        opt = self.proj_dropout(opt)

        return opt


class Mlp(nn.Module):

    def __init__(self, hidden_size, mlp_dim, dropout):
        super(Mlp, self).__init__()

        self.hidden_size = hidden_size
        self.mlp_dim = mlp_dim

        self.fc1 = nn.Linear(self.hidden_size, self.mlp_dim)
        self.fc2 = nn.Linear(self.mlp_dim, self.hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# 一个Transformer块
class Transblock(nn.Module):
    '''
    input x: [Nw*B, Ws**2, C]
    output : [Nw*B, Ws**2, C]
    '''

    def __init__(self, num_heads, hidden_size, mlp_dim, dropout):
        super(Transblock, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.mlp_dim = mlp_dim

        self.attn = Attention(self.num_heads, self.hidden_size, dropout)
        self.mlp = Mlp(self.hidden_size, self.mlp_dim, dropout)
        self.ln = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        nB, N, C = x.shape

        buffer_1 = x
        x = self.ln(x)
        x = self.attn(x)
        x = self.dropout(x) + buffer_1

        buffer_2 = x
        x = self.ln(x)
        x = self.mlp(x)
        opt = self.dropout(x) + buffer_2

        return opt


# 一个Transformer层
class Layer(nn.Module):
    '''
    input x: [B, C, H, W]
    output : [B, C, H, W]
    '''

    def __init__(self, window_size, n_block, num_heads, hidden_size, mlp_dim, dropout):
        super(Layer, self).__init__()

        self.window_size = window_size
        self.n_block = n_block
        self.transblocks = nn.ModuleList([
            Transblock(num_heads, hidden_size, mlp_dim, dropout)
            for i in range(self.n_block)])

    def forward(self, x):
        B, C, H, W = x.shape

        # window partition
        x = window_partition(x, self.window_size)  # [Nw*B, Ws, Ws, C]
        nB, Ws, Ws, C = x.shape
        x = x.reshape(nB, Ws * Ws, C)  # [Nw*B, Ws*Ws, C]

        # transblocks
        for t_block in self.transblocks:
            x = t_block(x)
        x = x.reshape(nB, Ws, Ws, C)

        # reverse
        x = window_reverse(x, self.window_size, H, W)

        return x


# 不同尺寸进行多次Transformer
class Multi_Wsize_Layer_v2(nn.Module):

    def __init__(self, num_heads, hidden_size, mlp_dim, dropout):
        super(Multi_Wsize_Layer_v2, self).__init__()

        self.layer_w2 = Layer(window_size=2,
                              n_block=1,
                              num_heads=num_heads,
                              hidden_size=hidden_size,
                              mlp_dim=mlp_dim,
                              dropout=dropout)

        self.layer_w4 = Layer(window_size=4,
                              n_block=1,
                              num_heads=num_heads,
                              hidden_size=hidden_size,
                              mlp_dim=mlp_dim,
                              dropout=dropout)

        self.layer_w8 = Layer(window_size=8,
                              n_block=1,
                              num_heads=num_heads,
                              hidden_size=hidden_size,
                              mlp_dim=mlp_dim,
                              dropout=dropout)

        # 卷积层
        self.combine = nn.Sequential(
            nn.Conv2d(in_channels=int(hidden_size * 3),
                      out_channels=hidden_size,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True)
        )

    # 三个尺寸的Transformer的输出进行拼接，然后通过卷积层
    def forward(self, x):
        x = torch.cat((self.layer_w2(x), self.layer_w4(x), self.layer_w8(x)), 1)
        opt = self.combine(x)
        return opt


# 多次卷积
class Residual_block(nn.Module):
    def __init__(self, nch_in, nch_out):
        super(Residual_block, self).__init__()
        self.align = nn.Conv2d(in_channels=nch_in,
                               out_channels=nch_out,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        self.dualconv = nn.Sequential(
            nn.Conv2d(in_channels=nch_out,
                      out_channels=nch_out,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=nch_out),
            nn.ELU(),
            nn.Conv2d(in_channels=nch_out,
                      out_channels=nch_out,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=nch_out)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.align(x)
        x1 = self.dualconv(x)
        opt = self.relu(torch.add(x, x1))
        return opt


def trans_down(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=4,
                  stride=2,
                  padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ELU()
    )


def trans_up(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=4,
                           stride=2,
                           padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ELU()
    )


class VFTransformer_v3(nn.Module):

    def __init__(self, nch_enc):
        super(VFTransformer_v3, self).__init__()
        self.nch_enc = nch_enc
        self.nch_dec = nch_enc[::-1]

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.td = nn.ModuleList()
        self.tu = nn.ModuleList()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=1)
        for i in range(len(self.nch_enc)):
            self.encoder.append(Multi_Wsize_Layer_v2(num_heads=self.nch_enc[i] // 2,
                                                     hidden_size=self.nch_enc[i],
                                                     mlp_dim=self.nch_enc[i] * 4,
                                                     dropout=0.))
            if i == 0:
                self.decoder.append(Residual_block(self.nch_dec[i], self.nch_dec[i + 1]))
            elif i == len(self.nch_enc) - 1:
                self.decoder.append(Residual_block(self.nch_dec[i] * 2, 2))
            else:
                self.decoder.append(Residual_block(self.nch_dec[i] * 2, self.nch_dec[i + 1]))

            if i < len(self.nch_enc) - 1:
                self.td.append(trans_down(self.nch_enc[i], self.nch_enc[i + 1]))
                self.tu.append(trans_up(self.nch_dec[i + 1], self.nch_dec[i + 1]))

    def forward(self, x):
        cats = []
        # x = self.conv2(x)

        x = x[:, :2, :, :]
        # encoder
        for i in range(len(self.nch_enc) - 1):
            layer_opt = self.encoder[i](x)
            x = self.td[i](layer_opt)
            cats.append(layer_opt)

        # bottom
        latent = self.encoder[-1](x)
        layer_opt = self.decoder[0](latent)

        # decoder
        for i in range(len(self.nch_dec) - 1):
            x = self.tu[i](layer_opt)
            x = torch.cat([x, cats[-1 - i]], dim=1)
            layer_opt = self.decoder[i + 1](x)
            # 假设卷积层的输出通道数仍然是 2


        y_pred = self.conv1(layer_opt)
        y_pred = torch.sigmoid(y_pred)
        return y_pred

