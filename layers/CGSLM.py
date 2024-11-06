from torch import nn
import torch
from .common import batched_index_select,batched_index_select
import torch.nn.functional as F

KMEANS_INIT_ITERS = 10


def exists(val):
    return val is not None


def similarity(x, means):
    return torch.einsum('bhld,hcd->bhlc', x, means)


def dists_and_buckets(x, means):
    dists = similarity(x, means)
    _, buckets = torch.max(dists, dim=-1)
    return dists, buckets


def expand_dim(t, dim, k):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def batched_bincount(index, num_classes, dim=-1):
    shape = list(index.shape)
    shape[dim] = num_classes
    out = index.new_zeros(shape)
    out.scatter_add_(dim, index, torch.ones_like(index, dtype=index.dtype))
    return out

def ema(old, new, decay):
    if not exists(old):
        return new
    return old * decay + new * (1 - decay)

def kmeans_iter(x, means, buckets=None):
    b, h, l, d, dtype, num_clusters = *x.shape, x.dtype, means.shape[1]

    if not exists(buckets):
        _, buckets = dists_and_buckets(x, means)

    bins = batched_bincount(buckets, num_clusters).sum(0, keepdim=True)
    zero_mask = bins.long() == 0

    means_ = buckets.new_zeros(b, h, num_clusters, d, dtype=dtype)
    # todo 好像是所有的向量相加为质心的向量
    means_.scatter_add_(-2, expand_dim(buckets, -1, d), x)
    means_ = F.normalize(means_.sum(0, keepdim=True), dim=-1).type(dtype)

    means = torch.where(zero_mask.unsqueeze(-1), means, means_)
    means = means.squeeze(0)
    return means



class Kmeans(nn.Module):
    def __init__(self, n_rounds, qk_dim, n_clusters, ema_decay=0.999):
        super().__init__()
        self.n_rounds = n_rounds
        self.n_clusters = n_clusters
        self.ema_decay = ema_decay
        # todo LSH rotated vectors [N, n_hashes, H*W, hash_buckets]
        self.register_buffer('means', torch.randn(n_rounds, n_clusters, qk_dim))
        self.register_buffer('initted', torch.tensor(False))
        self.num_new_means = 0
        self.new_means = None

    @torch.no_grad()
    def init(self, x):
        # todo 理解x的shape [batch, n_rounds, length, dim]
        if self.initted:
            return
        _, h, _, d, device, dtype = *x.shape, x.device, x.dtype

        n_clusters = self.means.shape[1]

        # 一个min batch内的所有特征聚合在一起
        means = x.transpose(0, 1).contiguous().view(h, -1, d)
        # 一个min batch内所有特征的数量
        n_samples = means.shape[1]

        if n_samples >= n_clusters:
            indices = torch.randperm(n_samples, device=device)[:n_clusters]
        else:
            indices = torch.randint(0, n_samples, (n_clusters,), device=device)

        means = means[:, indices]

        for _ in range(KMEANS_INIT_ITERS):
            # todo kmeans更新迭代函数，暂时可以不用细究
            means = kmeans_iter(x, means)

        self.num_new_means = 0
        self.means.data.copy_(means)
        self.initted.data.copy_(torch.tensor(True))

    def forward(self, x, update_means=False):
        x = expand_dim(x, 1, self.n_rounds)
        self.init(x)

        b, dtype = x.shape[0], x.dtype
        means = self.means.type(dtype)
        x = F.normalize(x, 2, dim=2).type(dtype)

        with torch.no_grad():
            dists, buckets = dists_and_buckets(x, means)

        if update_means:
            with torch.no_grad():
                means = kmeans_iter(x, means, buckets)
            # todo 这里控制kmeans更新速度，消融实验测试结果
            if self.ema_decay <= 0:
                ema_decay = self.num_new_means / (self.num_new_means + 1)
            else:
                ema_decay = self.ema_decay
            self.new_means = ema(self.new_means, means, ema_decay)
            self.num_new_means += 1

        offsets = torch.arange(self.n_rounds, device=x.device)
        offsets = torch.reshape(offsets * self.n_clusters, (1, -1, 1))
        bucket_codes = torch.reshape(buckets + offsets, (b, -1,))

        return bucket_codes




class CGSL(nn.Module):
    def __init__(self, channels=16,drop=0.0, reduction=1, n_clusters=128, window_size=1,
                ema_decay=0.9, n_rounds=1):
        super(CGSL, self).__init__()
        self.window_size = window_size
        self.n_rounds = n_rounds
        self.reduction = reduction
        self.kmeans = Kmeans(n_rounds, channels // reduction, n_clusters, ema_decay)
        self.x_embed = nn.Sequential(nn.Linear(channels, channels), nn.LayerNorm(channels), nn.ReLU())
        self.y_embed = nn.Sequential(nn.Linear(channels, channels), nn.LayerNorm(channels), nn.ReLU())
        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def add_adjacent_buckets(self, x):
        x_extra_back = torch.cat([x[:, -1:, ...], x[:, :-1, ...]], dim=1)
        x_extra_forward = torch.cat([x[:, 1:, ...], x[:, :1, ...]], dim=1)
        return torch.cat([x, x_extra_back, x_extra_forward], dim=2)

    def forward(self, inputs, training):

        N, _, _ = inputs.shape
        x_embed = self.x_embed(inputs)
        y_embed = self.y_embed(inputs)

        L, C = x_embed.shape[-2:]

        kmeans_codes = self.kmeans(x_embed, training)
        kmeans_codes = kmeans_codes.detach()

        _, indices = kmeans_codes.sort(dim=-1)
        _, undo_sort = indices.sort(dim=-1)
        mod_indices = indices

        x_att_buckets = batched_index_select(x_embed, mod_indices).reshape(N, -1, C)
        y_att_buckets = batched_index_select(y_embed, mod_indices).reshape(N, -1, C)


        x_att_buckets = torch.reshape(x_att_buckets, (N, -1, self.window_size, C))
        y_att_buckets = torch.reshape(y_att_buckets, (N, -1, self.window_size, C))
        x_match = F.normalize(x_att_buckets, p=2, dim=1, eps=5e-5)

        x_match = self.add_adjacent_buckets(x_match)
        y_att_buckets = self.add_adjacent_buckets(y_att_buckets)

        # unormalized attention score
        raw_score = torch.einsum('bkie,bkje->bkij', x_att_buckets, x_match)
        # softmax
        bucket_score = torch.logsumexp(raw_score, dim=-1, keepdim=True)
        score = torch.exp(raw_score - bucket_score)  # (after softmax) reuse
        bucket_score = torch.reshape(bucket_score, [N, -1])

        # attention
        ret = torch.einsum('bkij,bkje->bkie', score, y_att_buckets).squeeze(2)
        ret = batched_index_select(ret, undo_sort)
        bucket_score = bucket_score.gather(1, undo_sort).unsqueeze(-1)


        probs = nn.functional.softmax(bucket_score, dim=1)
        ret = (ret * probs)
        ret = self.alpha * inputs + (1 - self.alpha) * ret

        return ret


