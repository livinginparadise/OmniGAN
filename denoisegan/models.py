import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.nn.utils.parametrizations import spectral_norm

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    def _math_sdpa_ctx():
        return sdpa_kernel([SDPBackend.MATH])
except ImportError:
    def _math_sdpa_ctx():
        try:
            return torch.nn.attention.sdpa_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False
            )
        except Exception:
            return contextlib.nullcontext()


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def gen_conv2d(*args, **kwargs):
    m = nn.Conv2d(*args, **kwargs)
    nn.init.trunc_normal_(m.weight, std=0.02)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)
    return m


def disc_conv2d(*args, **kwargs):
    m = nn.Conv2d(*args, **kwargs)
    nn.init.trunc_normal_(m.weight, std=0.02)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)
    return spectral_norm(m)


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        with autocast(x.device.type, enabled=False):
            xf = x.float()
            mean = xf.mean(dim=1, keepdim=True)
            var = xf.var(dim=1, keepdim=True, unbiased=False)
            xn = (xf - mean) / torch.sqrt(var + self.eps)
        xn = xn.to(x.dtype)
        return xn * self.weight[None, :, None, None] + self.bias[None, :, None, None]


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    rt = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    rt.floor_()
    return x.div(keep_prob) * rt


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        return drop_path(x, self.p, self.training)


class NAFMixer(nn.Module):
    def __init__(self, dim, dw_expand=2):
        super().__init__()
        hidden = dim * dw_expand
        self.pw_in = gen_conv2d(dim, hidden, 1, bias=True)
        self.dw = gen_conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=True)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            gen_conv2d(hidden // 2, hidden // 2, 1, bias=True),
        )
        self.pw_out = gen_conv2d(hidden // 2, dim, 1, bias=True)

    def forward(self, x):
        x = self.pw_in(x)
        x = self.dw(x)
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * x2
        x = x.clamp(-256.0, 256.0)  # prevent runaway activations from unbounded product
        x = x * self.sca(x)
        x = self.pw_out(x)
        return x


class MDTA(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = gen_conv2d(dim, dim * 3, 1, bias=False)
        self.dw_qkv = gen_conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=False)
        self.proj = gen_conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.dw_qkv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        head_dim = C // self.num_heads
        q = q.view(B, self.num_heads, head_dim, H * W)
        k = k.view(B, self.num_heads, head_dim, H * W)
        v = v.view(B, self.num_heads, head_dim, H * W)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        with autocast(x.device.type, enabled=False):
            temp = self.temperature.float().clamp(0.01, 10.0)  # prevent attention collapse/explosion
            attn = (q.float() @ k.float().transpose(-2, -1)) * temp
            attn = attn.softmax(dim=-1)
            out = (attn @ v.float()).to(x.dtype)
        out = out.reshape(B, C, H, W)
        return self.proj(out)


class GDFN(nn.Module):
    def __init__(self, dim, expansion=2.0):
        super().__init__()
        hidden = int(dim * expansion)
        self.pw_in = gen_conv2d(dim, hidden * 2, 1, bias=False)
        self.dw = gen_conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=False)
        self.pw_out = gen_conv2d(hidden, dim, 1, bias=False)

    def forward(self, x):
        x = self.pw_in(x)
        x = self.dw(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = x.clamp(-256.0, 256.0)  # prevent runaway activations from unbounded gating product
        return self.pw_out(x)


class OmniBlockV2(nn.Module):
    def __init__(self, dim, use_mdta=False, num_heads=4, ffn_expansion=2.0,
                 drop_path_rate=0.0, layer_scale=1e-4):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        if use_mdta:
            self.mixer = MDTA(dim, num_heads=num_heads)
        else:
            self.mixer = NAFMixer(dim, dw_expand=2)
        self.gamma1 = nn.Parameter(torch.full((dim,), layer_scale))
        self.drop_path1 = DropPath(drop_path_rate)

        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim, expansion=ffn_expansion)
        self.gamma2 = nn.Parameter(torch.full((dim,), layer_scale))
        self.drop_path2 = DropPath(drop_path_rate)

    def forward(self, x):
        x = x + self.drop_path1(self.gamma1[None, :, None, None] * self.mixer(self.norm1(x)))
        x = x + self.drop_path2(self.gamma2[None, :, None, None] * self.ffn(self.norm2(x)))
        return x


class Downsample(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2)
        self.conv = gen_conv2d(in_c * 4, out_c, 1, bias=False)

    def forward(self, x):
        return self.conv(self.unshuffle(x))


class Upsample(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = gen_conv2d(in_c, out_c * 4, 1, bias=False)
        self.shuffle = nn.PixelShuffle(2)
        self.smooth = gen_conv2d(out_c, out_c, kernel_size=3, padding=1, groups=out_c, bias=False)

    def forward(self, x):
        return self.smooth(self.shuffle(self.conv(x)))


class AttentionGate(nn.Module):
    def __init__(self, F_int):
        super().__init__()
        self.w_g = gen_conv2d(F_int, F_int, 1, bias=False)
        self.w_x = gen_conv2d(F_int, F_int, 1, bias=False)
        self.ag_prelu = nn.PReLU(num_parameters=F_int, init=0.1)
        self.q = gen_conv2d(F_int, 1, 1, bias=False)

    def forward(self, x_enc, x_dec):
        g = self.w_g(x_dec)
        x = self.w_x(x_enc)
        alpha = torch.sigmoid(self.q(self.ag_prelu(g + x)))
        return x_enc * alpha


class BlockAttnRes(nn.Module):
    def __init__(self, channels, num_blocks):
        super().__init__()
        self.channels = channels
        self.num_blocks = num_blocks
        self.d_attn = max(channels // 4, 16)
        self.queries = nn.Parameter(torch.randn(num_blocks, self.d_attn) * 0.02)
        self.key_proj = nn.Linear(channels, self.d_attn, bias=False)
        nn.init.trunc_normal_(self.key_proj.weight, std=0.02)
        self.gate = nn.Parameter(torch.full((num_blocks,), -5.0))

    def forward(self, block_outputs, block_idx):
        if len(block_outputs) == 1:
            return block_outputs[0]
        pooled = [F.adaptive_avg_pool2d(v, 1).squeeze(-1).squeeze(-1) for v in block_outputs]
        pooled = torch.stack(pooled, dim=0)
        keys = self.key_proj(pooled)
        query = self.queries[block_idx].unsqueeze(0).unsqueeze(0)
        scale = self.d_attn ** (-0.5)
        scores = (keys * query).sum(dim=-1) * scale
        alpha = F.softmax(scores, dim=0)
        stacked = torch.stack(block_outputs, dim=0)
        alpha = alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        attn_out = (alpha * stacked).sum(dim=0)
        g = torch.sigmoid(self.gate[block_idx])
        standard = block_outputs[-1]
        return g * attn_out + (1 - g) * standard


class EncoderAttnRes(nn.Module):
    def __init__(self, channel_list):
        super().__init__()
        self.num_stages = len(channel_list)
        self.d_attn = 32
        self.key_projs = nn.ModuleList([nn.Linear(c, self.d_attn, bias=False) for c in channel_list])
        for proj in self.key_projs:
            nn.init.trunc_normal_(proj.weight, std=0.02)
        self.queries = nn.Parameter(torch.randn(self.num_stages - 1, self.d_attn) * 0.02)
        self.gate = nn.Parameter(torch.full((self.num_stages - 1,), -5.0))

    def forward(self, encoder_features):
        pooled = []
        for i, feat in enumerate(encoder_features):
            p = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
            pooled.append(self.key_projs[i](p))
        keys = torch.stack(pooled, dim=1)
        skip_weights = []
        for dec_idx in range(self.num_stages - 1):
            query = self.queries[dec_idx]
            scores = (keys * query.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
            scores = scores * self.d_attn ** (-0.5)
            alpha = F.softmax(scores, dim=-1)
            stage_weight = alpha[:, dec_idx]
            stage_weight = stage_weight * self.num_stages
            g = torch.sigmoid(self.gate[dec_idx])
            final_weight = g * stage_weight + (1 - g) * 1.0
            skip_weights.append(final_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
        return skip_weights


class DenoiseGenerator(nn.Module):
    def __init__(self, channels=(48, 96, 192, 320, 448), nc=3,
                 enc_blocks=(2, 2, 2, 2, 2), dec_blocks=(2, 2, 2, 2),
                 bottleneck_blocks=4, num_heads=8, drop_path_rate=0.1,
                 use_checkpoint=True):
        super().__init__()
        c1, c2, c3, c4, c5 = channels
        self.use_checkpoint = use_checkpoint

        depths = list(enc_blocks) + [bottleneck_blocks] + list(dec_blocks)
        total = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0.0, drop_path_rate, total)]
        idx = 0

        def make_blocks(dim, n, use_mdta=False, heads=num_heads):
            nonlocal idx
            blocks = []
            for _ in range(n):
                blocks.append(OmniBlockV2(dim, use_mdta=use_mdta, num_heads=heads,
                                          ffn_expansion=2.0,
                                          drop_path_rate=dp_rates[idx]))
                idx += 1
            return nn.Sequential(*blocks)

        self.in_proj = gen_conv2d(nc, c1, 3, 1, 1, bias=False)

        self.enc_0 = make_blocks(c1, enc_blocks[0], use_mdta=False)
        self.down_1 = Downsample(c1, c2)
        self.enc_1 = make_blocks(c2, enc_blocks[1], use_mdta=False)
        self.down_2 = Downsample(c2, c3)
        self.enc_2 = make_blocks(c3, enc_blocks[2], use_mdta=False)
        self.down_3 = Downsample(c3, c4)
        self.enc_3 = make_blocks(c4, enc_blocks[3], use_mdta=True)
        self.down_4 = Downsample(c4, c5)
        self.enc_4 = make_blocks(c5, enc_blocks[4], use_mdta=True)

        self.bottleneck_layers = nn.ModuleList([
            OmniBlockV2(c5, use_mdta=True, num_heads=num_heads,
                        ffn_expansion=2.0, drop_path_rate=dp_rates[idx + i])
            for i in range(bottleneck_blocks)
        ])
        idx += bottleneck_blocks
        self.bottleneck_attn = BlockAttnRes(channels=c5, num_blocks=bottleneck_blocks)

        self.encoder_attn = EncoderAttnRes([c1, c2, c3, c4, c5])

        self.up_4 = Upsample(c5, c4)
        self.ag_4 = AttentionGate(c4)
        self.mix_4 = gen_conv2d(c4 * 2, c4, 1, bias=False)
        self.dec_3 = make_blocks(c4, dec_blocks[0], use_mdta=True)

        self.up_3 = Upsample(c4, c3)
        self.ag_3 = AttentionGate(c3)
        self.mix_3 = gen_conv2d(c3 * 2, c3, 1, bias=False)
        self.dec_2 = make_blocks(c3, dec_blocks[1], use_mdta=False)

        self.up_2 = Upsample(c3, c2)
        self.ag_2 = AttentionGate(c2)
        self.mix_2 = gen_conv2d(c2 * 2, c2, 1, bias=False)
        self.dec_1 = make_blocks(c2, dec_blocks[2], use_mdta=False)

        self.up_1 = Upsample(c2, c1)
        self.ag_1 = AttentionGate(c1)
        self.mix_1 = gen_conv2d(c1 * 2, c1, 1, bias=False)
        self.dec_0 = make_blocks(c1, dec_blocks[3], use_mdta=False)

        self.out_norm = LayerNorm2d(c1)
        self.to_out = gen_conv2d(c1, nc, 3, 1, 1, bias=False)
        self.apply(weights_init)
        nn.init.zeros_(self.to_out.weight)

    def _ckpt(self, module, x):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, x):
        e0 = self._ckpt(self.enc_0, self.in_proj(x))
        e1 = self._ckpt(self.enc_1, self.down_1(e0))
        e2 = self._ckpt(self.enc_2, self.down_2(e1))
        e3 = self._ckpt(self.enc_3, self.down_3(e2))
        e4 = self._ckpt(self.enc_4, self.down_4(e3))

        outs = []
        b = e4
        for i, blk in enumerate(self.bottleneck_layers):
            b = self._ckpt(blk, b)
            outs.append(b)
            if i + 1 < len(self.bottleneck_layers):
                b = self.bottleneck_attn(outs, block_idx=i + 1)

        skip_w = self.encoder_attn([e0, e1, e2, e3, e4])

        d3 = self.up_4(b)
        d3 = self.mix_4(torch.cat([self.ag_4(e3 * skip_w[3], d3), d3], dim=1))
        d3 = self._ckpt(self.dec_3, d3)

        d2 = self.up_3(d3)
        d2 = self.mix_3(torch.cat([self.ag_3(e2 * skip_w[2], d2), d2], dim=1))
        d2 = self._ckpt(self.dec_2, d2)

        d1 = self.up_2(d2)
        d1 = self.mix_2(torch.cat([self.ag_2(e1 * skip_w[1], d1), d1], dim=1))
        d1 = self._ckpt(self.dec_1, d1)

        d0 = self.up_1(d1)
        d0 = self.mix_1(torch.cat([self.ag_1(e0 * skip_w[0], d0), d0], dim=1))
        d0 = self._ckpt(self.dec_0, d0)

        residual = self.to_out(self.out_norm(d0))
        out = (x + residual).clamp(-1.0, 1.0)
        return out


class NoiseTranslator(nn.Module):
    """Bias-free residual CNN that maps arbitrary/unknown noise into the noise
    distribution the frozen denoiser was trained on. Fully bias-free (no conv
    bias, no normalization) + ReLU + a global residual, so it is scale-
    equivariant: f(a*x) = a*f(x). That equivariance is exactly what makes a
    denoising front-end robust to noise magnitudes it never saw in training."""

    def __init__(self, nc=3, dim=64, num_blocks=10):
        super().__init__()
        self.head = nn.Conv2d(nc, dim, 3, padding=1, bias=False)
        body = []
        for _ in range(num_blocks):
            body += [nn.Conv2d(dim, dim, 3, padding=1, bias=False),
                     nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*body)
        self.tail = nn.Conv2d(dim, nc, 3, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.zeros_(self.tail.weight)  # start as identity (output == input)

    def forward(self, x):
        h = F.relu(self.head(x), inplace=True)
        h = self.body(h)
        return (x + self.tail(h)).clamp(-1.0, 1.0)


class DegradationAwareMod(nn.Module):
    """DASR-style conditioning: from the degradation/hint embedding, predict a
    per-sample depthwise dynamic conv kernel AND a channel scale. Richer than
    FiLM (which only rescales channels) because the dynamic kernel lets the noise
    example reshape the SPATIAL filtering. Both are identity-initialized, so the
    block reduces to a plain conv at start (warm-start preserves the backbone)."""

    def __init__(self, dim, emb, k=5):
        super().__init__()
        self.dim = dim
        self.k = k
        self.kernel_mlp = nn.Linear(emb, dim * k * k)
        self.scale_mlp = nn.Linear(emb, dim)
        nn.init.zeros_(self.kernel_mlp.weight)
        nn.init.zeros_(self.kernel_mlp.bias)
        nn.init.zeros_(self.scale_mlp.weight)
        nn.init.zeros_(self.scale_mlp.bias)
        idk = torch.zeros(dim, 1, k, k)
        idk[:, 0, k // 2, k // 2] = 1.0
        self.register_buffer('id_kernel', idk)

    def forward(self, h, e):
        B, C, H, W = h.shape
        k = self.k
        dk = self.kernel_mlp(e).view(B, C, 1, k, k)
        kernel = (self.id_kernel.unsqueeze(0) + dk).reshape(B * C, 1, k, k)
        y = F.conv2d(h.reshape(1, B * C, H, W), kernel, padding=k // 2, groups=B * C)
        y = y.reshape(B, C, H, W)
        scale = 1.0 + self.scale_mlp(e).view(B, C, 1, 1)
        return y * scale


class GuidedNoiseTranslator(nn.Module):
    """Hint-conditioned noise translator. Same bias-free residual backbone as
    NoiseTranslator, but each block is modulated by a degradation-aware module
    (DASR-style dynamic conv + channel scale) driven by an embedding computed
    from a single reference (noisy_patch, clean_patch) + its normalized position.
    The human cleans ONE 256x256 patch; the model reads it as an example of the
    unknown noise and translates the whole image accordingly.

    Conditioning is identity-initialized => at start the backbone behaves exactly
    like the unconditioned NoiseTranslator, so a Phase-1 (60k) translator can be
    warm-started in and only the hint-handling is learned in Phase 2."""

    def __init__(self, nc=3, dim=64, num_blocks=10, emb=128, mod_k=5):
        super().__init__()
        self.head = nn.Conv2d(nc, dim, 3, padding=1, bias=False)
        self.blocks = nn.ModuleList(
            [nn.Conv2d(dim, dim, 3, padding=1, bias=False) for _ in range(num_blocks)])
        self.mods = nn.ModuleList(
            [DegradationAwareMod(dim, emb, k=mod_k) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(dim, nc, 3, padding=1, bias=False)

        # hint encoder: [noisy_patch ; clean_patch ; residual] -> embedding
        self.enc = nn.Sequential(
            nn.Conv2d(nc * 3, 32, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.enc_mlp = nn.Sequential(
            nn.Linear(128 + 2, 256), nn.ReLU(inplace=True),
            nn.Linear(256, emb),
        )
        # contrastive projection head (used only by the Phase-2 contrastive loss,
        # not at inference): pulls embeddings of same-noise patches together.
        self.proj = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Linear(128, 64),
        )

        for m in [self.head, *self.blocks]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.zeros_(self.tail.weight)
        for m in self.enc:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def noise_feature(self, hint_noisy, hint_clean):
        """Position-free noise descriptor (B,128) — the target of contrastive learning."""
        r = hint_noisy - hint_clean
        return self.enc(torch.cat([hint_noisy, hint_clean, r], dim=1)).flatten(1)

    def project(self, v):
        return self.proj(v)

    def encode_hint(self, hint_noisy, hint_clean, pos):
        v = self.noise_feature(hint_noisy, hint_clean)
        return self.enc_mlp(torch.cat([v, pos], dim=1))

    def forward(self, x, hint_noisy, hint_clean, pos):
        e = self.encode_hint(hint_noisy, hint_clean, pos)
        h = F.relu(self.head(x), inplace=True)
        for conv, mod in zip(self.blocks, self.mods):
            y = mod(conv(h), e)
            h = F.relu(y, inplace=True)
        return (x + self.tail(h)).clamp(-1.0, 1.0)


class BlurPool2d(nn.Module):
    def __init__(self, channels, stride=2):
        super().__init__()
        self.stride = stride
        filt = torch.tensor([1., 4., 6., 4., 1.])
        filt2 = filt[:, None] * filt[None, :]
        filt2 = filt2 / filt2.sum()
        self.register_buffer('filt', filt2[None, None, :, :].repeat(channels, 1, 1, 1))
        self.pad = nn.ReflectionPad2d(2)
        self.channels = channels

    def forward(self, x):
        x = self.pad(x)
        return F.conv2d(x, self.filt, stride=self.stride, groups=self.channels)


class UNetDiscriminatorSN(nn.Module):
    def __init__(self, in_channels=3, num_feat=48):
        super().__init__()
        self.conv0 = disc_conv2d(in_channels, num_feat, 3, 1, 1, bias=True)

        self.conv1 = disc_conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.blur1 = BlurPool2d(num_feat * 2, stride=2)
        self.conv2 = disc_conv2d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.blur2 = BlurPool2d(num_feat * 4, stride=2)
        self.conv3 = disc_conv2d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.blur3 = BlurPool2d(num_feat * 8, stride=2)

        self.conv4 = disc_conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False)
        self.conv5 = disc_conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False)
        self.conv6 = disc_conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False)

        self.conv7 = disc_conv2d(num_feat, num_feat, 3, 1, 1, bias=False)
        self.conv8 = disc_conv2d(num_feat, num_feat, 3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        x0 = F.leaky_relu(self.conv0(x), 0.2, inplace=True)
        x1 = F.leaky_relu(self.blur1(self.conv1(x0)), 0.2, inplace=True)
        x2 = F.leaky_relu(self.blur2(self.conv2(x1)), 0.2, inplace=True)
        x3 = F.leaky_relu(self.blur3(self.conv3(x2)), 0.2, inplace=True)

        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), 0.2, inplace=True)
        x4 = x4 + x2

        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), 0.2, inplace=True)
        x5 = x5 + x1

        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), 0.2, inplace=True)
        x6 = x6 + x0

        out = F.leaky_relu(self.conv7(x6), 0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), 0.2, inplace=True)
        out = self.conv9(out)
        return out


class DinoProjectedHead(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.conv1 = disc_conv2d(in_dim, hidden, 1)
        self.conv2 = disc_conv2d(hidden, hidden, 3, 1, 1)
        self.conv3 = nn.Conv2d(hidden, 1, 1)

    def forward(self, tokens_2d):
        h = F.leaky_relu(self.conv1(tokens_2d), 0.2, inplace=True)
        h = F.leaky_relu(self.conv2(h), 0.2, inplace=True)
        return self.conv3(h)


class DinoProjectedDiscriminator(nn.Module):
    def __init__(self, layers=(2, 5, 8, 11), patch_size=14,
                 dino_name='dinov2_vits14'):
        super().__init__()
        try:
            backbone = torch.hub.load('facebookresearch/dinov2', dino_name, trust_repo=True)
        except Exception:
            backbone = torch.hub.load('facebookresearch/dinov2', dino_name)
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()
        self.backbone = backbone
        self.layers = tuple(layers)
        self.patch_size = patch_size
        embed_dim = getattr(backbone, 'embed_dim', 384)
        self.embed_dim = embed_dim
        self.heads = nn.ModuleList([DinoProjectedHead(embed_dim) for _ in layers])
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _preprocess(self, x):
        x = (x + 1.0) * 0.5
        x = x.clamp(0.0, 1.0)
        x = (x - self.mean) / self.std
        h, w = x.shape[-2:]
        nh = (h // self.patch_size) * self.patch_size
        nw = (w // self.patch_size) * self.patch_size
        if nh != h or nw != w:
            x = F.interpolate(x, size=(nh, nw), mode='bilinear', align_corners=False)
        return x

    def forward(self, x):
        x = self._preprocess(x)
        B = x.shape[0]
        H = x.shape[-2] // self.patch_size
        W = x.shape[-1] // self.patch_size
        self.backbone.eval()
        with _math_sdpa_ctx():
            feats_list = self.backbone.get_intermediate_layers(
                x, n=self.layers, reshape=False, return_class_token=False, norm=True
            )
        outs = []
        for feat, head in zip(feats_list, self.heads):
            if feat.dim() == 3:
                tokens_2d = feat.transpose(1, 2).reshape(B, -1, H, W)
            else:
                tokens_2d = feat
            outs.append(head(tokens_2d))
        return outs


class DualDiscriminator(nn.Module):
    def __init__(self, num_feat=64, dino_layers=(2, 5, 8, 11)):
        super().__init__()
        self.unet = UNetDiscriminatorSN(in_channels=3, num_feat=num_feat)
        self.dino = DinoProjectedDiscriminator(layers=dino_layers)

    def trainable_parameters(self):
        params = list(self.unet.parameters())
        for h in self.dino.heads:
            params.extend(list(h.parameters()))
        return params

    def forward(self, x):
        unet_logit = self.unet(x)
        dino_logits = self.dino(x)
        return unet_logit, dino_logits


def aggregate_logits(unet_logit, dino_logits):
    parts = [unet_logit.mean(dim=(1, 2, 3))]
    for d in dino_logits:
        parts.append(d.mean(dim=(1, 2, 3)))
    return torch.stack(parts, dim=0).mean(dim=0)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = DenoiseGenerator(channels=(48, 96, 192, 320, 448)).to(device)
    G.eval()
    x = torch.randn(2, 3, 256, 256, device=device)
    with torch.no_grad():
        y = G(x)
    print(f"G in:  {tuple(x.shape)}")
    print(f"G out: {tuple(y.shape)}")
    n_params = sum(p.numel() for p in G.parameters())
    print(f"G params: {n_params/1e6:.2f} M")

    D = UNetDiscriminatorSN().to(device)
    with torch.no_grad():
        d_out = D(x)
    print(f"D out:   {tuple(d_out.shape)}")
    nd = sum(p.numel() for p in D.parameters())
    print(f"D params: {nd/1e6:.2f} M")
