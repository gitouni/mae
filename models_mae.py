# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed
from typing import Union, Tuple

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size:Union[int, Tuple[int]]=224, patch_size:Union[int, Tuple[int]]=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, use_cls_token=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.in_chans = in_chans
        self.use_cls_token = use_cls_token

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)  # Input can be non-squared image or patch
        num_patches = self.patch_embed.num_patches
        total_num_patches = num_patches + 1 if use_cls_token else num_patches
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, total_num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, total_num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.grid_size[0], cls_token=self.use_cls_token)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.grid_size[0], cls_token=self.use_cls_token)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        if self.use_cls_token:
            torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m:Union[nn.Linear, nn.LayerNorm]):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def patch_size(self):
        return self.patch_embed.proj.kernel_size

    @property
    def grid_size(self):
        return self.patch_embed.grid_size

    @property
    def img_patch_dim(self):
        patch_size = self.patch_size
        return patch_size[0] * patch_size[1] * self.in_chans

    def patchify(self, imgs:torch.Tensor) -> torch.Tensor:
        """convert an image into transformer patches

        Args:
            imgs (torch.Tensor): (N, C, F, T)

        Returns:
            torch.Tensor: (N, L, patch_size[0]*patch_size[1]*C)
        """
        ph, pw = self.patch_size
        h, w = self.grid_size
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, ph, w, pw))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, self.img_patch_dim))
        return x

    def unpatchify(self, x:torch.Tensor, in_chans:Union[int, None]=None) -> torch.Tensor:
        """unpatchify a patched image into a noraml one

        Args:
            x (torch.Tensor): (N, L, patch_size[0]*patch_size[1]*C)
            in_chans (Union[int, None], optional): C. Defaults to None (self.in_chans).

        Returns:
            torch.Tensor: unpatched image: (N, C, F, T)
        """
        if in_chans is None:
            in_chans = self.in_chans
        ph, pw = self.patch_size
        h, w = self.grid_size
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, ph, pw, in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], in_chans, h * ph, h * pw))
        return imgs

    def random_masking(self, x:torch.Tensor, mask_ratio:float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        if mask_ratio == 0:
            mask = torch.zeros([N, L], device=x.device)
            ids_restore = torch.arange(L).to(torch.int)
            return x, mask, ids_restore
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def fixed_masking(self, x:torch.Tensor, masked_patch:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """fixed masking input patches x

        Args:
            x (torch.Tensor): (N, L, D)
            masked_patch (torch.Tensor): (N, L) 0 is keep, 1 is remove

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: x_masked, mask, ids_restore
        """
        N, L, D = x.shape  # batch, length, dim
        ids_shuffle = torch.argsort(masked_patch.reshape(N, -1), dim=1)  # (N, L)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        len_keep = (masked_patch == 0).sum(dim=1).min()  # int
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x:torch.Tensor, mask_ratio:float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """forward encoder with random masking

        Args:
            x (torch.Tensor): (N, C, H, W), input image
            mask_ratio (float): ratio of masked patches

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: x, mask, ids_restore
        """
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        if self.use_cls_token:
            x = x + self.pos_embed[:, 1:, :]
        else:
            x = x + self.pos_embed
        # masking: length -> length * mask_ratio
        
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        if self.use_cls_token:
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def inference_encoder(self, x:torch.Tensor, mask:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """forward encoder with fixed masking

        Args:
            x (torch.Tensor): (N, C, H, W), input image
            mask (torch.Tensor): (N, 1, H, W) or (N, H, W), input mask (0 means keep, 1 means remove)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: x, mask, ids_restore
        """
        N, L, D = x.shape  # batch, length, dim
        mask_ = mask.clone().detach()
        if mask_.ndim == 3:
            mask_ = mask_[:,None,...]  # (N, 1, H, W)
        mask_ = self.patchify(mask_)  # (N, L, D)
        masked_patch:torch.Tensor = torch.max(mask_, dim=-1)[0] # (N, L) 
        ids_shuffle = torch.argsort(masked_patch.reshape(N, -1), dim=1)  # (N, L)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        len_keep = (masked_patch[0] == 0).sum()  # int
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_decoder(self, x:torch.Tensor, ids_restore:torch.Tensor):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        if self.use_cls_token:
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        else:
            x = x_
        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        if self.use_cls_token:
            x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs:torch.Tensor, pred:torch.Tensor, mask:torch.Tensor):
        """
        imgs: [N, C, H, W]
        pred: [N, L, ph*pw*C]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs:torch.Tensor, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    def fixed_forward(self, imgs:torch.Tensor, masks:torch.Tensor):
        latent, mask, ids_restore = self.inference_encoder(imgs, masks)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
