# -*- coding: utf-8 -*-
# @Time    : 3/11/23 4:02 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : cav_mae.py

import os
os.environ['TORCH_HOME'] = './pretrained_models'
import random
import torch
import torch.nn as nn
import timm
import sys
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed, Block
from .pos_embed import get_2d_sincos_pos_embed

import sys

sys.path.append('/home/chenghao/Project/cav-mae')
from src.models.modeling_finetune import vit_base_patch16_160, vit_base_dim768_img224, vit_base_dim512_no_depth_patch16_160
from src.models.modeling_pretrain import PretrainVisionTransformerEncoder as video_encoder
from src.models.modeling_pretrain import PretrainVisionTransformerDecoder as video_decoder
from torchvision import transforms
from src.masking_generator import TubeMaskingGenerator, TubeWindowMaskingGenerator
from src.transforms import GroupNormalize, GroupMultiScaleCrop, IdentityTransform, Stack, ToTorchFormatTensor

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm1_a = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: 这里去掉了qk_scale
        # self.attn = Attention(
        #     dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm2_a = norm_layer(dim)
        self.norm2_v = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, modality=None):
        if modality == None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif modality == 'a':
            x = x + self.drop_path(self.attn(self.norm1_a(x)))
            x = x + self.drop_path(self.mlp(self.norm2_a(x)))
        elif modality == 'v':
            x = x + self.drop_path(self.attn(self.norm1_v(x)))
            x = x + self.drop_path(self.mlp(self.norm2_v(x)))
        return x

# our main proposed model, for pretraining only, for finetuning, use CAVMAEFT class
class MY_MODULE(nn.Module):
    """ MY Model
    """
    def __init__(self, img_size=224, audio_length=1024, patch_size=16, in_chans=3,
                 embed_dim=768, modality_specific_depth=11, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, tr_pos=False):
        super().__init__()
        print('A CAV-MAE Model')
        print('Use norm_pix_loss: ', norm_pix_loss)
        print('Learnable Positional Embedding: ', tr_pos)

        ###################################################
        #   mae parament #
        img_size=224
        patch_size=16
        encoder_in_chans=3
        encoder_num_classes=0
        encoder_embed_dim=768
        encoder_depth=12

        encoder_num_heads=12
        decoder_num_classes=1536 #  decoder_num_classes=768, 
        # decoder_embed_dim=512 # 相同的参数被注释， 冲突的参数被改名
        # decoder_depth=8
        V_decoder_num_heads=8
        # mlp_ratio=4.
        qkv_bias=False
        qk_scale=None
        drop_rate=0.
        attn_drop_rate=0.
        drop_path_rate=0. 
        # norm_layer=nn.LayerNorm 
        init_values=0.
        use_learnable_pos_emb=False
        tubelet_size=2
        self.num_frames=16
        self.input_size=224
        num_classes=0 # avoid the error from create_fn in timm
        in_chans=0 # avoid the error from create_fn in timm
        # me: for more attention types
        attn_type='joint'
        lg_region_size=(2, 2, 10)
        lg_first_attn_type='self'
        lg_third_attn_type='cross'
        # for local_global
        lg_attn_param_sharing_first_third=False
        lg_attn_param_sharing_all=False
        lg_no_second=False
        lg_no_third=False
        ###################################################

        # the encoder part
        # overide the timm package
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        timm.models.vision_transformer.Block = Block

        self.patch_embed_a = PatchEmbed(img_size, patch_size, 1, embed_dim)
        #self.patch_embed_v = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        self.patch_embed_a.num_patches = int(audio_length * 128 / 256)
        

        self.modality_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
        #self.modality_v = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed_a = nn.Parameter(torch.zeros(1, self.patch_embed_a.num_patches, embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding
        #self.pos_embed_v = nn.Parameter(torch.zeros(1, self.patch_embed_v.num_patches, embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding

        # audio-branch
        self.blocks_a = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(modality_specific_depth)])
        # visual-branch
        #self.blocks_v = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(modality_specific_depth)])
        self.blocks_maedfer = vit_base_patch16_160()
        # unified branch
        self.blocks_u = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(12-modality_specific_depth)])

        # mae-dfer block
        self.video_encoder = video_encoder(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=encoder_in_chans, 
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_learnable_pos_emb=use_learnable_pos_emb,
            attn_type=attn_type,
            lg_region_size=lg_region_size, lg_first_attn_type=lg_first_attn_type, # for local_global
            lg_third_attn_type=lg_third_attn_type,
            lg_attn_param_sharing_first_third=lg_attn_param_sharing_first_third,
            lg_attn_param_sharing_all=lg_attn_param_sharing_all,
            lg_no_second=lg_no_second, lg_no_third=lg_no_third,
        )
        self.project_video_avgpool =  nn.AdaptiveAvgPool1d(128)
        #print('video encoder', self.video_encoder)
        self.video_decoder = video_decoder(
            patch_size=patch_size, 
            num_patches=self.video_encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes, 
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=V_decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            tubelet_size=tubelet_size
        )
        #print('video decoder', self.video_decoder)

        print('Number of Audio Patches: {:d}, Visual Patches: {:d}'.format(self.patch_embed_a.num_patches, self.video_encoder.patch_embed.num_patches))
        # independent normalization layer for audio, visual, and audio-visual
        self.norm_a, self.norm_v, self.norm = norm_layer(embed_dim), norm_layer(embed_dim), norm_layer(embed_dim)

        # the decoder part
        # Project to lower dimension for the decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # token used for masking
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_modality_a = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        #self.decoder_modality_v = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed_a = nn.Parameter(torch.zeros(1, self.patch_embed_a.num_patches, decoder_embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding
        #self.decoder_pos_embed_v = nn.Parameter(torch.zeros(1, self.patch_embed_v.num_patches, decoder_embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        # project channel is different for two modality, use two projection head
        self.decoder_pred_a = nn.Linear(decoder_embed_dim, patch_size ** 2 * 1, bias=True)  # decoder to patch
        #self.decoder_pred_v = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        print('Audio Positional Embedding Shape:', self.pos_embed_a.shape)
        print('Visual Positional Embedding Shape:', self.video_encoder.pos_embed.shape)

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding, opt the cls token, add by myself
        pos_embed_a = get_2d_sincos_pos_embed(self.pos_embed_a.shape[-1], 8, int(self.patch_embed_a.num_patches/8), cls_token=False)
        self.pos_embed_a.data.copy_(torch.from_numpy(pos_embed_a).float().unsqueeze(0))
        
        #pos_embed_v = get_2d_sincos_pos_embed(self.pos_embed_v.shape[-1], int(self.patch_embed_v.num_patches ** .5), int(self.patch_embed_v.num_patches ** .5), cls_token=False)
        #self.pos_embed_v.data.copy_(torch.from_numpy(pos_embed_v).float().unsqueeze(0))

        decoder_pos_embed_a = get_2d_sincos_pos_embed(self.decoder_pos_embed_a.shape[-1], 8, int(self.patch_embed_a.num_patches/8), cls_token=False)
        self.decoder_pos_embed_a.data.copy_(torch.from_numpy(decoder_pos_embed_a).float().unsqueeze(0))

        #decoder_pos_embed_v = get_2d_sincos_pos_embed(self.decoder_pos_embed_v.shape[-1], int(self.patch_embed_v.num_patches ** .5), int(self.patch_embed_v.num_patches ** .5), cls_token=False)
        #self.decoder_pos_embed_v.data.copy_(torch.from_numpy(decoder_pos_embed_v).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed_a.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        #w = self.patch_embed_v.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.modality_a, std=.02)
        #torch.nn.init.normal_(self.modality_v, std=.02)
        torch.nn.init.normal_(self.decoder_modality_a, std=.02)
        #torch.nn.init.normal_(self.decoder_modality_v, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs, c, h, w, p=16):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * c))
        return x

    def unpatchify(self, x, c, h, w, p=16):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def random_masking_unstructured(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
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

    def video_masking(self, mask_ratio, window_size, mask_type, part_apply_symmetry, part_win_size, images): # 这个地方把数据增强（norma、transfom）的代码去掉了，你应该在dataset里面加回去
            """
            生成视频数据的掩码。

            参数:
            mask_ratio (float): 掩码的比例，范围在0到1之间。
            window_size (int): 窗口大小，用于确定掩码的范围。
            mask_type (str): 掩码类型，可以是'tube'或'part_window'。
            part_apply_symmetry (bool): 是否应用对称性，只在mask_type为'part_window'时有效。
            part_win_size (int): 部分窗口大小，只在mask_type为'part_window'时有效。

            返回:
            masked_position_generator (function): 一个函数，可以生成掩码位置。

            示例:
            >>> gen = masking_generator(0.5, 10, 'tube', False, 5)
            >>> mask_positions = gen()
            """
            
            if mask_type == 'tube':
                masked_position_generator = TubeMaskingGenerator(
                    window_size, mask_ratio
                )
            elif mask_type == 'part_window':
                print(f"==> Note: use 'part_window' masking generator (window_size={part_win_size[1:]}, apply_symmetry={part_apply_symmetry})")
                masked_position_generator = TubeWindowMaskingGenerator(
                    window_size, mask_ratio, win_size=part_win_size[1:], apply_symmetry=part_apply_symmetry
                )
            mask = masked_position_generator()
            print('image shape:',images.shape)
            print('mask shape:',mask.shape)
            
            
            return images, mask # TODO:mask的部分莫名其妙的，我只能说sunlicai代码写的和shi一样

    def random_masking_structured(self, x, mask_ratio, t=64, f=8, mode='time'):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        assert L == f * t
        noise = noise.reshape(N, f, t) # the audio patch is in shape [f,t], not [t,f]
        if mode == 'time':
            for i in range(N):
                mask_t_list = random.sample(range(t), int(t * mask_ratio))
                for k in mask_t_list:
                    noise[i, :, k] = 1.1  # large value will be removed
        elif mode == 'freq':
            for i in range(N):
                mask_f_list = random.sample(range(f), int(f * mask_ratio))
                for k in mask_f_list:
                    noise[i, k, :] = 1.1  # large value will be removed
        elif mode == 'tf':
            for i in range(N):
                mask_t_list = random.sample(range(t), int(t * mask_ratio * 0.7))
                for k in mask_t_list:
                    noise[i, :, k] = 1.1  # large value will be removed
            for i in range(N):
                mask_f_list = random.sample(range(f), int(f * mask_ratio * 0.7))
                for k in mask_f_list:
                    noise[i, k, :] = 1.1  # large value will be removed
        noise = noise.reshape(N, L)

        # sort noise for each sample, only need to manuplate these two ids_shuffle, ids_restore
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

    def forward_encoder(self, a, v, mask_ratio_a, mask_ratio_v, mask_mode='unstructured', video_mask_type='tube'):
        # embed patches
        a = a.unsqueeze(1)
        a = a.transpose(2, 3)
        a = self.patch_embed_a(a)
        a = a + self.pos_embed_a
        a = a + self.modality_a # 标识模态

        # v = self.patch_embed_v(v)
        # v = v + self.pos_embed_v
        # v = v + self.modality_v # 标识模态

        # by default, we always use unstructured masking
        if mask_mode == 'unstructured':
            a, mask_a, ids_restore_a = self.random_masking_unstructured(a, mask_ratio_a)
        # in ablation study, we tried time/freq/tf masking. mode in ['freq', 'time', 'tf']
        else:
            a, mask_a, ids_restore_a = self.random_masking_structured(a, mask_ratio_a, t=64, f=8, mode=mask_mode)

        # # visual branch always use unstructured masking
        # v, mask_v, ids_restore_v = self.random_masking_unstructured(v, mask_ratio_v)
        patch_size = self.video_encoder.patch_embed.patch_size
        print("Patch size = %s" % str(patch_size))
        window_size = (self.num_frames // 2, self.input_size // patch_size[0], self.input_size // patch_size[1]) # 8*16*16的window，一个图片切成16*16个patch，8是帧的一半

        v, mask_v = self.video_masking(mask_ratio_v, window_size=window_size, mask_type=video_mask_type, part_apply_symmetry=False, part_win_size=(2, 2, 10), images=v)

        ####################################################################
        
        _, _, T, _, _ = v.shape
        v = self.video_encoder(v, mask_v) # [B, N_vis, C_e]
        #v = self.project_video_avgpool(v.transpose(1, 2)).transpose(1, 2)
        print('after video encoder', v.shape)
        
        ####################################################################
        
        # audio and visual stream, independent blocks
        for blk in self.blocks_a:
            a = blk(a)

        
        print('a.shape',a.shape)
        #self.blocks_maedfer

        # for blk in self.blocks_v:
        #     v = blk(v)

        #v = self.blocks_maedfer(v)
        
        x = torch.cat((a, v), dim=1)
        print('x.shape',x.shape)
        # unified stream, shared blocks_u, but independent normalization layers
        for blk in self.blocks_u:
            x = blk(x)
        x = self.norm(x)

        for blk in self.blocks_u:
            ca = blk(a, 'a')
        ca = self.norm_a(ca)

        for blk in self.blocks_u:
            cv = blk(v, 'v')
        cv = self.norm_v(cv)

        #return x, mask_a, ids_restore_a, mask_v, ids_restore_v, ca, cv
        return x, mask_a, ids_restore_a, mask_v, ca, cv

    def forward_decoder(self, x, mask_a, ids_restore_a, mask_v, ids_restore_v):

        x = self.decoder_embed(x)

        # append mask tokens to sequence
        # mask_tokens_a in shape [B, #a_mask_token, mask_token_dim], get the number of masked samples from mask_a[0], which is the first example of the batch, all samples should have same number of masked tokens
        mask_tokens_a = self.mask_token.repeat(x.shape[0], int(mask_a[0].sum()), 1)
        a_ = torch.cat([x[:, :self.patch_embed_a.num_patches-int(mask_a[0].sum()), :], mask_tokens_a], dim=1)  # no cls token
        a_ = torch.gather(a_, dim=1, index=ids_restore_a.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # similar for the visual modality
        mask_tokens_v = self.mask_token.repeat(x.shape[0], int(mask_v[0].sum()), 1)
        v_ = torch.cat([x[:, self.patch_embed_a.num_patches-int(mask_a[0].sum()):, :], mask_tokens_v], dim=1)  # no cls token
        v_ = torch.gather(v_, dim=1, index=ids_restore_v.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # concatenate audio and visual tokens
        x = torch.cat([a_, v_], dim=1)

        decoder_pos_embed = torch.cat([self.decoder_pos_embed_a, self.decoder_pos_embed_v], dim=1)
        x = x + decoder_pos_embed

        # add modality indication tokens
        x[:, 0:self.patch_embed_a.num_patches, :] = x[:, 0:self.patch_embed_a.num_patches, :] + self.decoder_modality_a
        x[:, self.patch_embed_a.num_patches:, :] = x[:, self.patch_embed_a.num_patches:, :] + self.decoder_modality_v

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x_a = self.decoder_pred_a(x[:, :self.patch_embed_a.num_patches, :])
        # x_v = self.decoder_pred_v(x[:, self.patch_embed_a.num_patches:, :])

        # return audio and video tokens
        #return x_a, x_v
        return x_a

    def forward_contrastive(self, audio_rep, video_rep, bidirect_contrast=False):
        # calculate nce loss for mean-visual representation and mean-audio representation

        audio_rep = torch.nn.functional.normalize(audio_rep, dim=-1)
        video_rep = torch.nn.functional.normalize(video_rep, dim=-1)

        total = torch.mm(audio_rep, torch.transpose(video_rep, 0, 1)) / 0.05

        # by default we use single directional
        if bidirect_contrast == False:
            nce = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
            c_acc = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0), torch.arange(0, total.shape[0], device=audio_rep.device))) / total.shape[0]
            return nce, c_acc
        else:
            nce_1 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
            nce_2 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total.t(), dim=0)))
            c_acc_1 = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0), torch.arange(0, total.shape[0], device=audio_rep.device))) / total.shape[0]
            c_acc_2 = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total.t(), dim=0), dim=0), torch.arange(0, total.shape[0], device=audio_rep.device))) / total.shape[0]
            nce = (nce_1 + nce_2) / 2
            c_acc = (c_acc_1 + c_acc_2) / 2
            return nce, c_acc

    def forward_mae_loss(self, input, pred, mask, modality):
        if modality == 'a':
            # for audio, need to adjust the shape
            input = input.unsqueeze(1)
            input = input.transpose(2, 3)
            target = self.patchify(input, 1, int(input.shape[2]/self.patch_embed_a.patch_size[0]), int(input.shape[3]/self.patch_embed_a.patch_size[1]), 16)
        elif modality == 'v':
            target = self.patchify(input, 3, int(input.shape[2]/self.patch_embed_v.patch_size[0]), int(input.shape[3]/self.patch_embed_v.patch_size[1]), 16)

        # patch-wise normalization might minorly improve the classification performance, but will make the model lose inpainting function
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, audio, imgs, mask_ratio_a=0.75, mask_ratio_v=0.75, mae_loss_weight=1., contrast_loss_weight=0.01, mask_mode='unstructured'):
        # latent is used for reconstruction (mae), latent_c_{a,v} are used for contrastive learning
        #latent, mask_a, ids_restore_a, mask_v, ids_restore_v, latent_c_a, latent_c_v = self.forward_encoder(audio, imgs, mask_ratio_a, mask_ratio_v, mask_mode=mask_mode)
        latent, mask_a, ids_restore_a, mask_v, latent_c_a, latent_c_v = self.forward_encoder(audio, imgs, mask_ratio_a, mask_ratio_v, mask_mode=mask_mode)


        # B, N, C = x_vis.shape
        # # we don't unshuffle the correct visible token order, 
        # # but shuffle the pos embedding accorddingly.
        # expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        # pos_emd_vis = expand_pos_embed[~mask_v].reshape(B, -1, C)
        # pos_emd_mask = expand_pos_embed[mask_v].reshape(B, -1, C)
        # x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1) # [B, N, C_d]
        # x = self.video_decoder(x_full, pos_emd_mask.shape[1]) # [B, N_mask, 3 * 16 * 16]


        #TODO: 解码器，大麻烦啊
        # if mae loss is used
        if mae_loss_weight != 0:
            pred_a, pred_v = self.forward_decoder(latent, mask_a, ids_restore_a, mask_v, ids_restore_v)
            loss_mae_a = self.forward_mae_loss(audio, pred_a, mask_a, 'a')
            loss_mae_v = self.forward_mae_loss(imgs, pred_v, mask_v, 'v')
            loss_mae = mae_loss_weight * (loss_mae_a + loss_mae_v)
        else:
            loss_mae_a, loss_mae_v, loss_mae = torch.tensor(0.0, device=audio.device), torch.tensor(0.0, device=audio.device), torch.tensor(0.0, device=audio.device)

        # if contrastive loss is used
        if contrast_loss_weight != 0:
            # note this is single directional
            loss_c, c_acc = self.forward_contrastive(latent_c_a.mean(dim=1), latent_c_v.mean(dim=1))
            loss_c = contrast_loss_weight * loss_c
        else:
            loss_c, c_acc = torch.tensor(0.0, device=audio.device), torch.tensor(0.0, device=audio.device)

        loss = loss_mae + loss_c

        
        return loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc

    # used only for inpainting, ignore if inpainting is not of interest
    def forward_inpaint(self, audio, imgs, mask_ratio_a=0.75, mask_ratio_v=0.75, mask_mode='unstructured'):
        latent, mask_a, ids_restore_a, mask_v, ids_restore_v, latent_c_a, latent_c_v = self.forward_encoder(audio, imgs, mask_ratio_a, mask_ratio_v, mask_mode=mask_mode)
        pred_a, pred_v = self.forward_decoder(latent, mask_a, ids_restore_a, mask_v, ids_restore_v)  # [N, L, p*p*3]
        loss_pixel_a = self.forward_mae_loss(audio, pred_a, mask_a, 'a')
        loss_pixel_v = self.forward_mae_loss(imgs, pred_v, mask_v, 'v')
        return pred_a, pred_v, mask_a, mask_v, loss_pixel_a, loss_pixel_v

    # used for retrieval, ignore if retrieval is not of interest
    def forward_feat(self, a, v):
        # embed patches
        a = a.unsqueeze(1)
        a = a.transpose(2, 3)
        a = self.patch_embed_a(a)
        a = a + self.pos_embed_a
        a = a + self.modality_a

        v = self.patch_embed_v(v)
        v = v + self.pos_embed_v
        v = v + self.modality_v

        # the modality-specific stream
        for blk in self.blocks_a:
            a = blk(a)

        for blk in self.blocks_v:
            v = blk(v)

        # use modality specific normalization,
        for blk in self.blocks_u:
            a = blk(a, 'a')
        a = self.norm_a(a)

        for blk in self.blocks_u:
            v = blk(v, 'v')
        v = self.norm_v(v)
        return a, v

# the finetuned CAV-MAE model
class MY_FT(nn.Module):
    def __init__(self, label_dim, img_size=224, audio_length=1024, patch_size=16, in_chans=3,
                 embed_dim=768, modality_specific_depth=11, num_heads=12, mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, tr_pos=True, pretrained_mae_path=None):
        super().__init__()
        timm.models.vision_transformer.Block = Block
        print('Use norm_pix_loss: ', norm_pix_loss)

        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        timm.models.vision_transformer.Block = Block

        self.patch_embed_a = PatchEmbed(img_size, patch_size, 1, embed_dim)
        self.patch_embed_v = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        self.patch_embed_a.num_patches = int(audio_length * 128 / 256)
        print('Number of Audio Patches: {:d}, Visual Patches: {:d}'.format(self.patch_embed_a.num_patches, self.patch_embed_v.num_patches))

        self.modality_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_v = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed_a = nn.Parameter(torch.zeros(1, self.patch_embed_a.num_patches, embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding
        self.pos_embed_v = nn.Parameter(torch.zeros(1, self.patch_embed_v.num_patches, embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding

        self.blocks_a = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(modality_specific_depth)])
        self.blocks_v = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(modality_specific_depth)])
        self.blocks_u = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(12 - modality_specific_depth)])

        self.video_encoder = vit_base_dim768_img224() #vit_base_dim512_no_depth_patch16_160()
        self.norm_a = norm_layer(embed_dim)
        self.norm_v = norm_layer(embed_dim)
        self.norm = norm_layer(embed_dim)

        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, label_dim))

        self.initialize_weights(pretrained_mae_path)


        print('Audio Positional Embedding Shape:', self.pos_embed_a.shape)
        print('Visual Positional Embedding Shape:', self.pos_embed_v.shape)

    def get_patch_num(self, input_shape, stride):
        test_input = torch.zeros(1, 1, input_shape[0], input_shape[1])
        test_proj = torch.nn.Conv2d(1, 4, kernel_size=(16, 16), stride=(stride, stride))
        test_output = test_proj(test_input)
        print(test_output.shape)
        return test_output.shape[2], test_output[3], test_output[2] * test_output[2]

    def initialize_weights(self, pretrained_mae_path=None):
        pos_embed_a = get_2d_sincos_pos_embed(self.pos_embed_a.shape[-1], 8, int(self.patch_embed_a.num_patches/8), cls_token=False)
        self.pos_embed_a.data.copy_(torch.from_numpy(pos_embed_a).float().unsqueeze(0))

        pos_embed_v = get_2d_sincos_pos_embed(self.pos_embed_v.shape[-1], int(self.patch_embed_v.num_patches ** .5), int(self.patch_embed_v.num_patches ** .5), cls_token=False)
        self.pos_embed_v.data.copy_(torch.from_numpy(pos_embed_v).float().unsqueeze(0))

        w = self.patch_embed_a.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.patch_embed_v.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.modality_a, std=.02)
        torch.nn.init.normal_(self.modality_v, std=.02)

        # 加载预训练权重
        ckt = torch.load(pretrained_mae_path)
        # 例子：调整权重字典以匹配模型中的参数键
        # print(ckt.keys())

        msg = self.video_encoder.load_state_dict(ckt['model'], strict=False)  # 使用strict=False允许部分加载
        
        # # 获取所有权重的键
        # all_keys = set(ckt['model'].keys())

        # # 获取未加载的权重的键
        # missing_keys = set(msg.missing_keys)

        # # 计算成功加载的权重的键
        # loaded_keys = all_keys - missing_keys

        # # 打印成功加载的权重的键
        # print("Loaded keys:", loaded_keys)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, a, v, mode):
        # multi-modal fine-tuning, our default method for fine-tuning
        if mode == 'multimodal':
            ''' batchsize=32 embeding _dim=768 a_graph_size/(16*16)=512 v_graph_size/(16*16)=196
            raw a shape torch.Size([16, 1024, 128])
            raw v shape torch.Size([16, 3, 224, 224])
            raw a shape torch.Size([16, 1024, 128])
            raw v shape torch.Size([16, 3, 224, 224])
            a feature shape torch.Size([16, 512, 768])
            v feature shape torch.Size([16, 196, 768])
            a feature shape torch.Size([16, 512, 768])
            v feature shape torch.Size([16, 196, 768])
            '''
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            # v = self.patch_embed_v(v)
            # v = v + self.pos_embed_v
            # v = v + self.modality_v
            # print('after audio encoder shape', a.shape)
            # print('befoer video encoder shape',v.shape)
            v = self.video_encoder(v)
            # print('after video encoder shape',v.shape)

            
            for blk in self.blocks_a:
                a = blk(a)

            # for blk in self.blocks_v:
            #     v = blk(v)

            x = torch.cat((a, v), dim=1)

            for blk in self.blocks_u:
                x = blk(x)
            x = self.norm(x)

            x = x.mean(dim=1)
            x = self.mlp_head(x)
            return x

        # finetune with only audio (and inference with only audio when the model is finetuned with only audio)
        elif mode == 'audioonly':
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            for blk in self.blocks_a:
                a = blk(a)

            # note here uses the 'a' normalization, it is used in both training and inference, so it is fine
            for blk in self.blocks_u:
                a = blk(a, 'a')
            a = self.norm_a(a)
            x = a.mean(dim=1)
            x = self.mlp_head(x)
            return x

        # finetune with only image (and inference with only audio when the model is finetuned with only image)
        elif mode == 'videoonly':
            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            for blk in self.blocks_v:
                v = blk(v)

            # note here uses the 'v' normalization, it is used in both training and inference, so it is fine
            for blk in self.blocks_u:
                v = blk(v, 'v')
            v = self.norm_v(v)
            x = v.mean(dim=1)
            x = self.mlp_head(x)
            return x

        # used in case that the model is finetuned with both modality, but in inference only audio is given
        elif mode == 'missingaudioonly':
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            for blk in self.blocks_a:
                a = blk(a)

            # two forward passes to the block_u, one with modality-specific normalization, another with unified normalization
            u = a
            for blk in self.blocks_u:
                u = blk(u) # note here use unified normalization
            u = self.norm(u)
            u = u.mean(dim=1)

            for blk in self.blocks_u:
                a = blk(a, 'a') # note here use modality-specific normalization
            a = self.norm_a(a)
            a = a.mean(dim=1)

            # average the output of the two forward passes
            x = (u + a) / 2
            x = self.mlp_head(x)
            return x

        # used in case that the model is fine-tuned with both modality, but in inference only image is given
        elif mode == 'missingvideoonly':
            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            for blk in self.blocks_v:
                v = blk(v)

            # two forward passes to the block_u, one with modality-specific normalization, another with unified normalization
            u = v
            for blk in self.blocks_u:
                u = blk(u) # note here use unified normalization
            u = self.norm(u)
            u = u.mean(dim=1)

            for blk in self.blocks_u:
                v = blk(v, 'v') # note here use modality-specific normalization
            v = self.norm_v(v)
            v = v.mean(dim=1)

            # average the output of the two forward passes
            x = (u + v) / 2
            x = self.mlp_head(x)
            return x

    # for retrieval
    def forward_feat(self, a, v, mode='av'):
        # return both audio and visual
        if mode == 'av':
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            for blk in self.blocks_a:
                a = blk(a)

            for blk in self.blocks_v:
                v = blk(v)

            for blk in self.blocks_u:
                a = blk(a, 'a')
            a = self.norm_a(a)

            for blk in self.blocks_u:
                v = blk(v, 'v')

            v = self.norm_v(v)
            return a, v

        # return only audio
        if mode == 'a':
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            for blk in self.blocks_a:
                a = blk(a)

            for blk in self.blocks_u:
                a = blk(a, 'a')

            a = self.norm_a(a)
            return a