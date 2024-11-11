import typing as tp

import numpy as np
import torch.nn as nn
from squeezeformer_blocks import *
import qt
from timm.layers.norm_act import BatchNormAct2d

n_coords = 2
class FeatureExtractor(nn.Module):
    def __init__(self,
                 n_landmarks,
                 out_dim):
        super().__init__()
        self.in_channels = in_channels = (32 // 2) * n_landmarks
        self.stem_linear = nn.Linear(in_channels, out_dim, bias=False) # Linear(in_features=2080, out_features=128, bias=False)
        self.stem_bn = nn.BatchNorm1d(out_dim, momentum=0.95)

        self.conv_stem = nn.Conv2d(n_coords, 32, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), bias=False)
        self.bn_conv = BatchNormAct2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,
                                      act_layer=nn.SiLU, drop_layer=None)

    def forward(self, data, mask):
        xc = data.permute(0, 3, 1, 2) # torch.Size([64, 3, 384, 130])
        xc = self.conv_stem(xc) # torch.Size([64, 32, 384, 65])
        xc = self.bn_conv(xc) # torch.Size([64, 32, 384, 65])
        xc = xc.permute(0, 2, 3, 1) # torch.Size([64, 384, 65, 32])
        xc = xc.reshape(*data.shape[:2], -1) # torch.Size([64, 384, 2080])
        x = self.stem_linear(xc) # torch.Size([64, 384, 128])

        # Batchnorm without pads
        bs, slen, nfeat = x.shape # 64, 384, 128
        x = x.view(-1, nfeat) # torch.Size([24576, 128])
        x_bn = x[mask.view(-1) == 1].unsqueeze(0) # torch.Size([1, 10290, 128])
        x_bn = self.stem_bn(x_bn.permute(0, 2, 1)).permute(0, 2, 1) # torch.Size([1, 10290, 128])
        x[mask.view(-1) == 1] = x_bn[0] # 0 since we did unsqueeze(0)

        x = x.view(bs, slen, nfeat) # torch.Size([64, 384, 128])
        
        # Padding mask
        x = x.masked_fill(~mask.bool().unsqueeze(-1), 0.0) # torch.Size([64, 384, 128])

        return x

N_HIDDEN = 128
class FeatureExtractorTransposed(nn.Module):
    def __init__(self,
                 n_landmarks,
                 out_dim):
        super().__init__()

        self.out_channels = in_channels = (32 // 2) * n_landmarks
        self.stem_linear1 = nn.Linear(N_HIDDEN, (n_landmarks//2)*32, bias=False)

        # self.stem_linear2 = nn.Linear(512, 1024, bias=False)
        # self.stem_linear3 = nn.Linear(512, 2080, bias=False)

        self.stem_bn = nn.BatchNorm1d(out_dim, momentum=0.95)
        self.conv_stem = nn.ConvTranspose2d(32, n_coords, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1),
                                                        output_padding=(0, 1), bias=False)
        # WHY OUTPUT PADDING IS HERE? - because when the stride > 1 we need to pad output this is what documentation says

        self.bn_conv = BatchNormAct2d(n_coords, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,
                                      act_layer=nn.SiLU, drop_layer=None)
        self.do_batch_norm = True
        self.final_linear = nn.Linear(n_landmarks*n_coords, n_landmarks*n_coords, bias=False)

    def forward(self, data, mask):
        bs, slen, nfeat = data.shape
        # assert slen==384
        # assert nfeat == 128

        x = data
        
        # this batchnorm is only on non-masked regions
        if self.do_batch_norm:
            x = x.view(-1, nfeat)
            x_bn = x[mask.view(-1) == 1].unsqueeze(0)
            x_bn = self.stem_bn(x_bn.permute(0, 2, 1)).permute(0, 2, 1)
            x[mask.view(-1) == 1] = x_bn[0] # only batch norm over the
            x = x.view(bs, slen, nfeat)
            x = x.masked_fill(~mask.bool().unsqueeze(-1), 0.0)

        # from 128 for each frame to 2080
        # for padded frames it doesn't make a difference because those are all 0s and there is no bias
        x = self.stem_linear1(x)

        # from torch.Size([64, 384, 2080]) to [64, 384, -1, 32] where -1 must be 65 I think
        x = x.reshape(bs, slen, -1, 32) # 32 here is number of channels
        # HERE POTENTIALLY SHOULD BE BATCH NORM
        x = x.permute(0, 3, 1, 2)
        # torch.Size([64, 32, 384, 65])

        x = self.conv_stem(x)
        x = self.bn_conv(x)
        # TODO: HERE IT IS EXTREMELY LIKELY THAT THIS AFFECTS THE RESULTS FROM THE DECODER
        # CHECK THE SAME WAY I CHECKED WITHIN THE ENCODER
        # torch.Size([64, 3, 384, 130]) doubles the number from 65 to 130 because of the stride and transpose AND
        # from 32 to 3 because of the filters in conv

        x = x.permute(0, 2, 3, 1)
        # torch.Size([32, 384, 130, 3])
        #### final fully connected layer on those 54*3 ######
        x = x.reshape(*data.shape[:2], -1)
        x = self.final_linear(x)
        x = x.reshape(*data.shape[:2], -1, n_coords)
        
        return x

class MyEncoder(nn.Module):
    def __init__(
        self,
        channels: int = 80,
        dimension: int = 144,
        num_layers: int = 6,
        num_attention_heads: int = 4,
        feed_forward_expansion_factor: int = 1,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.1,
        feed_forward_dropout_p: float = 0.0,
        attention_dropout_p: float = 0.0,
        conv_dropout_p: float = 0.0,
        conv_kernel_size: int = 51,
        n_filters=32,
        n_residual_layers=3,
        ratios=[8,5,4,2],
    ):
        
        # TODOs: change ratios to something more real - check get_gesture_compression_model
        # and remove it from there
        super(MyEncoder, self).__init__()
        self.num_layers = num_layers
        self.recover_tensor = None
        self.dimension = dimension
        self.channels = channels
        self.blocks = nn.ModuleList()
        self.hop_length = np.prod(ratios)
        self.feature_extractor = FeatureExtractor(n_landmarks=channels, out_dim=self.dimension)
        # self.lstm = StreamableLSTM(self.dimension, num_layers=1)

        for idx in range(num_layers):
            self.blocks.append(
                SqueezeformerBlock(
                    encoder_dim=self.dimension,
                    num_attention_heads=num_attention_heads,
                    feed_forward_expansion_factor=feed_forward_expansion_factor,
                    conv_expansion_factor=conv_expansion_factor,
                    feed_forward_dropout_p=feed_forward_dropout_p,
                    attention_dropout_p=attention_dropout_p,
                    conv_dropout_p=conv_dropout_p,
                    conv_kernel_size=conv_kernel_size,
                )
            )

        # this one has downsampling with the stride!
        # self.blocks.append(
        #     SqueezeformerBlockWithDownsampling(
        #         encoder_dim=self.dimension,
        #         num_attention_heads=num_attention_heads,
        #         feed_forward_expansion_factor=feed_forward_expansion_factor,
        #         conv_expansion_factor=conv_expansion_factor,
        #         feed_forward_dropout_p=feed_forward_dropout_p,
        #         attention_dropout_p=attention_dropout_p,
        #         conv_dropout_p=conv_dropout_p,
        #         conv_kernel_size=conv_kernel_size,
        #     )
        # )

    def forward(self, batch):
        """
        Forward propagate a `inputs` for  encoder training.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            (Tensor, Tensor)
            * outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * output_lengths (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        x = batch['input']  # bs, seq_len, n_landmarks, 3
        mask = batch['input_mask'].long()
        x = self.feature_extractor(x, mask)

        for idx, block in enumerate(self.blocks):
            x = block(x, mask)

        feats = x

        return x, feats

class MyDecoder(nn.Module):
    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        num_layers: int = 6,
        num_attention_heads: int = 4,
        feed_forward_expansion_factor: int = 1,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.1,
        feed_forward_dropout_p: float = 0.0,
        attention_dropout_p: float = 0.0,
        conv_dropout_p: float = 0.0,
        conv_kernel_size: int = 51,
        n_filters=32,
        n_residual_layers=3,
        ratios=[8,5,4,2],
        activation='ELU',
        activation_params={'alpha': 1.0},
        norm='none',
        norm_params={},
        kernel_size=7,
        last_kernel_size=7,
        residual_kernel_size=3,
        dilation_base=2,
        causal=False,
        pad_mode='reflect',
        true_skip=True,
        compress=2,
        lstm=0,
        disable_norm_outer_blocks=0,
        trim_right_ratio=1,
        final_activation=None,
        final_activation_params=None,
    ):
        # TODOs: change ratios to something more real - check get_gesture_compression_model
        # and remove it from there
        super(MyDecoder, self).__init__()
        self.num_layers = num_layers
        self.recover_tensor = None
        self.dimension = dimension
        self.channels = channels
        self.blocks = nn.ModuleList()
        self.hop_length = np.prod(ratios)
        # self.transposed_feature_extractor = TransposedFeatureExtractor(n_landmarks=130, out_dim=self.dimension)
        # self.lstm = StreamableLSTM(self.dimension, num_layers=1)

        # self.upsampling_block = SqueezeformerBlockWithUpsampling(
        #         encoder_dim=self.dimension,
        #         num_attention_heads=num_attention_heads,
        #         feed_forward_expansion_factor=feed_forward_expansion_factor,
        #         conv_expansion_factor=conv_expansion_factor,
        #         feed_forward_dropout_p=feed_forward_dropout_p,
        #         attention_dropout_p=attention_dropout_p,
        #         conv_dropout_p=conv_dropout_p,
        #         conv_kernel_size=conv_kernel_size,
        #     )

        for idx in range(num_layers):
            self.blocks.append(
                SqueezeformerBlock(
                    encoder_dim=self.dimension,
                    num_attention_heads=num_attention_heads,
                    feed_forward_expansion_factor=feed_forward_expansion_factor,
                    conv_expansion_factor=conv_expansion_factor,
                    feed_forward_dropout_p=feed_forward_dropout_p,
                    attention_dropout_p=attention_dropout_p,
                    conv_dropout_p=conv_dropout_p,
                    conv_kernel_size=conv_kernel_size,
                )
            )

        # TODO: most likely needs a mask!!!
        self.transposed_feature_extractor = FeatureExtractorTransposed(n_landmarks=channels, out_dim=self.dimension)

    def forward(self, x, mask):
        """
        Forward propagate a `inputs` for  encoder training.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            (Tensor, Tensor)
            * outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * output_lengths (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        for idx, block in enumerate(self.blocks):
            x = block(x, mask)

        # x has bs, seq_len, 128
        x = self.transposed_feature_extractor(x, mask)

        ########### END OF ORIGINAL FORWARD   ###############
        return x #.transpose(1,2)



class EncodecModel(nn.Module):
    """Encodec model operating on the raw waveform.

    Args:
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        quantizer (qt.BaseQuantizer): Quantizer network.
        frame_rate (int): Frame rate for the latent representation.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        causal (bool): Whether to use a causal version of the model.
        renormalize (bool): Whether to renormalize the audio before running the model.
    """

    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 quantizer,
                 frame_rate: int,
                 sample_rate: int,
                 channels: int,
                 causal: bool = False,
                 renormalize: bool = False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.channels = channels
        self.renormalize = renormalize
        self.causal = causal
        self.lm_head = nn.Linear(128, 63, bias=False)

        if self.causal:
            # we force disabling here to avoid handling linear overlap of segments
            # as supported in original EnCodec codebase.
            assert not self.renormalize, 'Causal model does not support renormalize'

    @property
    def total_codebooks(self):
        """Total number of quantizer codebooks available."""
        return self.quantizer.total_codebooks

    @property
    def num_codebooks(self):
        """Active number of codebooks used by the quantizer."""
        return self.quantizer.num_codebooks

    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        self.quantizer.set_num_codebooks(n)

    @property
    def cardinality(self):
        """Cardinality of each codebook."""
        return self.quantizer.bins

    def forward(self, x: torch.Tensor) -> qt.QuantizedResult:
        # assert x.dim() == 4
        length = x['input'].shape[-1]
        mask = x['input_mask'].long()
        # x, scale = self.preprocess(x['input'])
        # assert torch.equal(save_x, x) # THIS ASSERT HOLDS - SO I CAN SKIP IT
        emb, feats = self.encoder(x)
        # emb = emb.masked_fill(~x['input_mask'].bool().unsqueeze(-1), 0.0)

        # emb[x['input_mask']] = 0
        q_res = self.quantizer(emb, x['input_mask'], self.frame_rate)
        # qres have many outputs - QuantizedResult(quantized, codes, bw, penalty=torch.mean(commit_loss))
        # get logit predictions from those dense vectors

        # I can just make context window very large?????? - leave like this for now but TODO
        logits = self.lm_head(q_res.x).permute(1, 0, 2) # maybe dimensions here are not correct
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)

        # I guess for the backprop
        # x_divided_by_4_mask = x['input_mask'].long()

        out = self.decoder(q_res.x, mask)

        # remove extra padding added by the encoder and decoder
        # assert out.shape[-1] >= length, (out.shape[-1], length)
        # we always know the length!
        # out = out[..., :length]

        q_res.x = out # no postptocessing since no preprocessing
        # q_res.x = self.postprocess(out, scale)

        return q_res, log_probs

    def encode(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        """Encode the given input tensor to quantized representation along with scale parameter.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T]

        Returns:
            codes, scale (tuple of torch.Tensor, torch.Tensor): Tuple composed of:
                codes a float tensor of shape [B, K, T] with K the number of codebooks used and T the timestep.
                scale a float tensor containing the scale for audio renormalizealization.
        """
        # assert x.dim() == 4 IT IS 4 FOR GESTURES
        q_res, feats = self.encoder(x)
        feats = feats.masked_fill(~x['input_mask'].bool().unsqueeze(-1), 0.0)
        # codes = self.quantizer.encode(emb)

        return feats

    def encode_to_discrete(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        """Encode the given input tensor to quantized representation along with scale parameter.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T]

        Returns:
            codes, scale (tuple of torch.Tensor, torch.Tensor): Tuple composed of:
                codes a float tensor of shape [B, K, T] with K the number of codebooks used and T the timestep.
                scale a float tensor containing the scale for audio renormalizealization.
        """

        emb, feats = self.encoder(x) # identical as in forward
        feats = feats.masked_fill(~x['input_mask'].bool().unsqueeze(-1), 0.0)
        codes = self.quantizer.encode(emb)

        return codes

    # def encode(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
    #     """Encode the given input tensor to quantized representation along with scale parameter.
    #
    #     Args:
    #         x (torch.Tensor): Float tensor of shape [B, C, T]
    #
    #     Returns:
    #         codes, scale (tuple of torch.Tensor, torch.Tensor): Tuple composed of:
    #             codes a float tensor of shape [B, K, T] with K the number of codebooks used and T the timestep.
    #             scale a float tensor containing the scale for audio renormalizealization.
    #     """
    #     # assert x.dim() == 4 IT IS 4 FOR GESTURES
    #     x, scale = self.preprocess(x)
    #     emb = self.encoder(x)
    #     codes = self.quantizer.encode(emb)
    #     return codes, scale

    def decode(self, codes: torch.Tensor, scale: tp.Optional[torch.Tensor] = None):
        """Decode the given codes to a reconstructed representation, using the scale to perform
        audio denormalization if needed.

        Args:
            codes (torch.Tensor): Int tensor of shape [B, K, T]
            scale (torch.Tensor, optional): Float tensor containing the scale value.

        Returns:
            out (torch.Tensor): Float tensor of shape [B, C, T], the reconstructed audio.
        """
        emb = self.decode_latent(codes)
        out = self.decoder(emb)
        # out = self.postprocess(out, scale)
        # out contains extra padding added by the encoder and decoder
        return out

    def decode_latent(self, codes: torch.Tensor):
        """Decode from the discrete codes to continuous latent space."""
        return self.quantizer.decode(codes)