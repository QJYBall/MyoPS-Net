import torch
import torch.nn as nn
from network.unet import UNet, UNetEncoder, UNetDecoder, UNetDecoderPlus


### Segment
class CrossModalSegNet(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(CrossModalSegNet, self).__init__()

        self.unet_C0 = UNet(in_ch = in_chs[0], out_ch = out_chs[0])

        self.encoder_LGE = UNetEncoder(in_ch = in_chs[1])
        self.decoder_LGE = UNetDecoderPlus(out_ch = out_chs[1])

        self.encoder_T2 = UNetEncoder(in_ch = in_chs[2])
        self.decoder_T2 = UNetDecoderPlus(out_ch = out_chs[2])

        self.encoder_mapping = UNetEncoder(in_ch = in_chs[3])
        self.decoder_mapping = UNetDecoderPlus(out_ch = out_chs[3])

    def forward(self, C0, LGE, T2, T1m, T2starm):

        img = torch.cat([C0, LGE, T2, T1m, T2starm], dim=1)
        seg_C0 = self.unet_C0(img)
        mask_C0 = torch.argmax(seg_C0, dim=1, keepdim=True)

        img_LGE = torch.cat([LGE, mask_C0.detach()], dim=1)
        img_T2 = torch.cat([T2, mask_C0.detach()], dim=1)  
        img_mapping = torch.cat([T1m, T2starm, mask_C0.detach()], dim=1)

        f_LGE = self.encoder_LGE(img_LGE)
        f_T2 = self.encoder_T2(img_T2)
        f_mapping = self.encoder_mapping(img_mapping)

        seg_LGE_input = []
        seg_T2_input = []
        seg_mapping_input = []

        for i in range(5):
            seg_LGE_input.append(torch.max(f_mapping[i],f_T2[i]))
            seg_T2_input.append(torch.max(f_LGE[i],f_mapping[i]))
            seg_mapping_input.append(torch.max(f_LGE[i],f_T2[i]))

        seg_LGE = self.decoder_LGE(seg_LGE_input,f_LGE)
        seg_T2 = self.decoder_T2(seg_T2_input,f_T2)
        seg_mapping = self.decoder_mapping(seg_mapping_input,f_mapping)

        return seg_C0, seg_LGE, seg_T2, seg_mapping
        