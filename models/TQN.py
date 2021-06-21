# modified from https://github.com/tensorflow/models/blob/master/research/slim/nets/s3dg.py
import torch.nn as nn
import torch
import math
import numpy as np  
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from .transformer import *
from utils.utils import tfm_mask


class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0,LayerNorm=False):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, 
                              kernel_size=kernel_size, stride=stride, 
                              padding=padding, bias=False)

        self.LayerNorm = LayerNorm
        if not self.LayerNorm:
            self.bn = nn.BatchNorm3d(out_planes)
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()
        self.relu = nn.ReLU(inplace=True)
        self.conv.weight.data.fill_(1)

    def forward(self, x):
        x = self.conv(x)
        if not self.LayerNorm:
            x = self.bn(x)
        x = self.relu(x)
        return x


class STConv3d(nn.Module):
    def __init__(self,in_planes,out_planes,kernel_size,stride,padding=0):
        super(STConv3d, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=(1,kernel_size,kernel_size),
                              stride=(1,stride,stride),padding=(0,padding,padding), bias=False)
        self.conv2 = nn.Conv3d(out_planes,out_planes,kernel_size=(kernel_size,1,1),
                               stride=(stride,1,1),padding=(padding,0,0), bias=False)

        self.bn1=nn.BatchNorm3d(out_planes)
        self.bn2=nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        
        # init
        self.conv1.weight.data.normal_(mean=0, std=0.01)
        self.conv2.weight.data.normal_(mean=0, std=0.01)

        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.zero_()
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.zero_()
    
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu(x)
        return x


class SelfGating(nn.Module):
    def __init__(self, input_dim):
        super(SelfGating, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, input_tensor):
        """Feature gating as used in S3D-G"""
        spatiotemporal_average = torch.mean(input_tensor, dim=[2, 3, 4])
        weights = self.fc(spatiotemporal_average)
        weights = torch.sigmoid(weights)
        return weights[:, :, None, None, None] * input_tensor


class SepInception(nn.Module):
    def __init__(self, in_planes, out_planes, gating=False,LayerNorm=False):
        super(SepInception, self).__init__()

        assert len(out_planes) == 6
        assert isinstance(out_planes, list)

        [num_out_0_0a, 
        num_out_1_0a, num_out_1_0b,
        num_out_2_0a, num_out_2_0b, 
        num_out_3_0b] = out_planes

        self.branch0 = nn.Sequential(
            BasicConv3d(in_planes, num_out_0_0a, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(in_planes, num_out_1_0a, kernel_size=1, stride=1),
            STConv3d(num_out_1_0a, num_out_1_0b, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(in_planes, num_out_2_0a, kernel_size=1, stride=1),
            STConv3d(num_out_2_0a, num_out_2_0b, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(in_planes, num_out_3_0b, kernel_size=1, stride=1,LayerNorm=LayerNorm),
        )

        self.out_channels = sum([num_out_0_0a, num_out_1_0b, num_out_2_0b, num_out_3_0b])

        self.gating = gating 
        if gating:
            self.gating_b0 = SelfGating(num_out_0_0a)
            self.gating_b1 = SelfGating(num_out_1_0b)
            self.gating_b2 = SelfGating(num_out_2_0b)
            self.gating_b3 = SelfGating(num_out_3_0b)

    def forward(self, x):
        if isinstance(x,tuple):
            x = x[0]

        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        if self.gating:
            x0 = self.gating_b0(x0)
            x1 = self.gating_b1(x1)
            x2 = self.gating_b2(x2)
            x3 = self.gating_b3(x3)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out




class TQN(nn.Module):

    def __init__(self, args,first_channel=3,features_out =False,gating=False,SUFB=False,mode='train'):
        super(TQN, self).__init__()

        self.gating = gating
        self.features_out = features_out
        self.d_model = args.d_model
        self.SUFB = SUFB
        self.mode = mode

        if SUFB:
            self.K =args.K

        ###################################
        '''S3D'''
        ###################################

        self.Conv_1a = STConv3d(first_channel, 64, kernel_size=7, stride=2, padding=3) 
        self.block1 = nn.Sequential(self.Conv_1a) # (64, 32, 112, 112)

        self.MaxPool_2a = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)) 
        self.Conv_2b = BasicConv3d(64, 64, kernel_size=1, stride=1) 
        self.Conv_2c = STConv3d(64, 192, kernel_size=3, stride=1, padding=1) 

        self.block2 = nn.Sequential(
            self.MaxPool_2a, # (64, 32, 56, 56)
            self.Conv_2b,    # (64, 32, 56, 56)
            self.Conv_2c)    # (192, 32, 56, 56)

        
        self.MaxPool_3a = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)) 
        self.Mixed_3b = SepInception(in_planes=192, out_planes=[64, 96, 128, 16, 32, 32], gating=gating)
        self.Mixed_3c = SepInception(in_planes=256, out_planes=[128, 128, 192, 32, 96, 64], gating=gating)

        self.block3 = nn.Sequential(
            self.MaxPool_3a,    # (192, 32 , 28, 28)
            self.Mixed_3b,      # (256, 32, 28, 28)
            self.Mixed_3c)      # (480, 32, 28, 28)

        self.MaxPool_4a = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.Mixed_4b = SepInception(in_planes=480, out_planes=[192, 96, 208, 16, 48, 64], gating=gating)
        self.Mixed_4c = SepInception(in_planes=512, out_planes=[160, 112, 224, 24, 64, 64], gating=gating)
        self.Mixed_4d = SepInception(in_planes=512, out_planes=[128, 128, 256, 24, 64, 64], gating=gating)
        self.Mixed_4e = SepInception(in_planes=512, out_planes=[112, 144, 288, 32, 64, 64], gating=gating)
        self.Mixed_4f = SepInception(in_planes=528, out_planes=[256, 160, 320, 32, 128, 128], gating=gating)

        self.block4 = nn.Sequential(
            self.MaxPool_4a,  # (480, 16, 14, 14)
            self.Mixed_4b,    # (512, 16, 14, 14)
            self.Mixed_4c,    # (512, 16, 14, 14)
            self.Mixed_4d,    # (512, 16, 14, 14)
            self.Mixed_4e,    # (528, 16, 14, 14)
            self.Mixed_4f)    # (832, 16, 14, 14)

        self.MaxPool_5a = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))
        self.Mixed_5b = SepInception(in_planes=832, out_planes=[256, 160, 320, 32, 128, 128], gating=gating)
        self.Mixed_5c = SepInception(in_planes=832, out_planes=[384, 192, 384, 48, 128, 128], gating=gating)

        self.block5 = nn.Sequential(
            self.MaxPool_5a,  # (832, 8, 7, 7)
            self.Mixed_5b,    # (832, 8, 7, 7)
            self.Mixed_5c)    # (1024, 8, 7, 7)

        self.AvgPool_0a = nn.AvgPool3d(kernel_size=(1, 7, 7), stride=1)



        ###################################
        ''' Query Decoder'''
        ###################################

        if not self.features_out:

            # Decoder Layers
            self.H = args.H 
            decoder_layer = TransformerDecoderLayer(self.d_model, args.H, 1024,
                                        0.1, 'relu',normalize_before=True)
            decoder_norm = nn.LayerNorm(self.d_model)
            self.decoder = TransformerDecoder(decoder_layer, args.N, decoder_norm,
                                  return_intermediate=False)

            # Learnable Queries
            self.query_embed = nn.Embedding(args.num_queries,self.d_model)
            self.dropout_feas = nn.Dropout(args.dropout)

            # Attribute classifier
            self.classifier = nn.Linear(self.d_model,args.attribute_set_size)

            # Class classifier
            self.cls_classifier = nn.Linear(self.d_model,args.num_classes)


        self.apply(self._init_weights)




    def forward(self, input):

        ''' Reshape Input Sequences '''
        if not self.SUFB:
            x, mask = input
            if len(x.shape) ==5:
                # the First stage training
                BK, C, T, H, W =x.shape 
                seg_per_video = mask.shape[-1] - mask.sum(1)

            else:
                # Feature extraction mode for full video sequence
                B, K, C, T, H, W = x.shape 
                x = x.reshape(B*K,C,T,H,W)
                seg_per_video = None

        else:
            # Training with a Stochastically Updated Feature Bank
            x, vids, ptrs = input
            B, K, C, T, H, W = x.shape 
            x = x.reshape(B*K,C,T,H,W)
            seg_per_video = None


        ''' Visual Backbone '''
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        features = self.AvgPool_0a(x).squeeze()

        if self.SUFB:
            features,Ts,mask = self.fill_SUFB(features,vids,ptrs) 

        if self.features_out: 
            return features
            
        else: 
            ''' Query Decoder '''
            if seg_per_video is not None:
                # first stage training
                features = self.reshape_features(features.squeeze(), 
                    seg_per_video)
                B = len(seg_per_video)
                K = int(BK // B)

            elif not self.SUFB:
                features = features.reshape(B,K,-1)

            if mask is not None:
                mask = mask.view(B,-1)

            features = features.transpose(0,1)
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)
            features = self.decoder(query_embed, features, 
                memory_key_padding_mask=mask, pos=None, query_pos=None)

            out = self.dropout_feas(features) # [T,B,C]
            x= self.classifier(out[:-1]).transpose(0,1)
            x_cls = self.cls_classifier(out[-1])

            return x, x_cls


    def reshape_features(self,features,seg_per_video):
        reshaped_features = []
        counter = 0
        for n_seg in seg_per_video:
            reshaped_features.append(features[counter:counter+n_seg])
            counter += n_seg
        return pad_sequence(reshaped_features,batch_first=True)
        

    def fill_SUFB(self,features,vids,ptrs):
        fea_dim = features.shape[-1]

        if self.mode =='train':
            # Update newly computed features in the SUFB,
            # And read all the features from the SUFB 
            full_features = []
            features = features.view(-1,self.K,fea_dim)
            features_split = torch.split(features, 1, dim=0)

            for f, vid, ptr in zip(features_split, vids, ptrs):
                vid = vid.item()
                end = min([len(self.queue[vid]), ptr + self.K])
                
                self.queue[vid][ptr:end] = f[0,:(end-ptr),:]
                full_features.append(self.queue[vid])
                self.queue[vid] = self.queue[vid].detach()


            Ts = [f.shape[0] for f in full_features]
            mask = tfm_mask(Ts).cuda()
            features = pad_sequence(full_features,batch_first=True).cuda()


        elif self.mode == 'test':
            # Test mode, compute all features online
            features = features.view(B,-1,fea_dim).cuda()
            Ts = [features[i].shape[0] for i in range(B)]
            mask = tfm_mask(Ts).cuda()

        return features,Ts,mask


    @staticmethod
    def _init_weights(module):
        r"""Initialize weights like BERT - N(0.0, 0.02), bias = 0."""

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


