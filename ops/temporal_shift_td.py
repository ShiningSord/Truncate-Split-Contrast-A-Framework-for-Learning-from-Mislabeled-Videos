# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pdb
import numpy as np

class TemporalDiff(nn.Module):
    def __init__(self, in_channels, out_channels, n_segment, kernel_size, stride=1, padding=0, bias=True, n_div=8):
        super(TemporalDiff, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_segment = n_segment
        self.kernel_size = kernel_size
        self.stride=stride
        self.padding=padding
        self.bias = bias
        self.n_div = n_div
        print('=> Using fold div: {}'.format(self.n_div))
        print('model equipped with temporal difference attention...')
        self.conv1_reduce = nn.Sequential(
                nn.Conv2d(in_channels, in_channels//n_div, kernel_size=1, stride=1, padding=padding, bias=bias),
                nn.BatchNorm2d(in_channels//n_div),
                nn.ReLU(inplace=True))
        self.conv2_reduce = nn.Sequential(
                nn.Conv2d(in_channels, in_channels//n_div, kernel_size=1, stride=1, padding=padding, bias=bias),
                nn.BatchNorm2d(in_channels//n_div),
                nn.ReLU(inplace=True))


        self.conv_inflate = nn.Sequential(
                nn.Conv2d(in_channels//n_div, in_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))


    def forward(self, x):
        # out = F.conv2d(x, self.weight, None, self.stride, 0, 1, self.channels)
        # out = F.pad(out, pad= [0, 0, 0, 1])
        # x.size = N*C*T*(H*W)


        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment


        out = F.adaptive_avg_pool2d(x, (1, 1))
        rc = self.in_channels // self.n_div
        # print(out.size())
        # exit()
        left = self.conv1_reduce(out).view(n_batch, self.n_segment, rc, 1, 1)
        right = self.conv2_reduce(out).view(n_batch, self.n_segment, rc, 1, 1)
        out = left[:, :-1] - right[:, 1:]
        out = out.view(n_batch*(self.n_segment-1), rc, 1, 1)
        out = self.conv_inflate(out)


        #out = torch.sqrt(out*out)
        out = torch.sigmoid(out)
        out = out.view(n_batch, self.n_segment-1, c)
        out = F.pad(out, pad=[0, 0, 0, 1], mode='constant', value=1)
        out = out.view(nt, c, 1, 1)

        return out*x

class ChannelGate(nn.Module):
    def __init__(self, channels):
        super(ChannelGate, self).__init__()
        self.linearProj = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        # pdb.set_trace()
        self.channels = channels
        start_idx = int((channels-1)*0.9)
        self.t_linear = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        # self.linearProj.weight.data.zero_()
        # self.linearProj.bias.data.zero_()
        # with torch.no_grad():
        #     self.linearProj.weight[start_idx,:,:,:] += 1.0

    def forward(self, x):

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.linearProj(x)
        t = self.t_linear(x)
        x = self.bn(x)
        x_in = self.relu(x)
        x_out = F.softmax(x_in/(self.sigmoid(t)+0.0001), dim=1)
        # pdb.set_trace()
        out = torch.cumsum(x_out, dim=1)
        # pdb.set_trace()
        # pdb.set_trace()
        # out = torch.zeros(x.shape).cuda()
        # start_idx = int(self.channels*0.3)
        # out[:,start_idx:,:,:] = 1
        # pdb.set_trace()

        return out

class AccumAtt(nn.Module):
    def __init__(self, channels, n_segment):

        super(AccumAtt, self).__init__()

        self.channels = channels
        self.n_segment = n_segment

        self.conv1_reduce = nn.Sequential(
                nn.Conv2d(channels, channels//8, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(channels//8),
                nn.ReLU(inplace=True))
        self.conv2_reduce = nn.Sequential(
                nn.Conv2d(channels, channels//8, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(channels//8),
                nn.ReLU(inplace=True))

        self.Win = nn.Conv2d(channels//8, channels//8, kernel_size=(1,1), bias=True)
        self.Wg  = nn.Conv2d(channels//8, channels//8, kernel_size=(1,1), bias=True)
        self.Wa  = nn.Conv2d(channels//8, channels, kernel_size=(1,1), bias=True)
        self.gamma = nn.Conv2d(channels//4, 1, kernel_size=(1,1), bias=True)
        self.bn = nn.BatchNorm2d(channels//8)
        self.relu = nn.ReLU(inplace=True)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.Win.weight)
        nn.init.xavier_uniform_(self.Wg.weight)
        nn.init.xavier_uniform_(self.Wa.weight)
        nn.init.xavier_uniform_(self.gamma.weight)
        self.Win.bias.data.zero_()
        self.Wg.bias.data.zero_()
        self.Wa.bias.data.zero_()
        self.gamma.bias.data.zero_()

    def forward(self, x):

        nt, c, w, h = x.size()
        n_batch = nt // self.n_segment

        x_vec = F.adaptive_avg_pool2d(x, (1,1))
        x = x.view(n_batch, self.n_segment, c, w, h)

        x_vec_left = self.conv1_reduce(x_vec).view(n_batch, self.n_segment, c//8, 1, 1)
        x_vec_right = self.conv2_reduce(x_vec).view(n_batch, self.n_segment, c//8, 1, 1)
        x_vec_diff = x_vec_left[:, :-1] - x_vec_right[:, 1:]
        x_vec_diff = F.pad(x_vec_diff, pad=[0, 0, 0, 0, 0, 0, 0, 1], mode='constant', value=1)

        x_global = x_vec_diff[:,-1] # torch.zeros((n_batch,c//8,1,1)).cuda()
        atts = []
        for t_idx in range(self.n_segment):
            # x_global = self.Win(x_vec_diff[:, t_idx, :, :, :].squeeze(1)) + self.Wg(x_global.clone())
            x_cat = torch.cat((x_vec_diff[:, t_idx, :, :, :].squeeze(1),x_global),dim=1)
            x_gamma = self.sigmoid(self.gamma(x_cat).repeat(1,c//8,1,1))
            x_global = x_vec_diff[:, t_idx, :, :, :].squeeze(1)*x_gamma + x_global.clone()*(1-x_gamma)
            x_att = self.sigmoid(self.Wa(x_global)).unsqueeze(1)
            atts.append(x_att)
        sigmas = torch.cat(atts, 1)
        out = x*sigmas

        # x_gather = []
       #  x_accum = torch.zeros((n_batch,1,c,w,h)).cuda()

        # for t_idx in range(self.n_segment):
        #     gamma = self.sigmoid(self.gamma(sigmas[:,t_idx,:,:,:]).unsqueeze(4).repeat(1,1,c,w,h))
        #     x_accum = (1-gamma)*x_accum.clone() + gamma*x_atts[:, t_idx, :, :, :].unsqueeze(1)
        #     x_gather.append(x_accum)
        # out = torch.cat(x_gather, 1).view(-1,c,w,h)

        return out.view(-1,c,w,h)

class TemporalDiffChannel(nn.Module):
    def __init__(self, in_channels, out_channels, n_segment, kernel_size, stride=1, padding=0, bias=True, n_div=8):
        super(TemporalDiffChannel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_segment = n_segment
        self.kernel_size = kernel_size
        self.stride=stride
        self.padding=padding
        self.bias = bias
        self.n_div = n_div


        self.diff_trans = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=bias),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True))

        # identity_weight = torch.empty(self.diff_trans[0].weight.shape)
        # nn.init.dirac_(identity_weight)
        # self.diff_trans[0].weight.data.copy_(identity_weight)
        # self.diff_trans[0].weight.data.zero_()
        # self.diff_trans[0].bias.data.zero_()
        nn.init.xavier_uniform_(self.diff_trans[0].weight)
        self.diff_trans[1].weight.data.fill_(1)
        self.diff_trans[1].bias.data.zero_()

        self.ori_trans = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=bias),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True))

        identity_weight = torch.empty(self.ori_trans[0].weight.shape)
        nn.init.dirac_(identity_weight)
        self.ori_trans[0].weight.data.copy_(identity_weight)
        # self.ori_trans[0].bias.data.zero_()
        self.ori_trans[1].weight.data.fill_(1)
        self.ori_trans[1].bias.data.zero_()

        self.channelGate = ChannelGate(in_channels)


        # self.shift_conv = TemporalShift(in_channels, (3,1), padding=(1,0), n_div=8, bias=False)

        # if 1 - 0.0 < 1e-5:
        #     self.shift_conv.weight.requires_grad = False
        # else:
        #     self.shift_conv.weight.requires_grad = True


    def forward(self, x):
        # out = F.conv2d(x, self.weight, None, self.stride, 0, 1, self.channels)
        # out = F.pad(out, pad= [0, 0, 0, 1])
        # x.size = N*C*T*(H*W)


        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment

        x_ori = self.ori_trans(x)


        x = x.view(n_batch, self.n_segment, c, h, w)
        x_diff = x[:, :-1] - x[:, 1:]
        x_diff = F.pad(x_diff, pad=[0, 0, 0, 0, 0, 0, 0, 1], mode='constant', value=1).view(n_batch*self.n_segment, c, h, w)
        x_diff = self.diff_trans(x_diff)

        # reshape_x = x.view(n_batch, -1, c, h*w).permute(0, 2, 1, 3).contiguous()
        # shift_x = self.shift_conv(reshape_x)
        # shift_x = shift_x.permute(0,2,1,3).contiguous().view(nt, c, h, w)
        # x_shift = self.diff_trans(shift_x)

        channel_w = self.channelGate(x_diff)


        return channel_w * x_diff + (1-channel_w) * x_ori


class TemporalShift(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, bias=True, n_div=8, p_init_type='tsm'):
        super(TemporalShift, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride=stride
        self.padding=padding
        self.bias = bias
        self.fold_div = n_div
        print('=> Using fold div: {}'.format(self.fold_div))
        print('model equipped with shift conv...')
        # self.shift_conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups, padding=padding, bias=bias)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        conv_params = torch.zeros((in_channels, 1)+kernel_size)
        # future_params = torch.zeros((in_channels, 1)+kernel_size)
        # print('params = ', data.size())
        fold = in_channels // n_div

        # TSM initialization
        if p_init_type == 'r_tsm':
            for i in range(in_channels):
                import random
                j = random.randint(0, kernel_size[0]-1)
                conv_params[i, :, j] = 1
            self.weight = nn.Parameter(conv_params)
        elif p_init_type == 'tsm':
            conv_params[:fold, :, kernel_size[0]//2+1] = 1
            conv_params[fold:2*fold, :, kernel_size[0]//2-1] = 1
            conv_params[2*fold: , :, kernel_size[0]//2] = 1
            self.weight = nn.Parameter(conv_params)
        elif p_init_type == 'TSN':
            conv_params[:, :, kernel_size[0]//2] = 1
            self.weight = nn.Parameter(conv_params)
        else:
            init.kaiming_uniform_(self.weight, a=math.sqrt(4))




    def forward(self, x):
        return F.conv2d(x, self.weight, None, self.stride, self.padding, 1, self.in_channels)

class TemporalShiftBilinearLocal(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, bias=True, n_div=8, p_init_type='tsm'):
        super(TemporalShiftBilinearLocal, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride=stride
        self.padding=padding
        self.bias = bias
        self.fold_div = n_div
        print('=> Using fold div: {}'.format(self.fold_div))
        print('model equipped with shift conv...')
        # self.shift_conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups, padding=padding, bias=bias)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        conv_params1 = torch.zeros((in_channels, 1)+kernel_size)
        conv_params2 = torch.zeros((in_channels, 1)+kernel_size)

        # future_params = torch.zeros((in_channels, 1)+kernel_size)
        # print('params = ', data.size())
        fold = in_channels // n_div

        # TSM initialization
        if p_init_type == 'r_tsm':
            for i in range(in_channels):
                import random
                j = random.randint(0, kernel_size[0]-1)
                conv_params1[i, :, j] = 1
            self.weight1 = nn.Parameter(conv_params1)
        elif p_init_type == 'tsm':
            conv_params1[:fold, :, kernel_size[0]//2+1] = 1
            conv_params1[fold:2*fold, :, kernel_size[0]//2-1] = 1
            conv_params1[2*fold: , :, kernel_size[0]//2] = 1
            self.weight1 = nn.Parameter(conv_params1)
        elif p_init_type == 'TSN':
            conv_params1[:, :, kernel_size[0]//2] = 1
            self.weight1 = nn.Parameter(conv_params1)
        else:
            init.kaiming_uniform_(self.weight1, a=math.sqrt(4))


        # TSM initialization
        if p_init_type == 'r_tsm':
            for i in range(in_channels):
                import random
                j = random.randint(0, kernel_size[0]-1)
                conv_params2[i, :, j] = 1
            self.weight2 = nn.Parameter(conv_params2)
        elif p_init_type == 'tsm':
            conv_params2[:fold, :, kernel_size[0]//2+1] = 1
            conv_params2[fold:2*fold, :, kernel_size[0]//2-1] = 1
            conv_params2[2*fold: , :, kernel_size[0]//2] = 1
            self.weight2 = nn.Parameter(conv_params2)
        elif p_init_type == 'TSN':
            conv_params2[:, :, kernel_size[0]//2] = 1
            self.weight2 = nn.Parameter(conv_params2)
        else:
            init.kaiming_uniform_(self.weight2, a=math.sqrt(4))

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # pdb.set_trace()
        # first-order bilinear
        # n, c, t, s = x.size()
        # x_div1 = x[:,:c//4,:,:]
        # x_div2 = x[:,c//4:,:,:]
        # identity = x_div1
        # # pdb.set_trace()
        # shift = F.conv2d(x_div1, self.weight, None, self.stride, self.padding, 1, 2*self.in_channels//self.fold_div)
        # x_div1 = torch.tanh(identity*shift)
        #
        # out = torch.cat((x_div1, x_div2), 1)
        #
        # return out

        # second-order bilinear
        n, c, t, s = x.size()
        x_div1 = x
        # x_div1 = x[:,:c//4,:,:]
        # x_div2 = x[:,c//4:,:,:]
        # identity = x_div1
        # pdb.set_trace()
        x_div1 = torch.mul(torch.sign(x_div1),torch.sqrt(torch.abs(x_div1)+1e-12))
        shift1 = F.conv2d(x_div1, self.weight1, None, self.stride, self.padding, 1, self.in_channels)
        shift2 = F.conv2d(x_div1, self.weight2, None, self.stride, self.padding, 1, self.in_channels)

        x_div1 = self.bn(shift1*shift2)
        x_div1 = self.relu(x_div1)

        # out = torch.cat((x_div1, x_div2), 1)
        out = x_div1
        return out

class TemporalGlobal(nn.Module):
    def __init__(self, net, n_segment=3, has_att=False, include_loss=False, n_div=8, shift_kernel=3, shift_grad=0.0):
        super(TemporalGlobal, self).__init__()
        self.net = net
        assert isinstance(net, torchvision.models.resnet.Bottleneck)
        self.n_segment = n_segment
        self.has_att = has_att
        self.fold_div = n_div
        self.include_loss = include_loss
        if has_att:
            # self.diff_conv = TemporalDiffChannel(net.conv1.in_channels, net.conv1.in_channels, n_segment, (1,1), padding=(0, 0), n_div=n_div, bias=False)
            self.accum_att = AccumAtt(net.conv1.in_channels, n_segment)
        self.shift_conv = TemporalShift(net.conv1.in_channels, (shift_kernel,1), padding=(shift_kernel//2,0), n_div=n_div, bias=False)
        # self.shift_conv_bilinear = TemporalShiftBilinearLocal(net.conv1.in_channels, (shift_kernel,1), padding=(shift_kernel//2,0), n_div=n_div, bias=False)
        if shift_grad - 0.0 < 1e-5:
            self.shift_conv.weight.requires_grad = False
        else:
            self.shift_conv.weight.requires_grad = True

        # Index for Temporal Pairwise Consine Similarity
        n_repeat = np.linspace(n_segment-1, 1, n_segment-1).astype(int)
        self.ind1 = np.linspace(0, n_segment-2, n_segment-1).astype(int).repeat(n_repeat, axis=0)
        self.ind1 = torch.tensor(self.ind1).long()

        self.ind2 = np.empty(0)
        for i in range(n_segment-1):
            ind_new = np.linspace(i+1, n_segment-1, n_segment-i-1).astype(int)
            self.ind2 = np.concatenate((self.ind2, ind_new), axis=0)
        self.ind2 = torch.tensor(self.ind2).long()

        # Index for neighbor frames
        self.ind3 = np.linspace(0, n_segment-2, n_segment-1).astype(int)
        self.ind3 = torch.tensor(self.ind3).long()
        self.ind4 = self.ind3 + 1

        # Index for frames far away
        n_repeat = np.linspace(n_segment-2, 1, n_segment-2).astype(int)
        self.ind5 = np.linspace(0, n_segment-3, n_segment-2).astype(int).repeat(n_repeat, axis=0)
        self.ind6 = np.empty(0)
        for i in range(n_segment-2):
            ind_new = np.linspace(i+2, n_segment-1, n_segment-i-2).astype(int)
            self.ind6 = np.concatenate((self.ind6, ind_new), axis=0)
        self.ind6 = torch.tensor(self.ind6).long()

        # pdb.set_trace()


    def forward(self, in_vec):

        x = in_vec[0]
        td_loss = in_vec[1]

        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment

        identity = x
        if self.has_att:
            out = self.accum_att(x)
        else:
            out = x
        reshape_x = out.view(n_batch, -1, c, h*w).permute(0, 2, 1, 3).contiguous()
        shift_x = self.shift_conv(reshape_x)
        shift_x = shift_x.permute(0,2,1,3).contiguous().view(nt, c, h, w)


        if self.include_loss:
            out_td = shift_x
            nt, nc, w, h = out_td.size()
            n_batch = nt // self.n_segment

            nc_split = nc//2

            out_t = out_td.view(n_batch, self.n_segment, nc, w, h).permute(0,2,1,3,4).view(n_batch, nc, self.n_segment, -1)
            out_t1 = out_t.index_select(2, self.ind1.cuda())
            out_t2 = out_t.index_select(2, self.ind2.cuda())
            td_loss += torch.cosine_similarity(out_t1, out_t2, dim=3).mean(2).sum(1)
            td_simi = torch.cosine_similarity(out_t1[:,:nc_split], out_t2[:,:nc_split],dim=3).mean(2).sum(1)
            td_diff = torch.cosine_similarity(out_t1[:,nc_split:], out_t2[:,nc_split:],dim=3).mean(2).sum(1)
            td_loss += td_diff - td_simi

            # out_t = out.view(n_batch, self.n_segment, nc, w, h).permute(0,2,1,3,4).view(n_batch, nc, self.n_segment, -1)
            # out_t1 = out_t.index_select(2, self.ind3.cuda())
            # out_t2 = out_t.index_select(2, self.ind4.cuda())
            # td_simi = ((1-torch.cosine_similarity(out_t1, out_t2, dim=3))/2).mean(2).sum(1)
            #
            # out_t3 = out_t.index_select(2, self.ind5.cuda())
            # out_t4 = out_t.index_select(2, self.ind6.cuda())
            # pdb.set_trace()
            # td_diff = torch.cosine_similarity(out_t3, out_t4, dim=3).mean(2).sum(1)


        else:

            td_loss += 0

#        identity = x
#        if self.has_att:
#            out = self.accum_att(x)
#        else:
#            out = x
#        reshape_x = out.view(n_batch, -1, c, h*w).permute(0, 2, 1, 3).contiguous()
#        shift_x = self.shift_conv(reshape_x)
#        shift_x = shift_x.permute(0,2,1,3).contiguous().view(nt, c, h, w)

        out = self.net.conv1(shift_x)
        out = self.net.bn1(out)
        out = self.net.relu(out)

        out = self.net.conv2(out)
        out = self.net.bn2(out)
        out = self.net.relu(out)

        out = self.net.conv3(out)
        out = self.net.bn3(out)

        if self.net.downsample is not None:
            identity = self.net.downsample(x)

        out += identity
        out = self.net.relu(out)

        return out, td_loss


class TemporalPool(nn.Module):
    def __init__(self, net, n_segment):
        super(TemporalPool, self).__init__()
        self.net = net
        self.n_segment = n_segment

    def forward(self, x):
        x = self.temporal_pool(x, n_segment=self.n_segment)
        return self.net(x)

    @staticmethod
    def temporal_pool(x, n_segment):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
        x = F.max_pool3d(x, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        x = x.transpose(1, 2).contiguous().view(nt // 2, c, h, w)
        return x


def make_temporal_shift(net, n_segment, has_att=False, n_div=8, place='blockres', shift_type='iCover', shift_kernel=3, shift_grad=0.0, temporal_pool=False):
    if temporal_pool:
        n_segment_list = [n_segment, n_segment // 2, n_segment // 2, n_segment // 2]
    else:
        n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0
    print('=> n_segment per stage: {}'.format(n_segment_list))

    import torchvision
    if isinstance(net, torchvision.models.ResNet):
        if place == 'block':
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                # print(blocks)
                # exit()
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i] = TemporalShift(b, n_segment=this_segment, n_div=n_div, shift_type=shift_type)
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])

        elif 'blockres' in place:
            n_round = 1
            if len(list(net.layer3.children())) >= 23:
                print('=> Using n_round {} to insert temporal shift'.format(n_round))

            def make_block_temporal(stage, this_segment, has_att, stg):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks residual'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    # if i == len(blocks)-1:
                        # blocks[i].conv1 = TemporalShift(b.conv1, n_segment=this_segment, n_div=n_div, shift_type=shift_type)
                    blocks[i] = TemporalGlobal(b, n_segment=this_segment, has_att=has_att, n_div=n_div, include_loss=True if (i == len(blocks)-1 and stg != 1) else False)
                return nn.Sequential(*blocks)

            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0], has_att, 1)
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1], has_att, 2)
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2], has_att, 3)
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3], has_att, 4)
    else:
        raise NotImplementedError(place)


def make_temporal_pool(net, n_segment):
    import torchvision
    if isinstance(net, torchvision.models.ResNet):
        print('=> Injecting nonlocal pooling')
        net.layer2 = TemporalPool(net.layer2, n_segment)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    # test inplace shift v.s. vanilla shift
    tsm1 = TemporalShift(nn.Sequential(), n_segment=8, n_div=8, inplace=False)
    tsm2 = TemporalShift(nn.Sequential(), n_segment=8, n_div=8, inplace=True)

    print('=> Testing CPU...')
    # test forward
    with torch.no_grad():
        for i in range(10):
            x = torch.rand(2 * 8, 3, 224, 224)
            y1 = tsm1(x)
            y2 = tsm2(x)
            assert torch.norm(y1 - y2).item() < 1e-5

    # test backward
    with torch.enable_grad():
        for i in range(10):
            x1 = torch.rand(2 * 8, 3, 224, 224)
            x1.requires_grad_()
            x2 = x1.clone()
            y1 = tsm1(x1)
            y2 = tsm2(x2)
            grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
            grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
            assert torch.norm(grad1 - grad2).item() < 1e-5

    print('=> Testing GPU...')
    tsm1.cuda()
    tsm2.cuda()
    # test forward
    with torch.no_grad():
        for i in range(10):
            x = torch.rand(2 * 8, 3, 224, 224).cuda()
            y1 = tsm1(x)
            y2 = tsm2(x)
            assert torch.norm(y1 - y2).item() < 1e-5

    # test backward
    with torch.enable_grad():
        for i in range(10):
            x1 = torch.rand(2 * 8, 3, 224, 224).cuda()
            x1.requires_grad_()
            x2 = x1.clone()
            y1 = tsm1(x1)
            y2 = tsm2(x2)
            grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
            grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
            assert torch.norm(grad1 - grad2).item() < 1e-5
    print('Test passed.')
