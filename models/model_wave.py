import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch.autograd as autograd

QUANTIZATION_FLAG = True

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


def weight_quantization(w, bit):
    w = torch.tanh(w)
    alpha = w.data.abs().max()
    w = torch.clamp(w/alpha,min=-1,max=1)
    w = w*(2**(bit-1)-1)
    w_hat = (w.round()-w).detach()+w
    return w_hat*alpha/(2**(bit-1)-1)


def draw_hist(x, ly_idx):
    MEDIUM_SIZE = 15
    BIG_SIZE = 29

    plt.rc('font', size=BIG_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIG_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIG_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIG_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIG_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIG_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIG_SIZE)  # fontsize of the figure title
    plt.rcParams['lines.linewidth'] = 2
    fig, ax = plt.subplots(figsize=(7, 5.5))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    vis_z = x
    vis_z[vis_z == vis_z.max()] = -1000
    np_visual = torch.log(vis_z).flatten().cpu().numpy()
    plt.hist(np_visual, bins=150,color="#8DA5C1")
    plt.xlim([0, 6.0])
    # plt.yticks([1e4])
    plt.ylabel('Frequency')
    plt.xlabel('Spike Timing')
    plt.tick_params(direction='in')
    fig.tight_layout()
    plt.savefig('vis/hist_vis_{}.pdf'.format(ly_idx))


class t_Linear(torch.nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=False,
            weight_ext=None,
            max_spike_time=1e1, tau = 1):
        super(t_Linear, self).__init__(in_features, out_features, bias)

        self.MAX_SPIKE_TIME = max_spike_time
        self.in_features = in_features
        self.out_features = out_features
        self.th = 1
        self.tau =tau

        if weight_ext is None:
            torch.nn.init.uniform_(self.weight, 0.0, 10 / self.in_features)
        else:
            self.weight = torch.nn.Parameter(weight_ext)


    def forward(self, inp):

        batch_num = inp.shape[0]  ## for later dimension extension

        temp_MAX_SPIKE_TIME = self.MAX_SPIKE_TIME


        sorted_inp, indic = torch.sort(inp)

        indic = torch.unsqueeze(indic, -1).requires_grad_(False)
        indic = torch.tile(indic, (1, 1, self.out_features)).requires_grad_(False)

        sorted_inp = torch.unsqueeze(sorted_inp, -1)
        sorted_inp = torch.tile(sorted_inp, (1, 1, self.out_features))  ## extend the dimension to mul with weights



        weight_extend = torch.unsqueeze(torch.transpose(self.weight, 0, 1), 0)
        weight_extend = torch.tile(weight_extend, (batch_num, 1, 1))  ## transform the weights
        weight_sorted = torch.gather(weight_extend, 1, indic)

        weight_input_mul = torch.multiply(sorted_inp, weight_sorted)


        weight_sum = torch.cumsum(weight_sorted, dim=1)
        input_weight_sum = torch.cumsum(weight_input_mul, dim=1)

        out_all = torch.div(input_weight_sum, torch.clamp(weight_sum - self.th, 1e-10, 1e10))

        out_spike = torch.where(weight_sum < self.th, temp_MAX_SPIKE_TIME * torch.ones_like(out_all), out_all)
        out_spike_time_valid = torch.where(out_spike < sorted_inp, temp_MAX_SPIKE_TIME * torch.ones_like(out_spike),
                                           out_spike)
        input_sorted_slice = sorted_inp[:, (-self.in_features + 1):, :]
        one_tensor = torch.ones_like(input_sorted_slice)[:, 0:1, :].requires_grad_(False)
        input_sorted_slice_cat = torch.cat((input_sorted_slice, one_tensor), 1)
        out_spike_valid = torch.where(out_spike_time_valid >= input_sorted_slice_cat,
                                      temp_MAX_SPIKE_TIME * torch.ones_like(out_spike_time_valid), out_spike_time_valid)


        out, _ = torch.min(out_spike_valid, dim=1)


        return out

    def w_sum_cost(self):
        threshold = self.th
        p1 = threshold  - torch.sum(self.weight, 1)
        p2 = torch.where(p1 > 0, p1, torch.zeros_like(p1))
        return torch.mean(p2)

    def l2_cost(self):
        w_sqr = torch.square(self.weight)
        return torch.mean(w_sqr)


class t_Linear_c(torch.nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=False,
            weight_ext=None,
            bias_ext=None,
            max_spike_time=1e2):
        super(t_Linear_c, self).__init__(in_features, out_features, bias)

        self.MAX_SPIKE_TIME = max_spike_time
        self.in_features = in_features
        self.out_features = out_features
        self.th = 1
        if weight_ext is None:
            torch.nn.init.uniform_(self.weight, 0.0, 10 / self.in_features)
        else:
            self.weight = torch.nn.Parameter(weight_ext)



    def forward(self, inp):

        batch_num = inp.shape[0]  ## for later dimension extension

        sorted_inp, indic = torch.sort(inp) # indic is ascending

        indic = torch.unsqueeze(indic, -1)
        indic = torch.tile(indic, (1, 1, self.out_features))

        sorted_inp = torch.unsqueeze(sorted_inp, -1)
        sorted_inp = torch.tile(sorted_inp, (1, 1, self.out_features))  ## extend the dimension to mul with weights

        weight_extend = torch.unsqueeze(torch.unsqueeze(self.weight.permute(1, 0), 0),0)
        weight_extend = torch.tile(weight_extend, (batch_num, inp.shape[1], 1, 1))  ## transform the weights

        weight_sorted = torch.gather(weight_extend, 2, indic)

        if QUANTIZATION_FLAG:
            weight_sorted = weight_quantization(weight_sorted,bit=8)

        weight_input_mul = torch.multiply(sorted_inp, weight_sorted)

        weight_sum = torch.cumsum(weight_sorted, dim=-2)
        input_weight_sum = torch.cumsum(weight_input_mul, dim=-2)
        out_all = torch.div(input_weight_sum, torch.clamp(weight_sum - self.th, 1e-10, 1e10))


        out_spike = torch.where(weight_sum < self.th, self.MAX_SPIKE_TIME * torch.ones_like(out_all), out_all) # dinominator > 0: valid condition
        out_spike_time_valid = torch.where(out_spike < sorted_inp, self.MAX_SPIKE_TIME * torch.ones_like(out_spike), # out spike happens before input spike
                                           out_spike)

        input_sorted_slice = sorted_inp[:, :, (-self.in_features + 1):, :]
        one_tensor = torch.ones_like(input_sorted_slice)[:, :, 0:1, :].requires_grad_(False)
        input_sorted_slice_cat = torch.cat((input_sorted_slice, one_tensor), 2)
        out_spike_valid = torch.where(out_spike_time_valid > input_sorted_slice_cat,
                                      self.MAX_SPIKE_TIME * torch.ones_like(out_spike_time_valid), out_spike_time_valid)
        out, _ = torch.min(out_spike_valid, dim=-2)

        return out

    def w_sum_cost(self):
        threshold = self.th
        p1 = threshold - torch.sum(self.weight, 1)
        p2 = torch.where(p1 > 0, p1, torch.zeros_like(p1))
        return torch.mean(p2)

    def l2_cost(self):
        w_sqr = torch.square(self.weight)
        return torch.mean(w_sqr)


class t_Conv(torch.nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            padding=1,
            strides=1,
            bias=False,
            weight_ext=None,
            bias_ext=None,
            max_spike_time = 1e2):
        super(t_Conv, self).__init__()

        self.MAX_SPIKE_TIME = max_spike_time
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.strides = strides
        self.padding = padding
        self.kernel = t_Linear_c(in_features=self.kernel_size * self.kernel_size * self.in_channel,
                                 out_features=self.out_channel, weight_ext=weight_ext, max_spike_time=max_spike_time)

    def forward(self, inp):
        input_shape = inp.shape


        patches = F.unfold(inp, self.kernel_size, padding=self.padding, stride=self.strides)

        patches = patches.permute(0,2,1)
        patches = torch.where(torch.lt(patches, 0.1), self.MAX_SPIKE_TIME * torch.ones_like(patches), patches) # put ,max spike time if (input time)<0.1, why 0.1?
        out = self.kernel(patches).transpose(-1, -2)
        out_shape_h = math.floor((input_shape[-2] + self.padding * 2 - self.kernel_size) / self.strides) + 1
        out_shape_w = math.floor((input_shape[-1] + self.padding * 2 - self.kernel_size) / self.strides) + 1
        out = out.view(input_shape[0], self.out_channel, out_shape_h, out_shape_w)
        return out



class mid_vgg_direct_residual_wave(nn.Module):
    def __init__(self, max_t, n_class=10):
        super(mid_vgg_direct_residual_wave, self).__init__()
        self.scale = max_t
        self.max_z = (np.exp(max_t))
        ratio=2
        self.conv1 = nn.Conv2d(1, 16*ratio, 5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(16*ratio)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = t_Conv(16*ratio, 16*ratio, 3, padding=1, strides=2, bias=False, max_spike_time=self.max_z)
        self.conv3 = t_Conv(16*ratio, 16*ratio, 3, padding=1, strides=1, bias=False, max_spike_time=self.max_z)


        self.conv4 = t_Conv(16*ratio, 32*ratio, 3, padding=1, strides=2, bias=False, max_spike_time=self.max_z*2)
        self.conv5 = t_Conv(32*ratio, 32*ratio, 3, padding=1, strides=1, bias=False, max_spike_time=self.max_z*2)

        self.fc_1 = t_Linear(8 * 8 * 32*ratio, n_class, bias=False, max_spike_time=self.max_z*4)

    def forward(self, inp, less_than_first = False):

        x = F.relu(self.bn1(self.conv1(inp)))  # input range 1~2^max
        x = self.pool1(x)
        x = torch.clamp(x, max=self.scale)
        z1 = torch.exp(x)

        z2 = self.conv2(z1)
        w2 = self.conv2.kernel.w_sum_cost()
        z3 = self.conv3(z2)
        w3 = self.conv3.kernel.w_sum_cost()
        pre_min = -nn.MaxPool2d(3,stride=2,padding=0)(-nn.ConstantPad2d(1, 1e5)(z1))
        z3_combine = (z3*pre_min)


        z4 = self.conv4(z3_combine)
        w4 = self.conv4.kernel.w_sum_cost()
        z5 = self.conv5(z4)
        w5 = self.conv5.kernel.w_sum_cost()
        pre_min = -nn.MaxPool2d(3,stride=2,padding=0)(-(nn.ConstantPad2d(1, 1e5)(z3_combine)))
        pre_min = torch.cat([pre_min,pre_min], dim = 1)


        z5_combine = (z5*pre_min)
        z5_combine = z5_combine.contiguous().view(z5.shape[0], -1)


        z_fc = self.fc_1(z5_combine)
        w_fc = self.fc_1.w_sum_cost()


        if less_than_first == True:
            first_spike_time, predicted = torch.min(z_fc.data, 1)
            first_spike_time = first_spike_time.unsqueeze(1).unsqueeze(2).unsqueeze(3)

            conv2_val_percent = ((z2 <= first_spike_time).sum() / torch.numel(z2)).cpu().item() * 100
            conv3_val_percent = ((z3 <= first_spike_time).sum() / torch.numel(z3)).cpu().item() * 100
            conv4_val_percent = ((z4 <= first_spike_time).sum() / torch.numel(z4)).cpu().item() * 100
            conv5_val_percent = ((z5 <= first_spike_time).sum() / torch.numel(z5)).cpu().item() * 100
            spike_time_val_list = [conv2_val_percent, conv3_val_percent, conv4_val_percent, conv5_val_percent]



        if self.training:
            return z_fc, w2 + w3 + w4 +w5 +w_fc
        else:
            if less_than_first == False:
                return z_fc
            else:
                return z_fc, spike_time_val_list



class mid_shufflenet_direct_wave(nn.Module):
    def __init__(self, max_t, t_init=0.5, n_class=10, img_res=32):
        super(mid_shufflenet_direct_wave, self).__init__()
        self.scale = max_t
        self.max_z = (np.exp(max_t))
        ratio = 1
        self.conv1 = nn.Conv2d(1, 64 * ratio, 5, padding=2, bias=False)
        # self.conv1 = nn.Conv2d(3, 64, 7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64 * ratio)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = t_Conv(32 * ratio, 32 * ratio, 3, padding=1, strides=2, bias=False, max_spike_time=self.max_z)
        self.conv3 = t_Conv(32 * ratio, 32 * ratio, 3, padding=1, strides=1, bias=False, max_spike_time=self.max_z)

        # self.skip1 = t_Conv(16*ratio, 16*ratio, 1, padding=0, strides=2, bias=False, max_spike_time=self.max_z)
        # self.skip2 = t_Conv(32*ratio, 32*ratio, 1, padding=0, strides=2, bias=False, max_spike_time=self.max_z)

        self.conv4 = t_Conv(32 * ratio, 64 * ratio, 3, padding=1, strides=2, bias=False, max_spike_time=self.max_z)
        self.conv5 = t_Conv(64 * ratio, 64 * ratio, 3, padding=1, strides=1, bias=False, max_spike_time=self.max_z)
        # self.conv6 = t_Conv(32*ratio, 32*ratio, 3, padding=1, strides=2, bias=False, max_spike_time=self.max_z)

        # self.fc_1 = t_Linear(4 * 4 * (32+64), 10, bias=False, max_spike_time=self.max_z)
        self.fc_1 = t_Linear(8 * 8 *(32 + 64) * ratio, n_class, bias=False,
                             max_spike_time=self.max_z)

        # self.fc_1 = t_Linear(3 * 3 * 32*ratio, 10, bias=False, max_spike_time=self.max_z**4)
        # self.fc_out = t_Linear(64,10,bias=False)
        # self.input_gamma =nn.Parameter(torch.FloatTensor([0.1]))
        # self.t_shift2 =nn.Parameter(torch.FloatTensor([1]))

        self.time_init = t_init
        # self.t_shift1 = nn.Parameter(torch.ones([1, 1, 1, 1]) * self.time_init)
        # self.t_shift2 = nn.Parameter(torch.ones([1, 1, 1, 1]) * self.time_init)
        self.t_shift1 = nn.Parameter(torch.ones([1, 32 * ratio, 1, 1]) * self.time_init)
        self.t_shift2 = nn.Parameter(torch.ones([1, 32 * ratio, 1, 1]) * self.time_init)
        # self.t_shift1 = nn.Parameter(torch.ones([1, 32*ratio, 7, 7]))
        # self.t_shift2 = nn.Parameter(torch.ones([1, 32*ratio, 4, 4]))
        # self.norm_p = 1

    def forward(self, inp, less_than_first=False):
        x = F.relu(self.bn1(self.conv1(inp)))  # input range 1~2^max
        x = self.pool1(x)
        # x = x**(torch.clamp(self.input_gamma,min=0, max=1))
        x = torch.clamp(x, max=self.scale)
        z1 = torch.exp(x)

        x1, x2 = z1.chunk(2, dim=1)

        z2 = self.conv2(x2)
        w2 = self.conv2.kernel.w_sum_cost()
        z3 = self.conv3(z2)
        w3 = self.conv3.kernel.w_sum_cost()
        pre_min = -nn.MaxPool2d(3, stride=2, padding=0)(-nn.ConstantPad2d(1, 1e5)(x1)) * torch.exp(torch.clamp(self.t_shift1, min=0))

        # draw_hist(z3, 'conv_branch1')
        # draw_hist(pre_min, 'skip1')

        z3_mean = z3.mean(3).mean(2).mean(1).unsqueeze(1)
        pre_min_mean = pre_min.mean(3).mean(2).mean(1).unsqueeze(1)
        force_loss1 = nn.MSELoss()(z3_mean, pre_min_mean)

        x = torch.cat((pre_min, z3), dim=1)
        x = channel_shuffle(x, 2)

        # draw_hist(x, 'combine')

        x1, x2 = x.chunk(2, dim=1)

        z4 = self.conv4(x2)
        w4 = self.conv4.kernel.w_sum_cost()
        z5 = self.conv5(z4)
        w5 = self.conv5.kernel.w_sum_cost()
        pre_min = -nn.MaxPool2d(3, stride=2, padding=0)(-(nn.ConstantPad2d(1, 1e5)(x1))) * torch.exp(torch.clamp(self.t_shift2, min=0))

        # draw_hist(z5, 'conv_branch2')
        # draw_hist(pre_min, 'skip2')

        z5_mean = z5.mean(3).mean(2).mean(1)
        pre_min_mean = pre_min.mean(3).mean(2).mean(1)
        force_loss2 = nn.MSELoss()(z5_mean, pre_min_mean)

        x = torch.cat((pre_min, z5), dim=1)
        x = channel_shuffle(x, 2)
        # print (x.size())
        # draw_hist(x, 'combine2')

        # x = nn.AdaptiveAvgPool2d((1,1))(x)

        z6 = x.contiguous().view(x.shape[0], -1)

        z_fc = self.fc_1(z6)
        w_fc = self.fc_1.w_sum_cost()

        if less_than_first == True:
            first_spike_time, predicted = torch.min(z_fc.data, 1)
            first_spike_time = first_spike_time.unsqueeze(1).unsqueeze(2).unsqueeze(3)

            # print('conv 2-------- valid spike rate')
            conv2_val_percent = ((z2 <= first_spike_time).sum() / torch.numel(z2)).cpu().item() * 100
            # print(conv2_val_percent)
            conv3_val_percent = ((z3 <= first_spike_time).sum() / torch.numel(z3)).cpu().item() * 100
            # print(conv3_val_percent)
            conv4_val_percent = ((z4 <= first_spike_time).sum() / torch.numel(z4)).cpu().item() * 100
            # print(conv4_val_percent)
            conv5_val_percent = ((z5 <= first_spike_time).sum() / torch.numel(z5)).cpu().item() * 100
            # print(conv5_val_percent)
            spike_time_val_list = [conv2_val_percent, conv3_val_percent, conv4_val_percent, conv5_val_percent]

            # print('conv 3-------- valid spike rate')
            # print(((z3 <= first_spike_time).sum() / torch.numel(z3)).cpu().item() * 100)
            # print('conv 4-------- valid spike rate')
            # print(((z4 <= first_spike_time).sum() / torch.numel(z4)).cpu().item() * 100)
            # print('conv 5-------- valid spike rate')
            # print(((z5 <= first_spike_time).sum() / torch.numel(z5)).cpu().item() * 100)

        if self.training:
            return z_fc, w2 + w3 + w4 + w5 + w_fc, force_loss1 + force_loss2
        else:
            if less_than_first == False:
                return z_fc
            else:
                return z_fc, spike_time_val_list



class mid_vgg_direct_wave(nn.Module):
    def __init__(self, max_t,n_class):
        super(mid_vgg_direct_wave, self).__init__()
        self.scale = max_t
        self.max_z = (np.exp(max_t))
        ratio = 2
        self.conv1 = nn.Conv2d(1, 16*ratio, 5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(16*ratio)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = t_Conv(16*ratio, 16*ratio, 3, padding=1, strides=2, bias=False, max_spike_time=self.max_z)
        self.conv3 = t_Conv(16*ratio, 16*ratio, 3, padding=1, strides=1, bias=False, max_spike_time=self.max_z)

        self.conv4 = t_Conv(16*ratio, 32*ratio, 3, padding=1, strides=2, bias=False, max_spike_time=self.max_z)
        self.conv5 = t_Conv(32*ratio, 32*ratio, 3, padding=1, strides=1, bias=False, max_spike_time=self.max_z)


        self.fc_1 = t_Linear(8 * 8 * 32*ratio, n_class, bias=False, max_spike_time=self.max_z)
        self.t_shift1 =nn.Parameter(torch.FloatTensor([1]))
        self.t_shift2 =nn.Parameter(torch.FloatTensor([1]))

    def forward(self, inp, less_than_first = False):

        x = F.relu(self.bn1(self.conv1(inp)))  # input range 1~2^max
        x = self.pool1(x)
        x = torch.clamp(x, max=self.scale)
        z1 = torch.exp(x)

        z2 = self.conv2(z1)
        w2 = self.conv2.kernel.w_sum_cost()

        z3 = self.conv3(z2)
        w3 = self.conv3.kernel.w_sum_cost()

        z4 = self.conv4(z3)
        w4 = self.conv4.kernel.w_sum_cost()

        z5 = self.conv5(z4)
        w5 = self.conv5.kernel.w_sum_cost()

        z6 = z5.contiguous().view(z5.shape[0], -1)

        z_fc = self.fc_1(z6)
        w_fc = self.fc_1.w_sum_cost()


        if less_than_first == True:
            first_spike_time, predicted = torch.min(z_fc.data, 1)
            first_spike_time = first_spike_time.unsqueeze(1).unsqueeze(2).unsqueeze(3)

            conv2_val_percent = ((z2 <= first_spike_time).sum() / torch.numel(z2)).cpu().item() * 100
            conv3_val_percent = ((z3 <= first_spike_time).sum() / torch.numel(z3)).cpu().item() * 100
            conv4_val_percent = ((z4 <= first_spike_time).sum() / torch.numel(z4)).cpu().item() * 100
            conv5_val_percent = ((z5 <= first_spike_time).sum() / torch.numel(z5)).cpu().item() * 100
            spike_time_val_list = [conv2_val_percent, conv3_val_percent, conv4_val_percent, conv5_val_percent]


        if self.training:
            return z_fc, w2 + w3 + w4 + w5 + w_fc
        else:
            if less_than_first == False:
                return z_fc
            else:
                return z_fc, spike_time_val_list

