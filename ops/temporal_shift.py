# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch
import torch.nn as nn
import torch.nn.functional as F

from ops.fex import FEX


class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False, comu_type='replace', is_first_block=False):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace

        self.comu_type = comu_type
        # self.is_first_block = is_first_block

        if self.comu_type == 'FEX':
            print('=> Using FEX')
            self.shift_block = FEX(in_channels=net.in_channels, n_segment=n_segment, n_div=n_div, is_first_block=is_first_block)

        else:
            raise NotImplementedError

        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        x = self.shift_block(x)
        return self.net(x)


def make_temporal_shift(net, n_segment, n_div=8, place='blockres', temporal_pool=False, comu_type='replace'):
    if temporal_pool:
        n_segment_list = [n_segment, n_segment // 2, n_segment // 2, n_segment // 2]
    else:
        n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0
    print('=> n_segment per stage: {}'.format(n_segment_list))

    import torchvision
    if isinstance(net, torchvision.models.ResNet):

        print("==> using comutype:{}".format(comu_type))

        if 'BB' in place:
            borrow_BN=True
            print("=>Borrow BN!!!")
        else:
            borrow_BN=False

        if 'inblockres' in place and borrow_BN:
            n_round = 1
            if len(list(net.layer3.children())) >= 23:
                n_round = 2
                print('=> Using n_round {} to insert temporal shift'.format(n_round))

            def make_block_temporal(stage, this_segment, comu_type='replace'):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks residual'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    if i == 0:
                        blocks[i].conv1 = TemporalShift(b.conv1, n_segment=this_segment, n_div=n_div
                                                        , comu_type=comu_type, is_first_block=True)
                    elif i % n_round == 0:
                        blocks[i].conv1 = TemporalShift(b.conv1, n_segment=this_segment, n_div=n_div
                                                        , comu_type=comu_type)
                return nn.Sequential(*blocks)

            # net.layer1 = make_block_temporal(net.layer1, n_segment_list[0], comu_type=comu_type)
            # net.layer2 = make_block_temporal(net.layer2, n_segment_list[1], comu_type=comu_type)
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2], comu_type=comu_type)
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3], comu_type=comu_type)
        elif 'inblockres' in place and not borrow_BN:
            n_round = 1
            if len(list(net.layer3.children())) >= 23:
                n_round = 2
                print('=> Using n_round {} to insert temporal shift'.format(n_round))

            def make_block_temporal(stage, this_segment, comu_type='replace'):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks residual'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    if i == 0:
                        blocks[i].conv1 = TemporalShift(b.conv1, n_segment=this_segment, n_div=n_div
                                                        , comu_type=comu_type, is_first_block=True)
                    elif i % n_round == 0:
                        blocks[i].conv2 = TemporalShift(b.conv2, n_segment=this_segment, n_div=n_div
                                                        , comu_type=comu_type)
                return nn.Sequential(*blocks)

            # net.layer1 = make_block_temporal(net.layer1, n_segment_list[0], comu_type=comu_type)
            # net.layer2 = make_block_temporal(net.layer2, n_segment_list[1], comu_type=comu_type)
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2], comu_type=comu_type)
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3], comu_type=comu_type)

        elif 'blockres' in place:
            n_round = 1
            if len(list(net.layer3.children())) >= 23:
                n_round = 2
                print('=> Using n_round {} to insert temporal shift'.format(n_round))

            def make_block_temporal(stage, this_segment, comu_type='replace'):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks residual'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    if i == 0:
                        blocks[i].conv1 = TemporalShift(b.conv1, n_segment=this_segment, n_div=n_div
                                                        , comu_type=comu_type, is_first_block=True)
                    elif i % n_round == 0:
                        blocks[i].conv1 = TemporalShift(b.conv1, n_segment=this_segment, n_div=n_div
                                                        , comu_type=comu_type)
                return nn.Sequential(*blocks)

            # net.layer1 = make_block_temporal(net.layer1, n_segment_list[0], comu_type=comu_type)
            # net.layer2 = make_block_temporal(net.layer2, n_segment_list[1], comu_type=comu_type)
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2], comu_type=comu_type)
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3], comu_type=comu_type)
        else:
            raise NotImplementedError(place)
    else:
        raise NotImplementedError(place)


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
