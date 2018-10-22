from common import *
from net.imagenet_pretrain_model.resnet import *



class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        self.bn = SynchronizedBatchNorm2d(out_channels)
        #self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        return x




###########################################################################################3


class UNet(nn.Module):


    def __init__(self ):
        super(UNet, self).__init__()

        self.down1 = nn.Sequential(
            ConvBn2d(  1,  64, kernel_size=3, stride=1, padding=1 ),
            nn.ReLU(inplace=True),
            ConvBn2d( 64,  64, kernel_size=3, stride=1, padding=1 ),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            ConvBn2d( 64, 128, kernel_size=3, stride=1, padding=1 ),
            nn.ReLU(inplace=True),
            ConvBn2d(128, 128, kernel_size=3, stride=1, padding=1 ),
            nn.ReLU(inplace=True),
        )
        self.down3 = nn.Sequential(
            ConvBn2d(128, 256, kernel_size=3, stride=1, padding=1 ),
            nn.ReLU(inplace=True),
            ConvBn2d(256, 256, kernel_size=3, stride=1, padding=1 ),
            nn.ReLU(inplace=True),
            ConvBn2d(256, 256, kernel_size=3, stride=1, padding=1 ),
            nn.ReLU(inplace=True),
        )
        self.down4 = nn.Sequential(
            ConvBn2d(256, 512, kernel_size=3, stride=1, padding=1 ),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1 ),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1 ),
            nn.ReLU(inplace=True),
        )
        self.down5 = nn.Sequential(
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1 ),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1 ),
            nn.ReLU(inplace=True),
        )

        self.center = nn.Sequential(
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1 ),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1 ),
            nn.ReLU(inplace=True),
        )

        self.up5 = nn.Sequential(
            ConvBn2d(1024,512, kernel_size=3, stride=1, padding=1 ),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1 ),
            nn.ReLU(inplace=True),
        )

        self.up4 = nn.Sequential(
            ConvBn2d(1024,512, kernel_size=3, stride=1, padding=1 ),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1 ),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 256, kernel_size=3, stride=1, padding=1 ),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            ConvBn2d(512, 256, kernel_size=3, stride=1, padding=1 ),
            nn.ReLU(inplace=True),
            ConvBn2d(256, 256, kernel_size=3, stride=1, padding=1 ),
            nn.ReLU(inplace=True),
            ConvBn2d(256, 128, kernel_size=3, stride=1, padding=1 ),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            ConvBn2d(256, 128, kernel_size=3, stride=1, padding=1 ),
            nn.ReLU(inplace=True),
            ConvBn2d(128,  64, kernel_size=3, stride=1, padding=1 ),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.Sequential(
            ConvBn2d(128,  64, kernel_size=3, stride=1, padding=1 ),
            nn.ReLU(inplace=True),
            ConvBn2d( 64,  64, kernel_size=3, stride=1, padding=1 ),
            nn.ReLU(inplace=True),
        )
        self.feature = nn.Sequential(
            ConvBn2d( 64,  64, kernel_size=1, stride=1, padding=0 ),
            nn.ReLU(inplace=True),
        )
        self.logit = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0 )





    def forward(self, x):
        #batch_size,C,H,W = x.shape

        down1 = self.down1(x)
        f     = F.max_pool2d(down1, kernel_size=2, stride=2) #, return_indices=True)
        down2 = self.down2(f)
        f     = F.max_pool2d(down2, kernel_size=2, stride=2)
        down3 = self.down3(f)
        f     = F.max_pool2d(down3, kernel_size=2, stride=2)
        down4 = self.down4(f)
        f     = F.max_pool2d(down4, kernel_size=2, stride=2)
        down5 = self.down5(f)
        f     = F.max_pool2d(down5, kernel_size=2, stride=2)

        f = self.center(f)

        #f = F.max_unpool2d(f, i4, kernel_size=2, stride=2)
        f = F.upsample(f, scale_factor=2, mode='bilinear', align_corners=True)
        f = self.up5(torch.cat([down5, f],1))

        f = F.upsample(f, scale_factor=2, mode='bilinear', align_corners=True)
        f = self.up4(torch.cat([down4, f],1))

        f = F.upsample(f, scale_factor=2, mode='bilinear', align_corners=True)
        f = self.up3(torch.cat([down3, f],1))

        f = F.upsample(f, scale_factor=2, mode='bilinear', align_corners=True)
        f = self.up2(torch.cat([down2, f],1))

        f = F.upsample(f, scale_factor=2, mode='bilinear', align_corners=True)
        f = self.up1(torch.cat([down1, f],1))

        f = self.feature(f)
        #f = F.dropout(f, p=0.5)
        logit = self.logit(f)

        return logit

    ##-----------------------------------------------------------------

    def criterion(self, logit, truth):
        #loss = PseudoBCELoss2d()(logit, truth)
        #loss = RobustFocalLoss2d()(logit, truth, type='sigmoid', limit=2)
        loss = FocalLoss2d()(logit, truth, type='sigmoid')
        return loss


    def metric(self, logit, truth, threshold=0.5 ):
        prob = F.sigmoid(logit)
        correct = accuracy(prob, truth, threshold=threshold, is_average=True)
        return correct

    ##-----------------------------------------------------------------

    def set_mode(self, mode ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError


SaltNet = UNet



### run ##############################################################################



def run_check_net():

    batch_size = 8
    C,H,W = 1, 128, 128

    input = np.random.uniform(0,1, (batch_size,C,H,W)).astype(np.float32)
    truth = np.random.choice (2,   (batch_size,C,H,W)).astype(np.float32)


    #------------
    input = torch.from_numpy(input).float().cuda()
    truth = torch.from_numpy(truth).float().cuda()


    #---
    net = SaltNet().cuda()
    net.set_mode('train')
    # print(net)
    # exit(0)


    logit = net(input)
    loss  = net.criterion(logit, truth)
    correct = net.metric(logit, truth)

    print('loss : %0.8f'%loss.item())
    print('dice : %0.8f'%correct.item())
    print('')


    # dummy sgd to see if it can converge ...
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                      lr=0.1, momentum=0.9, weight_decay=0.0001)

    #optimizer = optim.Adam(net.parameters(), lr=0.001)

    i=0
    optimizer.zero_grad()
    while i<=500:

        logit = net(input)
        loss  = net.criterion(logit, truth)
        correct = net.metric(logit, truth)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i%20==0:
            print('[%05d] loss, correct  :  %0.5f,%0.5f'%(i, loss.item(), correct.item()))
        i = i+1







########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_net()

    print( 'sucessful!')