import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #'3,2' #'3,2,1,0'

from common import *
from data   import *


DY0, DY1, DX0, DX1 = compute_center_pad(101,101, factor=32)
Y0, Y1, X0, X1 =  DY0, DY0+101, DX0, DX0+101

##----------------------------------------
from mean_teacher import *
from model_unet_bn import SaltNet as Net



def valid_augment(image,mask,index):
    cache = Struct(image = image.copy(), mask = mask.copy())
    image, mask = do_pad2(image, mask, DY0, DY1, DX0, DX1)
    return image,mask,index,cache


def train_augment(image,mask,index):
    cache = Struct(image = image.copy(), mask = mask.copy())

    if np.random.rand() < 0.5:
         image, mask = do_horizontal_flip2(image, mask)
         pass

    if np.random.rand() < 0.5:
        c = np.random.choice(4)
        if c==0:
            image, mask = do_random_shift_scale_crop_pad2(image, mask, 0.2)
        if c==1:
            image, mask = do_random_affine_shift_scale_crop_pad2(image, mask, 0.1)
        if c==2:
            image, mask = do_horizontal_shear2( image, mask, dx=np.random.uniform(-0.2,0.2) )
        if c==3:
            image, mask = do_vertical_shear2( image, mask, dy=np.random.uniform(-0.2,0.2) )
        #if c==4:
        #    image, mask = do_shift_scale_rotate2( image, mask, dx=0, dy=0, scale=1, angle=np.random.uniform(-15,15))  #10
        #if c==5:
        #    image, mask = do_elastic_transform2(image, mask, grid=np.random.randint(16,32), distort=np.random.uniform(0,0.2))#0.10
        #if c==6:
        #    image, mask = do_elastic1_transform2(image, mask,
        #        dx=np.random.uniform(-0.01,0.01), dy=np.random.uniform(-0.01,0.01),distort=np.random.uniform(-0.5,0.5))#0.10
        pass

    # if np.random.rand() < 0.5:
    #     c = np.random.choice(3)
    #     if c==0:
    #         image = do_brightness_shift(image,np.random.uniform(-0.1,+0.1))
    #     if c==1:
    #         image = do_brightness_multiply(image,np.random.uniform(1-0.08,1+0.08))
    #     if c==2:
    #         image = do_gamma(image,np.random.uniform(1-0.08,1+0.08))
    #     # if c==1:
    #     #     image = do_invert_intensity(image)


    image, mask = do_pad2(image, mask, DY0, DY1, DX0, DX1)
    return image,mask,index,cache



### training ##############################################################

def do_valid( net, valid_loader ):

    valid_num  = 0
    valid_loss = np.zeros(3,np.float32)

    predicts = []
    truths   = []

    for input, truth, index, cache in valid_loader:
        input = input.cuda()
        truth = truth.cuda()
        with torch.no_grad():
            logit = net(input) #data_parallel(net,input) #net(input)
            prob  = F.sigmoid(logit)
            loss  = net.criterion(logit, truth)
            correct  = net.metric(logit, truth)

        batch_size = len(index)
        valid_loss += batch_size*np.array(( loss.item(), correct.item(), 0))
        valid_num += batch_size

        prob  = prob [:,:,Y0:Y1, X0:X1]
        predicts.append(prob.data.cpu().numpy())

        for c in cache:
             truths.append(c.mask)


    assert(valid_num == len(valid_loader.sampler))
    valid_loss  = valid_loss/valid_num

    #--------------------------------------------------------
    predicts = np.concatenate(predicts).squeeze()
    truths   = np.array(truths)

    precision, result, threshold, iou  = do_kaggle_metric(predicts, truths)
    valid_loss[2] = precision.mean()

    return valid_loss




def run_train():

    out_dir = '/root/share/project/kaggle/tgs/results/new/unet/fold0_reference_single2'
    initial_checkpoint = \
        None  #'/root/share/project/kaggle/tgs/results/se_resnet50_256/fold2-pretrain/checkpoint/00021000_model.pth'


    pretrain_file = \
       None  #'/root/share/project/kaggle/tgs/data/model/resnet34-333f7ec4.pth'



    ## setup  -----------------
    os.makedirs(out_dir +'/checkpoint', exist_ok=True)
    os.makedirs(out_dir +'/train', exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')
    log.write('\t<additional comments>\n')
    log.write('\t  ... \n')
    log.write('\n')


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    batch_size = 8  #16  #6 #

    train_dataset_l = TsgDataset('list_train0a_1800', train_augment, 'train')
    train_loader_l  = DataLoader(
                        train_dataset_l,
                        sampler     = FixLengthRandomSampler(train_dataset_l,3600),
                        #sampler     = RandomSampler(train_dataset_l),
                        #sampler     = ConstantSampler(train_dataset,[31]*batch_size*100),
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = 4,
                        pin_memory  = True,
                        collate_fn  = null_collate)

    train_dataset_u = TsgDataset('list_train0b_1800', train_augment, 'train')
    train_loader_u  = DataLoader(
                        train_dataset_u,
                        sampler     = FixLengthRandomSampler(train_dataset_u,3600),
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = 4,
                        pin_memory  = True,
                        collate_fn  = null_collate)


    valid_dataset = TsgDataset('list_valid0_400', valid_augment, 'train')
    valid_loader  = DataLoader(
                        valid_dataset,
                        sampler     = SequentialSampler(valid_dataset),  #
                        batch_size  = 16,
                        drop_last   = False,
                        num_workers = 4,
                        pin_memory  = True,
                        collate_fn  = null_collate)

    assert(len(train_dataset_l)>=batch_size)
    assert(len(train_dataset_u)>=batch_size)
    assert(len(valid_dataset  )>=batch_size)

    log.write('batch_size = %d\n'%(batch_size))
    log.write('train_dataset_l.split = %s\n'%(train_dataset_l.split ))
    log.write('train_dataset_u.split = %s\n'%(train_dataset_u.split ))
    log.write('valid_dataset.split   = %s\n'%(valid_dataset.split ))

    log.write('\n')

    #debug
    # if 0: #debug  ##-------------------------------
    #
    #     for input, truth, index, cache in train_loader:
    #         images = input.cpu().data.numpy().squeeze()
    #         masks  = truth.cpu().data.numpy().squeeze()
    #         batch_size = len(index)
    #         for b in range(batch_size):
    #             image = images[b]*255
    #             image = np.dstack([image,image,image])
    #
    #             mask = masks[b]
    #
    #             image_show('image',image,resize=2)
    #             image_show_norm('mask', mask, max=1,resize=2)
    #             overlay0 = draw_mask_overlay(mask, image, color=[0,0,255])
    #             overlay0 = draw_mask_to_contour_overlay(mask, overlay0, 2, color=[0,0,255])
    #
    #             image_show('overlay0',overlay0,resize=2)
    #             cv2.waitKey(0)
    # #--------------------------------------




    ## net ----------------------------------
    log.write('** net setting **\n')
    net_student = Net().cuda()


    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net_student.load_state_dict(torch.load(
            initial_checkpoint.replace('_model.pth','_model.student.pth'), map_location=lambda storage, loc: storage))
        net_teacher.load_state_dict(torch.load(
            initial_checkpoint.replace('_model.pth','_model.teacher.pth'), map_location=lambda storage, loc: storage))

    if pretrain_file is not None:
        log.write('\tpretrain_file = %s\n' % pretrain_file)
        net_student.load_pretrain(pretrain_file)
        net_teacher.load_pretrain(pretrain_file)



    log.write('%s\n'%(type(net_student)))
    log.write('\n')



    ## optimiser ----------------------------------
    num_iters   = 300  *1000
    iter_smooth = 20
    iter_log    = 50
    iter_valid  = 100
    iter_save   = [0, num_iters-1]\
                   + list(range(0,num_iters,1000))#1*1000


    schduler = None  #LR = StepLR([ (0, 0.01),  (200, 0.001),  (300, -1)])
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net_student.parameters()),
                          lr=0.01, momentum=0.9, weight_decay=0.0001)

    start_iter = 0
    start_epoch= 0
    if initial_checkpoint is not None:
        checkpoint  = torch.load(initial_checkpoint.replace('_model.pth','_optimizer.pth'))
        start_iter  = checkpoint['iter' ]
        start_epoch = checkpoint['epoch']

        rate = get_learning_rate(optimizer)  #load all except learning rate
        #optimizer.load_state_dict(checkpoint['optimizer'])
        adjust_learning_rate(optimizer, rate)
        pass


    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write('                              |                       valid                     |                   train                          |  \n')
    log.write('--------------------------------------------------------------------------------------------------------------------------------------------------\n')
    log.write('                              |      student           |      teacher           |      loss       |  student acc     teacher acc   |  \n')
    log.write('  rate   iter   epoch  lamda  |  loss     acc  (   LB) |  loss     acc  (   LB) |  label  label   |  label    label  label  label  |  \n')
    log.write('--------------------------------------------------------------------------------------------------------------------------------------------------\n')

    train_loss   = np.zeros(6,np.float32)
    batch_loss   = np.zeros(6,np.float32)
    valid_loss_s = np.zeros(3,np.float32)
    valid_loss_t = np.zeros(3,np.float32)
    unlabel_loss_weight = 0
    rate = 0
    iter = 0
    i    = 0

    start = timer()
    while  iter<num_iters:
        sum_train_loss = np.zeros(6,np.float32)
        sum = 0

        #print('start of while\n',flush=True)
        optimizer.zero_grad()
        for (input_l, truth_l, index_l, cache_l), (input_u, truth_u, index_u, cache_u)  in \
                zip(train_loader_l, train_loader_u):

            len_train_dataset = len(train_dataset_l)
            batch_size = len(index_l)
            iter = i + start_iter
            epoch = (iter-start_iter)*batch_size/len_train_dataset + start_epoch
            num_samples = epoch*len_train_dataset

            #print('start of loader\n',flush=True)
            if iter % iter_valid==0:
            #if 0:
                net_student.set_mode('valid')
                valid_loss_s = do_valid(net_student, valid_loader)


                print('\r',end='',flush=True)
                log.write('%0.4f  %5.1f  %6.1f  %0.3f  |  %0.3f  %0.3f  (%0.3f) |  %0.3f  %0.3f  (%0.3f) |  %0.3f   %0.3f  |  %0.3f  %0.3f    %0.3f  %0.3f  | %s \n' % (\
                         rate, iter/1000, epoch, unlabel_loss_weight,
                         valid_loss_s[0], valid_loss_s[1], valid_loss_s[2],
                         valid_loss_t[0], valid_loss_t[1], valid_loss_t[2],
                         train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4], train_loss[5],
                         #batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3],batch_loss[4], batch_loss[5],
                         time_to_str((timer() - start),'min')))
                time.sleep(0.01)

            #if 1:
            if iter in iter_save:
                # torch.save(net_student.state_dict(),out_dir +'/checkpoint/%08d_model.student.pth'%(iter))
                # torch.save(net_teacher.state_dict(),out_dir +'/checkpoint/%08d_model.teacher.pth'%(iter))
                # torch.save({
                #     #'optimizer': optimizer.state_dict(),
                #     'iter'     : iter,
                #     'epoch'    : epoch,
                # }, out_dir +'/checkpoint/%08d_optimizer.pth'%(iter))
                pass


            #print('!!!')
            # learning rate schduler -------------
            if schduler is not None:
                lr = schduler.get_rate(iter)
                if lr<0 : break
                adjust_learning_rate(optimizer, lr)
            rate = get_learning_rate(optimizer)


            # one iteration update  -------------
            net_student.set_mode('train')

            input_l = input_l.cuda()
            truth_l = truth_l.cuda()
            input_u = input_u.cuda()
            truth_u = truth_u.cuda()


            logit_s_l = net_student(input_l) #data_parallel(net_student,input_l)
            logit_s_u = net_student(input_u) #data_parallel(net_student,input_u)

            loss_l = net_student.criterion(logit_s_l, truth_l)
            loss_u = net_student.criterion(logit_s_u, truth_u)

            correct_s_l = net_student.metric(logit_s_l, truth_l)
            correct_s_u = net_student.metric(logit_s_u, truth_u)  ## <debug only>

            loss = (loss_l + loss_u)/2
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            #torch.nn.utils.clip_grad_norm(net.parameters(), 1)


            # Use the true average until the exponential average is more correct
            #alpha = alpha #min(1 - 1 / (step + 1), alpha)

            # print statistics  ------------
            batch_loss = np.array((
                           loss_l.item(), loss_u.item(),
                           correct_s_l.item(), correct_s_u.item(), 0, 0,
                         ))
            sum_train_loss += batch_loss
            sum += 1
            if iter%iter_smooth == 0:
                train_loss = sum_train_loss/sum
                sum_train_loss = np.zeros(6,np.float32)
                sum = 0


            print('\r%0.4f  %5.1f  %6.1f  %0.3f  |  %0.3f  %0.3f  (%0.3f) |  %0.3f  %0.3f  (%0.3f) |  %0.3f   %0.3f  |  %0.3f  %0.3f    %0.3f  %0.3f  | %s' % (\
                         rate, iter/1000, epoch, unlabel_loss_weight,
                         valid_loss_s[0], valid_loss_s[1], valid_loss_s[2],
                         valid_loss_t[0], valid_loss_t[1], valid_loss_t[2],
                         #train_loss[0], train_loss[1], train_loss[2], train_loss[3],train_loss[4], train_loss[5],
                         batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3],batch_loss[4], batch_loss[5],
                         time_to_str((timer() - start), 'min')), end='',flush=True)
            i=i+1


            #<debug> ===================================================================
            # if 0:
            # #if iter%200==0:
            #     #voxel, aux, query, link, truth, cache = make_valid_batch(valid_dataset.dataset, batch_size=2)
            #
            #     net.set_mode('test')#
            #     with torch.no_grad():
            #         logit = net(input)
            #         prob  = F.sigmoid(logit)
            #         loss  = net.criterion(logit, truth)
            #         dice  = net.metric(logit, truth)
            #
            #         if 0:
            #             loss  = net.criterion(logit, truth)
            #             accuracy,hit_rate,precision_rate = net.metric(logit, truth)
            #             valid_loss[0] = loss.item()
            #             valid_loss[1] = accuracy.item()
            #             valid_loss[2] = hit_rate.item()
            #             valid_loss[3] = precision_rate.item()
            #
            #
            #
            #     #show only b in batch ---
            #     b = 1
            #     prob   = prob.data.cpu().numpy()[b].squeeze()
            #     truth  = truth.data.cpu().numpy()[b].squeeze()
            #     input  = input.data.cpu().numpy()[b].squeeze()
            #
            #     all = np.hstack([input,truth,prob])
            #     image_show_norm('all',all,max=1,resize=3)
            #     cv2.waitKey(100)
            #
            #     net.set_mode('train')
            #<debug> ===================================================================


        pass  #-- end of one data loader --
    pass #-- end of all iterations --


    if 1: #save last
        torch.save(net.state_dict(),out_dir +'/checkpoint/%d_model.pth'%(i))
        torch.save({
            'optimizer': optimizer.state_dict(),
            'iter'     : i,
            'epoch'    : epoch,
        }, out_dir +'/checkpoint/%d_optimizer.pth'%(i))

    log.write('\n')



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_train()


    print('\nsucess!')



#  ffmpeg -f image2  -pattern_type glob -r 33 -i "iterations/*.png" -c:v libx264  iterations.mp4
#  convert *.png animated.gif
#
