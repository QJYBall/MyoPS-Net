import os
import torch
from itertools import cycle
import torch.optim as optim
from criterion.loss import MyoPSLoss
from utils.tools import weights_init
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from network.model import MyoPSNet
from validation import Validation2d
from utils.dataloader import CrossModalDataLoader


def MyoPSNetTrain(args):

    # C0(5,3) LGE(2,3) T2(2,3) mapping(3,3)
    model = MyoPSNet(in_chs=(5,2,2,3), out_chs=(3,3,3,3)).cuda()
    model.apply(weights_init)

    mlsc_loss = MyoPSLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6, last_epoch=-1)

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    Train_Image = CrossModalDataLoader(path=args.path, file_name='train.txt', dim=args.dim, max_iters=100*args.batch_size, stage='Train')
    Train_loader = cycle(DataLoader(Train_Image, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True))
  
    Valid_Image = CrossModalDataLoader(path=args.path, file_name='validation.txt', dim=args.dim, max_iters=None, stage='Valid')
    Valid_loader = cycle(DataLoader(Valid_Image, batch_size=1, shuffle=False, num_workers=0, drop_last=False))

    writer = SummaryWriter()

    for epoch in range(args.start_epoch, args.end_epoch):

        # Train
        model.train()

        train_C0 = torch.FloatTensor(args.batch_size, 1, args.dim, args.dim).cuda()
        train_LGE = torch.FloatTensor(args.batch_size, 1, args.dim, args.dim).cuda()
        train_T2 = torch.FloatTensor(args.batch_size, 1, args.dim, args.dim).cuda()
        train_T1m = torch.FloatTensor(args.batch_size, 1, args.dim, args.dim).cuda()
        train_T2starm = torch.FloatTensor(args.batch_size, 1, args.dim, args.dim).cuda()

        cardiac_gd = torch.FloatTensor(args.batch_size, 3, args.dim, args.dim).cuda()
        scar_gd = torch.FloatTensor(args.batch_size, 3, args.dim, args.dim).cuda()
        edema_gd = torch.FloatTensor(args.batch_size, 3, args.dim, args.dim).cuda()

        IterCount = int(len(Train_Image)/args.batch_size)

        for iteration in range(IterCount):
            
            # Sup
            img_C0, img_LGE, img_T2, img_T1m, img_T2starm, label_cardiac, label_scar, label_edema, _ = next(Train_loader) 

            train_C0.copy_(img_C0)
            train_LGE.copy_(img_LGE)
            train_T2.copy_(img_T2)
            train_T1m.copy_(img_T1m)
            train_T2starm.copy_(img_T2starm)

            cardiac_gd.copy_(label_cardiac)
            scar_gd.copy_(label_scar)
            edema_gd.copy_(label_edema)

            seg_C0, seg_LGE, seg_T2, seg_mapping = model(train_C0, train_LGE, train_T2, train_T1m, train_T2starm)

            seg = {'C0': seg_C0, 'LGE': seg_LGE, 'T2': seg_T2, 'mapping': seg_mapping}
            label = {'cardiac': cardiac_gd, 'scar': scar_gd, 'edema': edema_gd}

            loss_seg, loss_invariant, loss_inclusive, loss = mlsc_loss(seg, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # write to log
            with open('log_training.txt', 'a') as segment_log:
                segment_log.write("==> Epoch: {:0>3d}/{:0>3d} || ".format(epoch + 1, args.end_epoch))
                segment_log.write("Iteration: {:0>3d}/{:0>3d} - ".format(iteration + 1, IterCount))
                segment_log.write("LR: {:.6f} | ".format(float(optimizer.param_groups[0]['lr'])))
                segment_log.write("loss_seg: {:.6f} + ".format(loss_seg.detach().cpu()))
                segment_log.write("loss_invariant: {:.6f} + ".format(loss_invariant.detach().cpu()))
                segment_log.write("loss_inclusive: {:.6f} + ".format(loss_inclusive.detach().cpu()))
                segment_log.write("loss: {:.6f}\n".format(loss.detach().cpu()))
                
            # write to tensorboard
            writer.add_scalar('seg loss', loss_seg.detach().cpu(), epoch * (IterCount + 1) + iteration)
            writer.add_scalar('invariant loss', loss_invariant.detach().cpu(), epoch * (IterCount + 1) + iteration)
            writer.add_scalar('inclusive loss', loss_inclusive.detach().cpu(), epoch * (IterCount + 1) + iteration)
            writer.add_scalar('total loss', loss.detach().cpu(), epoch * (IterCount + 1) + iteration)

        lr_scheduler.step()

        # Validation
        model.eval()
        avg_dice_2d = Validation2d(args, epoch, model, Valid_Image, Valid_loader, writer, 'result_validation_2d.txt', tensorboardImage=True)

        if avg_dice_2d > args.threshold:
            torch.save(model.state_dict(), os.path.join('checkpoints', str(avg_dice_2d) + '['+ str(epoch+1) + '].pth'))
