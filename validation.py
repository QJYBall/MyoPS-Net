import torch
from torchvision.utils import make_grid
from process import LargestConnectedComponents
from criterion.metrics import Criterion, DiceMeter, Logger2d
from utils.tools import Normalization, ResultTransform


@torch.no_grad()
def Validation2d(args, epoch, model, valid_image, valid_loader, writer, log_name, tensorboardImage):
    
    logger = Logger2d()
    criterion = Criterion()
    normalize = Normalization()
    keepLCC = LargestConnectedComponents()
    result_transform = ResultTransform(ToOriginal=False)

    valid_C0 = DiceMeter()
    valid_LGE = DiceMeter()
    valid_T2 = DiceMeter()
    valid_mapping = DiceMeter()
    valid_pathology = DiceMeter()

    test_C0 = torch.FloatTensor(1, 1, args.dim, args.dim).cuda()
    test_LGE = torch.FloatTensor(1, 1, args.dim, args.dim).cuda() 
    test_T2 = torch.FloatTensor(1, 1, args.dim, args.dim).cuda() 
    test_T1m = torch.FloatTensor(1, 1, args.dim, args.dim).cuda() 
    test_T2starm = torch.FloatTensor(1, 1, args.dim, args.dim).cuda() 

    cardiac_gd = torch.FloatTensor(args.dim, args.dim)
    scar_gd = torch.FloatTensor(args.dim, args.dim)
    edema_gd = torch.FloatTensor(args.dim, args.dim)
    pathology_gd = torch.FloatTensor(args.dim, args.dim)

    if tensorboardImage == True:
        pic = torch.FloatTensor(10*int(len(valid_image)), 1, args.dim, args.dim).cuda()

    for iter in range(int(len(valid_image))):

        C0_Image, LGE_Image, T2_Image, T1m_Image, T2starm_Image, cardiac_label, scar_label, edema_label, pathology_label = next(valid_loader)

        test_C0.copy_(C0_Image)
        test_LGE.copy_(LGE_Image)
        test_T2.copy_(T2_Image)
        test_T1m.copy_(T1m_Image)
        test_T2starm.copy_(T2starm_Image)
        
        res_C0, res_LGE, res_T2, res_mapping = model(test_C0, test_LGE, test_T2, test_T1m, test_T2starm)

        seg_C0 = torch.argmax(res_C0, dim=1).squeeze(0)
        seg_LGE = torch.argmax(res_LGE, dim=1).squeeze(0)
        seg_T2 = torch.argmax(res_T2, dim=1).squeeze(0)
        seg_mapping = torch.argmax(res_mapping, dim=1).squeeze(0)

        seg_C0 = keepLCC(seg_C0.cpu(), 'cardiac')
        seg_LGE = keepLCC(seg_LGE.cpu(), 'scar')
        seg_T2 = keepLCC(seg_T2.cpu(), 'edema')
        seg_mapping = keepLCC(seg_mapping.cpu(), 'scar')

        seg_pathology = result_transform(seg_LGE, seg_mapping, seg_T2)

        cardiac_gd.copy_(cardiac_label[0,0,...])
        scar_gd.copy_(scar_label[0,0,...])
        edema_gd.copy_(edema_label[0,0,...])
        pathology_gd.copy_(pathology_label[0,0,...])

        myo_C0, lv_C0 = criterion(seg_C0, cardiac_gd, 'cardiac')
        myo_LGE, scar_LGE = criterion(seg_LGE, scar_gd, 'scar')
        myo_T2, edema_T2 = criterion(seg_T2, edema_gd, 'edema')
        myo_mapping, scar_mapping = criterion(seg_mapping, scar_gd, 'scar')
        scar_pathology, edema_pathology = criterion(seg_pathology, pathology_gd, 'pathology')

        valid_C0.update(myo_C0, lv_C0, 0, 0)
        valid_LGE.update(myo_LGE, 0, scar_LGE, 0)
        valid_T2.update(myo_T2, 0, 0, edema_T2)
        valid_mapping.update(myo_mapping, 0, scar_mapping, 0)
        valid_pathology.update(0, 0, scar_pathology, edema_pathology)

        if tensorboardImage == True:
            pic[10*iter:10*iter+10,...].copy_(torch.cat([
                normalize(cardiac_gd, 'Max_Min').unsqueeze(0).unsqueeze(0),
                normalize(seg_C0, 'Max_Min').unsqueeze(0).unsqueeze(0),
                normalize(scar_gd, 'Max_Min').unsqueeze(0).unsqueeze(0),
                normalize(seg_LGE, 'Max_Min').unsqueeze(0).unsqueeze(0),
                normalize(scar_gd, 'Max_Min').unsqueeze(0).unsqueeze(0),
                normalize(seg_mapping, 'Max_Min').unsqueeze(0).unsqueeze(0),
                normalize(edema_gd, 'Max_Min').unsqueeze(0).unsqueeze(0),
                normalize(seg_T2, 'Max_Min').unsqueeze(0).unsqueeze(0),
                normalize(pathology_gd, 'Max_Min').unsqueeze(0).unsqueeze(0),
                normalize(seg_pathology, 'Max_Min').unsqueeze(0).unsqueeze(0)], dim=0))
    
    if tensorboardImage == True:
        writer.add_image('pic', make_grid(pic, nrow=10, padding=2), epoch)  

    C0_dice = {'myo': valid_C0.myo['avg'], 'lv': valid_C0.lv['avg']}
    LGE_dice = {'myo': valid_LGE.myo['avg'], 'scar': valid_LGE.scar['avg']}
    T2_dice = {'myo': valid_T2.myo['avg'], 'edema': valid_T2.edema['avg']}
    mapping_dice = {'myo': valid_mapping.myo['avg'], 'scar': valid_mapping.scar['avg']}
    pathology_dice = {'scar': valid_pathology.scar['avg'], 'edema': valid_pathology.edema['avg']}

    logger(epoch+1, args.end_epoch, log_name, C0_dice, LGE_dice, T2_dice, mapping_dice, pathology_dice)

    writer.add_scalar('valid scar 2d', pathology_dice['scar'], epoch)
    writer.add_scalar('valid edema 2d', pathology_dice['edema'], epoch)

    avg_dice = (pathology_dice['scar'] + pathology_dice['edema']) / 2

    return avg_dice
