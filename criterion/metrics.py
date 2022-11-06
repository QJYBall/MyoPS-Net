class Criterion(object):
    def calc_Dice(self, output, target):   
        num = (output * target).sum()
        den = output.sum() + target.sum()
        dice = 2 * num / (den + 1e-8)  
        return dice

    def __call__(self, output, target, mode):

        output = output.numpy()
        target = target.numpy()

        # 0-0-bg, 1-200-myo, 2-500-lv
        if mode == 'cardiac':
            myo_dice = self.calc_Dice((output==1), (target==1)) # myo
            lv_dice = self.calc_Dice((output==2), (target==2)) # lv
            return myo_dice, lv_dice
    
        # 0-0-bg, 1-200-myo-scar, 2-2221-scar
        if mode == 'scar':
            myo_dice = self.calc_Dice(((output==1)|(output==2)), ((target==1)|(target==2))) # myo
            scar_dice = self.calc_Dice((output==2), (target==2)) # scar
            return myo_dice, scar_dice

        # 0-0-bg, 1-200-myo-edema, 2-1220-edema
        if mode == 'edema':
            myo_dice = self.calc_Dice(((output==1)|(output==2)), ((target==1)|(target==2))) # myo
            edema_dice = self.calc_Dice((output==2), (target==2)) # edema
            return myo_dice, edema_dice

        # 0-0-bg, 1-1220-edema, 2-2221-scar
        if mode == 'pathology':
            edema_dice = self.calc_Dice(((output==1)|(output==2)), ((target==1)|(target==2))) # edema
            scar_dice = self.calc_Dice((output==2), (target==2)) # scar
            return scar_dice, edema_dice


class DiceMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.myo = {'sum':0.0, 'avg':0.0}
        self.lv = {'sum':0.0, 'avg':0.0}
        self.scar = {'sum':0.0, 'avg':0.0}
        self.edema = {'sum':0.0, 'avg':0.0}
        self.count = 0

    def update(self, myo_dice, lv_dice, scar_dice, edema_dice):
        self.count += 1
        self.myo['sum'] += myo_dice
        self.myo['avg']= self.myo['sum'] / self.count
        self.lv['sum'] += lv_dice
        self.lv['avg']= self.lv['sum'] / self.count
        self.scar['sum'] += scar_dice
        self.scar['avg']= self.scar['sum'] / self.count
        self.edema['sum'] += edema_dice
        self.edema['avg'] = self.edema['sum'] / self.count


class Logger2d(object):
    def __call__(self, epoch, total_epoch, file_name, C0_dice, LGE_dice, T2_dice, mapping_dice, pathology_dice):   
        with open(file_name, 'a') as f:
            f.write("=> Epoch: {:0>3d}/{:0>3d} || ".format(epoch, total_epoch))
            f.write("C0(myo,lv): {:.4f} - {:.4f} + ".format(C0_dice['myo'], C0_dice['lv']))
            f.write("LGE(myo,scar): {:.4f} - {:.4f} + ".format(LGE_dice['myo'], LGE_dice['scar']))
            f.write("T2(myo,edema): {:.4f} - {:.4f} + ".format(T2_dice['myo'], T2_dice['edema']))
            f.write("mapping(myo,scar): {:.4f} - {:.4f} + ".format(mapping_dice['myo'], mapping_dice['scar']))
            f.write("Combine(scar,edema): {:.4f} - {:.4f} + ".format(pathology_dice['scar'], pathology_dice['edema']))
            f.write("Avg: {:.4f}\n".format((pathology_dice['scar'] + pathology_dice['edema'])/2))
