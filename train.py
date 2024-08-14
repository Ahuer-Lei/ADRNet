#!/usr/bin/python3
from torch.utils.data import DataLoader
from dataset import TrainData, TestData, img_save
from options import TrainOptions
from model import ADRNet
from utils.saver import Saver
from time import time
from tqdm import tqdm
from modules.losses import *
from build_table import *
from torch.utils.data import DataLoader


class Train_and_test():
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = ADRNet(self.config).cuda()
        self.model.set_scheduler(self.config, now_ep=-1) 
        self.saver = Saver(self.config)

        traindataset = TrainData(config)
        self.train_loader = DataLoader(traindataset, batch_size=config.batch_size, shuffle=True, num_workers=config.nThreads)

        testdataset = TestData(config)
        self.test_loader = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=config.nThreads)

    def train_and_test(self):
        lowest_re = 50
  
        for ep in range(self.config.n_ep):
            self.model.train()
            total_tp_loss = total_disp_loss = 0
            start = time()
            p_train_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for it, [rgb, sar, rgb_warp, sar_warp, gt_tp, gt_disp] in p_train_bar:

                image_sar = sar.cuda()      # [8,1,256,256] 
                image_rgb = rgb.cuda()
                image_sar_warp = sar_warp.cuda()
                image_rgb_warp = rgb_warp.cuda()
                gt_tp = gt_tp.squeeze(1).cuda()         #
                gt_disp = gt_disp.squeeze(1).cuda()

                self.model.update_RF(image_rgb, image_sar, image_rgb_warp, image_sar_warp, gt_tp, gt_disp)

                total_tp_loss = total_tp_loss + self.model.loss_tp
                total_disp_loss = total_disp_loss + self.model.loss_disp
                end = time()
                p_train_bar.set_description(f'training ep: %d | time : {str(round(end - start, 0))}' % ep)

            avg_tp = total_tp_loss / len(self.train_loader)
            avg_disp = total_disp_loss / len(self.train_loader)
           
            train_loss_record(self.config, ep=ep, tp_loss=avg_tp, disp_loss=avg_disp, lr=self.model.RES_opt.param_groups[0]['lr'])
            
        
            self.saver.write_img(ep=ep, model=self.model)

            if ep % 10 ==0 or ep > 250:
                self.model.eval()
  
                total_re_loss = 0
                total_cor_rmse_loss = 0
                total_disp_rmse_loss = 0
 
                start = time()
                p_test_bar = tqdm(enumerate(self.test_loader), total=len(self.test_loader))
                for it, [rgb_, sar_, rgb_warp_, sar_warp_, gt_tp_, gt_disp_] in p_test_bar:
                    image_sar_ = sar_.cuda()    # [1,1,256,256]
                    image_rgb_ = rgb_.cuda()
                    image_sar_warp_ = sar_warp_.cuda()
                    image_rgb_warp_ = rgb_warp_.cuda() 
                    gt_tp_ = gt_tp_.squeeze(1).cuda()      # [1,2,3]
                    gt_disp_ = gt_disp_.squeeze(1).cuda()  # [1,256,256,2]
                    with torch.no_grad():
                        image_sar_reg, cor_rmse, disp_rmse, sar_re = self.model.test_forward(image_rgb_, image_sar_, image_rgb_warp_, image_sar_warp_, gt_tp_, gt_disp_)

                    total_re_loss = total_re_loss + sar_re
                    total_cor_rmse_loss = total_cor_rmse_loss + cor_rmse
                    total_disp_rmse_loss = total_disp_rmse_loss + disp_rmse

                end = time()
                p_test_bar.set_description(f'testing  ep: %d | time : {str(round(end - start, 2))}' % ep)
         

                avg_disp = (total_disp_rmse_loss / len(self.test_loader)).sqrt()
                avg_cor = (total_cor_rmse_loss / len(self.test_loader))
                avg_re = total_re_loss / len(self.test_loader)
              
   
                test_loss_record(self.config, ep=ep, re=avg_re, cor_rmse=avg_cor, disp_rmse=avg_disp)

                if lowest_re > avg_re:
                    result= {'re':str(avg_re.cpu().numpy()), 'cor_rmse':str(avg_cor.cpu().numpy()),'avg_disp':str(avg_disp.cpu().numpy())}
                    self.saver.save_model(self.config, ep=ep, model=self.model, mode='ing')

            if self.config.n_ep_decay > -1:     # 从那个epoch开始减少学习率
                self.model.update_lr()
        method_dict={'method':'opt_sar','id':0,'batch_size':self.config.batch_size,'epoch':self.config.n_ep}
        set_result(result=result,method_dict=method_dict)


    



def train_loss_record(config, ep, tp_loss, disp_loss, lr):
    os.makedirs(config.train_logs_dir, exist_ok=True)
    log_path = os.path.join(config.train_logs_dir, 'train_log.txt')
    with open(log_path, 'a') as f:
        f.write('No.%d Epoch: tp_loss:%.6f | disp_loss:%.6f | lr:%.6f \n' % (ep, tp_loss, disp_loss, lr))


def test_loss_record(config, ep, re, cor_rmse, disp_rmse):
    os.makedirs(config.test_logs_dir, exist_ok=True)
    log_path = os.path.join(config.test_logs_dir, 'test_log.txt')
    with open(log_path, 'a') as f:
        f.write('No.%d Epoch: re:%.4f | cor:%.4f | disp:%.4f \n' % (ep, re, cor_rmse, disp_rmse))


