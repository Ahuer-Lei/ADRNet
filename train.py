#!/usr/bin/python3
from torch.utils.data import DataLoader
from dataset import TrainData, TestData, img_save
from options import TrainOptions
from model import HR
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

        self.model = HR(self.config).cuda()
        self.model.set_scheduler(self.config, now_ep=-1) 
        self.saver = Saver(self.config)

        traindataset = TrainData(config)
        self.train_loader = DataLoader(traindataset, batch_size=config.batch_size, shuffle=True, num_workers=config.nThreads)

        testdataset = TestData(config)
        self.test_loader = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=config.nThreads)

    def train_and_test(self):
        lowest_sar_self_loss = 50
        lowest_opt_self_loss = 50
 
        for ep in range(self.config.n_ep):
            self.model.train()
            total_tp_loss = total_disp_loss = 0
            start = time()
            p_train_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for it, [opt, sar, opt_warp, sar_warp, gt_tp, gt_disp] in p_train_bar:

                image_sar = sar.cuda()      # [8,1,256,256] 
                image_opt = opt.cuda()
                image_sar_warp = sar_warp.cuda()
                image_opt_warp = opt_warp.cuda()
                gt_tp = gt_tp.squeeze(1).cuda()         #
                gt_disp = gt_disp.squeeze(1).cuda()

                self.model.update_RF(image_opt, image_sar, image_opt_warp, image_sar_warp, gt_tp, gt_disp)

                total_tp_loss = total_tp_loss + self.model.loss_tp
                total_disp_loss = total_disp_loss + self.model.loss_disp
                end = time()
                p_train_bar.set_description(f'training ep: %d | time : {str(round(end - start, 0))}' % ep)

            avg_tp = total_tp_loss / len(self.train_loader)
            avg_disp = total_disp_loss / len(self.train_loader)
           
            train_loss_record(self.config, ep=ep, tp_loss=avg_tp, disp_loss=avg_disp, lr=self.model.RES_opt.param_groups[0]['lr'])
            
        
            self.saver.write_img(ep=ep, model=self.model)

   
            self.model.eval()
  
            total_sarf_self_loss = 0
            total_optf_self_loss = 0
            total_s2o_disp_loss = 0
            total_o2s_disp_loss = 0

            start = time()
            p_test_bar = tqdm(enumerate(self.test_loader), total=len(self.test_loader))
            for it, [opt_, sar_, opt_warp_, sar_warp_, gt_tp_, gt_disp_, name] in p_test_bar:
                image_sar_ = sar_.cuda()    # [1,1,256,256]
                image_opt_ = opt_.cuda()
                image_sar_warp_ = sar_warp_.cuda()
                image_opt_warp_ = opt_warp_.cuda() 
                gt_tp_ = gt_tp_.squeeze(1).cuda()      # [1,2,3]
                gt_disp_ = gt_disp_.squeeze(1).cuda()  # [1,256,256,2]
                with torch.no_grad():
                    image_sar_reg, image_opt_reg, sarf_self_loss, optf_self_loss, loss_s2o_disp, loss_o2s_disp = self.model.test_forward(image_opt_, image_sar_, image_opt_warp_, image_sar_warp_, gt_tp_, gt_disp_)


                total_sarf_self_loss = total_sarf_self_loss + sarf_self_loss
                total_optf_self_loss = total_optf_self_loss + optf_self_loss
                total_s2o_disp_loss = total_s2o_disp_loss + loss_s2o_disp
                total_o2s_disp_loss = total_o2s_disp_loss + loss_o2s_disp

                end = time()
                p_test_bar.set_description(f'testing  ep: %d | time : {str(round(end - start, 2))}' % ep)
         

                avg_s2o_disp = (total_s2o_disp_loss / len(self.test_loader)).sqrt()
                avg_o2s_disp = (total_o2s_disp_loss / len(self.test_loader)).sqrt()
                avg_sarf_self = total_sarf_self_loss / len(self.test_loader)
                avg_optf_self = total_optf_self_loss / len(self.test_loader)
   
                test_loss_record(self.config, ep=ep, sarf_self=avg_sarf_self, optf_self=avg_optf_self, s2o_disp_loss=avg_s2o_disp, o2s_disp_loss=avg_o2s_disp)

                if lowest_sar_self_loss > avg_sarf_self and lowest_opt_self_loss > avg_optf_self:
                    lowest_sar_self_loss = avg_sarf_self
                    lowest_opt_self_loss = avg_optf_self

                    result= {'sarf_self':str(avg_sarf_self.cpu().numpy()), 'optf_self':str(avg_optf_self.cpu().numpy()),'s2o_disp_error':str(avg_s2o_disp.cpu().numpy()), 'o2s_disp_error':str(avg_o2s_disp.cpu().numpy()), 'detail':'acm optsar，动态lr', 'loss':'10_1_1'}
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


def test_loss_record(config, ep, sarf_self, optf_self, s2o_disp_loss, o2s_disp_loss):
    os.makedirs(config.test_logs_dir, exist_ok=True)
    log_path = os.path.join(config.test_logs_dir, 'test_log.txt')
    with open(log_path, 'a') as f:
        f.write('No.%d Epoch: sarf_self:%.4f | optf_self:%.4f | s2o_disp_loss:%.4f | o2s_disp_loss:%.4f\n' % (ep, sarf_self, optf_self, s2o_disp_loss, o2s_disp_loss))


