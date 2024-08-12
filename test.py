import torch
from torch.utils.data import DataLoader
from options import TrainOptions
from dataset import InferData, img_save
from model import HR
from utils.saver import Saver
from time import time
from tqdm import tqdm
from modules.losses import *
from build_table import *
import torch.nn as nn
# from thop import profile


def test(opts):

    model = HR(opts)
    model.cuda()
    resweight = torch.load('/data/zcl/work2/ADRNet_ACM/ADRNet_optsar/model_save/optsar/optsaring_159.pth')['RES']
    unweight = torch.load('/data/zcl/work2/ADRNet_ACM/ADRNet_optsar/model_save/optsar/optsaring_159.pth')['UN']
    model.RES.load_state_dict(resweight)
    model.UN.load_state_dict(unweight)


    test_dataloader = InferData(opts)
    loader = DataLoader(test_dataloader, batch_size=1, shuffle=False)
    p_bar = tqdm(enumerate(loader), total=len(loader))
    model.eval() 
    start = time()
    total_loss = 0
    l1loss = nn.L1Loss()
    for idx, [opt, sar, opt_warp, sar_warp, gt_disp, name] in p_bar:
     
        vi_tensor = opt.squeeze(0).cuda()
        ir_tensor = sar.squeeze(0).cuda()
        vi_warp_tensor = opt_warp.squeeze(0).cuda()
        ir_warp_tensor = sar_warp.squeeze(0).cuda()
        gt_disp = gt_disp.squeeze(0).cuda()

        b,c,h,w = vi_tensor.shape
    
        with torch.no_grad():
            
            sar_reg_aff, flow = model.test_forward(vi_tensor, ir_tensor, vi_warp_tensor, ir_warp_tensor)

            gt = F.grid_sample(ir_warp_tensor, gt_disp, mode='bilinear', padding_mode='zeros', align_corners=True)
            loss = torch.sum(abs(flow-gt_disp).pow(2))/(h*w)
            # loss = l1loss(sar_reg_aff*255, gt*255)
            total_loss = total_loss + loss
            # img_save(sar_reg_aff, os.path.join("/data/zcl/work2/ADRNet_ACM/ADRNet_optsar/results/reg", name[0]+'.jpg'))
      
    print((total_loss/335).sqrt())
   



if __name__ == '__main__':
    parser = TrainOptions()   # 加载选项信息
    opt = parser.parse()
    print('\n--- options load success ---')
    test(opts=opt)
