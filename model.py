from modules.losses import *
import kornia
import kornia.utils as KU
from modules.modules import resnet, unet, get_scheduler, gaussian_weights_init, SpatialTransformer
import cv2


def four_point_RMSE_loss(tp_pre, tp_gt):
    m = np.array([[0, 0, 1]])
    tp_pre = np.array(tp_pre.cpu()).reshape(2, 3)
    tp_gt = np.array(tp_gt.cpu()).reshape(2, 3)

    matrix_pre = np.concatenate((tp_pre, m))
    matrix_gt = np.concatenate((tp_gt, m))

    T = np.array([[2 / 256, 0, -1],
                  [0, 2 / 256, -1],
                  [0, 0, 1]])

    matrix_pre_tran = np.linalg.inv(T) @ np.linalg.inv(matrix_pre) @ T
    matrix_gt_tran = np.linalg.inv(T) @ np.linalg.inv(matrix_gt) @ T
    four_corners = np.array([[0, 0], [0, 255], [255, 0], [255, 255]], dtype=np.float32).reshape(-1, 1, 2)
    four_point_pre = cv2.perspectiveTransform(four_corners, matrix_pre_tran)
    four_point_gt = cv2.perspectiveTransform(four_corners, matrix_gt_tran)
    rmse = np.sum(np.sqrt(np.sum((four_point_gt - four_point_pre) ** 2, axis=2)))/4

    return rmse

def generate_mask(img_warp):
    mask = torch.ge(img_warp.squeeze(1), 1)
    mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(1)
    num = mask.view(img_warp.size(0), img_warp.size(1), -1).sum(dim=-1)
    num = num.reshape(1, -1)
    return mask, num

def affine_to_flow(tp, b):
    tp = tp.reshape(-1, 2, 3)
    a = torch.Tensor([[[0, 0, 1]]]).cuda().repeat(b, 1, 1)
    tp = torch.cat((tp, a), dim=1)
    grid = KU.create_meshgrid(256, 256).cuda().repeat(b, 1, 1, 1)
    flow = kornia.geometry.linalg.transform_points(tp, grid)
    return flow, flow-grid


def normalize_image(x):
    return x[:, 0:1, :, :]


def border_suppression(img, mask):
    return (img * (1 - mask)).mean()


def STN(img, pre_tps):
    aff_mat = pre_tps.reshape(-1, 2, 3)
    img_grid = F.affine_grid(aff_mat, img.size())
    img_reg = F.grid_sample(img, img_grid)
    return img_reg



class HR(nn.Module):
    def __init__(self, config=None):
        super(HR, self).__init__()

        lr = 0.0001

        self.RES = resnet()
        self.ST = SpatialTransformer(256, 256, True)
        self.UN = unet()

        self.RES_opt = torch.optim.Adam(self.RES.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.00001)
        self.UN_opt = torch.optim.Adam(self.UN.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.00001)

        self.gradient_loss = gradient_loss()
        self.ncc_loss = ncc_loss()
        self.mi_loss = mi_loss()
        self.l1_loss = nn.L1Loss()
        self.deformation_1 = {}
        self.deformation_2 = {}
        self.border_mask = torch.zeros([1, 1, 256, 256])
        self.border_mask[:, :, 10:-10, 10:-10] = 1
        self.AP = nn.AvgPool2d(5, stride=1, padding=2)
        self.initialize()

    def initialize(self):
        self.RES.apply(gaussian_weights_init)
        self.UN.apply(gaussian_weights_init)

    def set_scheduler(self, opts, now_ep=0):
        self.RES_sch = get_scheduler(self.RES_opt, opts, now_ep)
        self.UN_sch = get_scheduler(self.UN_opt, opts, now_ep)


    def test_forward(self, opt, sar, opt_warp, sar_warp):
        # opt_gt_reg = STN(opt_warp, gt_tp)
        # sar_gt_reg = STN(sar_warp, gt_tp)

        b, c, h, w = sar.shape
        sar_stack = torch.cat([sar_warp, sar])
        opt_stack = torch.cat([opt, opt_warp])
        sw2o, _, _, ow2s = self.RES(sar_stack, opt_stack)

        _, sw2o_disp = affine_to_flow(sw2o, b)
        _, ow2s_disp = affine_to_flow(ow2s, b)

        opt_reg_aff = STN(opt_warp, ow2s)
        sar_reg_aff = STN(sar_warp, sw2o)

        sar_stack_ = torch.cat([sar_reg_aff, sar])
        opt_stack_ = torch.cat([opt, opt_reg_aff])
        _, _, disp, _ = self.UN(sar_stack_, opt_stack_)

        pre_disp1 = sw2o_disp + disp['sar2opt'].permute(0,2,3,1)
        pre_disp2 = ow2s_disp + disp['opt2sar'].permute(0,2,3,1)

        img_stack = torch.cat([sar_warp, opt_warp])
        disp_stack = torch.cat([pre_disp1, pre_disp2])
        img_reg, flow = self.ST(img_stack, disp_stack)

        image_sar_reg, image_opt_reg = torch.split(img_reg, b, dim=0)

        # mask_sar_reg, num_sar = generate_mask(image_sar_reg*255)
        # mask_sar_areg, anum_sar = generate_mask(sar_reg_aff*255)

        # loss_pts = four_point_RMSE_loss(sw2o, gt_tp)  # 每个角的位移差

        # loss_s2o_disp = torch.sum(abs(pre_disp1-gt_disp).pow(2))/(h*w) # 每个像素的位移差
        # loss_o2s_disp = torch.sum(abs(pre_disp2-gt_disp).pow(2))/(h*w)

        # sarf_self_loss = self.l1_loss(sar_gt_reg*255, image_sar_reg*255)
        # optf_self_loss = self.l1_loss(opt_gt_reg*255, image_opt_reg*255)


        # sarf_self_loss = torch.sum(abs(sar*mask_sar_reg*255-image_sar_reg*255)) / num_sar        
        # sara_self_loss = torch.sum(abs(sar*mask_sar_areg*255-sar_reg_aff*255)) / anum_sar

        return image_sar_reg,  flow[0:1,...]


    def forward(self, vi_tensor, ir_tensor, vi_warp_tensor, ir_warp_tensor):
        b, c, h, w = vi_tensor.shape
        sar_stack = torch.cat([ir_warp_tensor, ir_tensor])
        opt_stack = torch.cat([vi_tensor, vi_warp_tensor])
        sw2o, _, _, ow2s = self.RES(sar_stack, opt_stack)

        # _, sw2o_disp = affine_to_flow(sw2o, b)
        # _, ow2s_disp = affine_to_flow(ow2s, b)

        opt_reg_aff = STN(vi_warp_tensor, ow2s)
        sar_reg_aff = STN(ir_warp_tensor, sw2o)

        # sar_stack_ = torch.cat([sar_reg_aff, ir_tensor])
        # opt_stack_ = torch.cat([vi_tensor, opt_reg_aff])
        # _, _, disp, _ = self.UN(sar_stack_, opt_stack_)

        # pre_disp1 = sw2o_disp + disp['sar2opt'].permute(0,2,3,1)
        # pre_disp2 = ow2s_disp + disp['opt2sar'].permute(0,2,3,1)

        # img_stack = torch.cat([ir_warp_tensor, vi_warp_tensor])
        # disp_stack = torch.cat([pre_disp1, pre_disp2])
        # img_reg = self.ST(img_stack, disp_stack)

        # image_sar_reg, image_opt_reg = torch.split(img_reg, b, dim=0)

        return sar_reg_aff

    def train_forward(self):
        b = self.image_sar_warp.shape[0]

        sar_stack = torch.cat([self.image_sar_warp, self.image_sar])
        opt_stack = torch.cat([self.image_opt, self.image_opt_warp])
        
        self.sw2o, self.s2ow, self.o2sw, self.ow2s = self.RES(sar_stack, opt_stack)

        _, self.sw2o_disp = affine_to_flow(self.sw2o, b)
        _, self.ow2s_disp = affine_to_flow(self.ow2s, b)    # [8,256,256,2]

        self.mask, _ = generate_mask(self.image_sar_warp*255)
        self.mask_true = STN(self.mask, self.gt_tp)


        self.opt_reg_aff = STN(self.image_opt_warp, self.ow2s)
        self.sar_reg_aff = STN(self.image_sar_warp, self.sw2o)
     
        sar_stack_ = torch.cat([self.sar_reg_aff, self.image_sar])
        opt_stack_ = torch.cat([self.image_opt, self.opt_reg_aff])

        self.u, self.v, self.disp, self.disp1 = self.UN(sar_stack_, opt_stack_)  # [8,2,256,256]

        self.pre_disp_sw2o = self.sw2o_disp + self.disp['sar2opt'].permute(0,2,3,1)
        self.pre_disp_ow2s = self.ow2s_disp + self.disp['opt2sar'].permute(0,2,3,1)

        img_stack = torch.cat([self.image_sar_warp, self.image_opt_warp])
        disp_stack = torch.cat([self.pre_disp_sw2o, self.pre_disp_ow2s])
        img_reg_stack = self.ST(img_stack, disp_stack)

        self.image_sar_reg, self.image_opt_reg = torch.split(img_reg_stack, b, dim=0)


    def update_RF(self, image_opt, image_sar, image_opt_warp, image_sar_warp, gt_tp, gt_disp):
        self.image_opt = image_opt
        self.image_sar = image_sar
        self.image_opt_warp = image_opt_warp
        self.image_sar_warp = image_sar_warp
        self.gt_tp = gt_tp
        self.gt_disp = gt_disp

        self.RES_opt.zero_grad()
        self.UN_opt.zero_grad()

        self.train_forward()

        self.backward_RF()

        nn.utils.clip_grad_norm_(self.RES.parameters(), 5)
        nn.utils.clip_grad_norm_(self.UN.parameters(), 5)

        self.RES_opt.step()
        self.UN_opt.step()

    def img_loss(self, src, tgt, mask=1, weights=None):
        if weights is None:
            weights = [0.1, 0.9]
        return weights[0] * (l1loss(src, tgt, mask) + l2loss(src, tgt, mask)) + weights[1] * self.gradient_loss(src, tgt,
                                                                                                               mask)

    def weight_filed_loss(self, ref, tgt, disp, disp_gt):
        ref = (ref - ref.mean(dim=[-1, -2], keepdim=True)) / (ref.std(dim=[-1, -2], keepdim=True) + 1e-5)
        tgt = (tgt - tgt.mean(dim=[-1, -2], keepdim=True)) / (tgt.std(dim=[-1, -2], keepdim=True) + 1e-5)
        g_ref = KF.spatial_gradient(ref, order=2).mean(dim=1).abs().sum(dim=1).detach().unsqueeze(1)
        g_tgt = KF.spatial_gradient(tgt, order=2).mean(dim=1).abs().sum(dim=1).detach().unsqueeze(1)
        w = ((g_ref + g_tgt) * 2 + 1) * self.border_mask.to(device)
        return (w * (1000 * (disp.permute(0,3,1,2) - disp_gt.permute(0,3,1,2)).abs().clamp(min=1e-2).pow(2))).mean()


    def sym_loss(self, disp1, disp2):
        grid = KU.create_meshgrid(256, 256).cuda()
        flow1 = grid + disp1
        flow2 = grid + disp2
        a = F.grid_sample(flow1.permute(0,3,1,2), flow2).permute(0,2,3,1)
        mask = torch.eq(a, 0)
        mask = torch.tensor(~mask, dtype=torch.float32)
        num = mask.permute(0,3,1,2).view(mask.permute(0,3,1,2).size(0), mask.permute(0,3,1,2).size(1) , -1).sum(dim=-1).sum(dim=-1)/2
        grid1 = grid*mask

        loss = (torch.sum(abs(a-grid1).pow(2), dim=[-1,-2,-3])/num).mean().sqrt()
        return loss


    def backward_RF(self):
        b, c, h, w = self.image_opt.shape
    
        a = torch.Tensor([[[0,0,1]]]).cuda().repeat(b, 1, 1)
        unitmat = torch.Tensor([[[1,0,0],[0,1,0],[0,0,1]]]).cuda().repeat(b, 1, 1)

        sw2o = self.sw2o.reshape(-1, 2, 3)
        sw2o_mat = torch.cat((sw2o, a), dim=1)
        o2sw = self.o2sw.reshape(-1, 2, 3)
        o2sw_mat = torch.cat((o2sw, a), dim=1)  
        e1 = torch.matmul(sw2o_mat, o2sw_mat)

        ow2s = self.ow2s.reshape(-1, 2, 3)
        ow2s_mat = torch.cat((ow2s, a), dim=1)    
        s2ow = self.s2ow.reshape(-1, 2, 3)
        s2ow_mat = torch.cat((s2ow, a), dim=1)
        e2 = torch.matmul(ow2s_mat, s2ow_mat)
                
        dc1 = torch.sum(abs(e1-unitmat), dim=[-2,-1]).mean() + torch.sum(abs(e2-unitmat),dim=[-2,-1]).mean()
  
        ld_loss = self.img_loss(self.image_opt, self.image_opt_reg, self.mask_true) + \
                  self.img_loss(self.image_sar, self.image_sar_reg, self.mask_true)
 
        loss_tp = torch.sum(abs(self.sw2o-self.gt_tp.reshape(b, -1))).mean() + torch.sum(abs(self.ow2s-self.gt_tp.reshape(b, -1))).mean()
 
        loss_ncc2 = self.mi_loss(self.image_opt*self.mask_true, self.opt_reg_aff) + self.mi_loss(self.image_sar*self.mask_true, self.sar_reg_aff)

        loss_reg = 10*loss_tp + loss_ncc2 + dc1

    
        dc2 = self.sym_loss(self.disp['sar2opt'].permute(0,2,3,1), self.disp1['opt2sar'].permute(0,2,3,1)) + self.sym_loss(self.disp['opt2sar'].permute(0,2,3,1), self.disp1['sar2opt'].permute(0,2,3,1))
 
        loss_disp1 = (torch.sum(abs(self.pre_disp_sw2o - self.gt_disp).pow(2))/(h*w)).sqrt()
        loss_disp2 = (torch.sum(abs(self.pre_disp_ow2s - self.gt_disp).pow(2))/(h*w)).sqrt()
        loss_disp = loss_disp1 + loss_disp2

        loss_ncc1 = self.mi_loss(self.image_opt*self.mask_true, self.image_opt_reg) + self.mi_loss(self.image_sar*self.mask_true, self.image_sar_reg) 
        
        loss_smooth_down2 = smoothloss(self.u)
        loss_smooth_down4 = smoothloss(self.v)
        loss_smooth = loss_smooth_down2 + loss_smooth_down4

        loss_re = border_suppression(self.image_sar_reg, self.mask_true) + border_suppression(self.image_opt_reg,self.mask_true) 
           
        loss_d = 10*ld_loss + 10*loss_disp + loss_smooth + loss_re + loss_ncc1 + dc2

        loss_total = loss_reg + loss_d

        loss_total.backward()

        self.loss_tp = loss_tp/2
        self.loss_disp = loss_disp/2


    def update_lr(self):

        self.RES_sch.step()
        self.UN_sch.step()

    def save(self, filename):
        state = {

            'RES': self.RES.state_dict(),
            'UN': self.UN.state_dict(),
            'RES_opt': self.RES_opt.state_dict(),
            'UN_opt': self.UN_opt.state_dict(),

        }
        torch.save(state, filename)
        return

    def assemble_outputs(self):
        images_ir = normalize_image(self.image_sar).detach()
        images_vi = normalize_image(self.image_opt).detach()
        images_ir_warp = normalize_image(self.image_sar_warp).detach()
        images_vi_warp = normalize_image(self.image_opt_warp).detach()
        images_ir_Reg = normalize_image(self.sar_reg_aff).detach()
        images_vi_Reg = normalize_image(self.opt_reg_aff).detach()
        images_ir_fake = normalize_image(self.image_sar_reg).detach()
        images_vi_fake = normalize_image(self.image_opt_reg).detach()

        row1 = torch.cat(
            (images_ir[0:1, ::], images_ir_warp[0:1, ::], images_ir_Reg[0:1, ::], images_ir_fake[0:1, ::]), 3)
        row2 = torch.cat(
            (images_vi[0:1, ::], images_vi_warp[0:1, ::], images_vi_Reg[0:1, ::], images_vi_fake[0:1, ::]), 3)
        return torch.cat((row1, row2), 2)
