import os
import torchvision


class Saver():
    def __init__(self, opts):
        self.model_dir = os.path.join(opts.model_dir, opts.data_name)
        self.image_dir = os.path.join(opts.train_img_dir, opts.data_name, 'train', 'image')

        # make directory
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

    # save result images
    def write_img(self, ep, model):
        assembled_images = model.assemble_outputs()
        img_filename = '%s/train_%05d.jpg' % (self.image_dir, ep)
        torchvision.utils.save_image(assembled_images, img_filename, nrow=1)
        if ep == -1:
            assembled_images = model.assemble_outputs()
            img_filename = '%s/train_last.jpg' % (self.image_dir, ep)
            torchvision.utils.save_image(assembled_images, img_filename, nrow=1)

    # save model
    def save_model(self, opts, ep, model, mode):
        if mode == 'ing':
            mode_save_path = '%s/%s%s_%d.pth' % (self.model_dir, opts.data_name, mode, ep)
            model.save(mode_save_path)
        if mode == 'last':
            mode_save_path = '%s/%s%s_%d.pth' % (self.model_dir, opts.data_name, mode, ep)
            model.save(mode_save_path)
            

