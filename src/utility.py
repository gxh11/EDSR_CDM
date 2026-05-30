import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

def bg_target(queue):
    while True:
        if not queue.empty():
            filename, tensor = queue.get()
            if filename is None: break
            
            # 👈 【核心绝杀点】在子进程写盘前，判断是否为单通道红外图
            # tensor 的 shape 此时为 [H, W, 1]，如果是，直接 squeeze 掉变成 [H, W] 2D 灰度矩阵
            if tensor.ndim == 3 and tensor.shape[2] == 1:
                tensor = tensor.squeeze(2)
                
            imageio.imwrite(filename, tensor.numpy())
class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join('..', 'experiment', args.save)
        else:
            self.dir = os.path.join('..', 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_{}.pdf'.format(d)))
            plt.close(fig)

    

    def begin_background(self):
        self.queue = Queue()

        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        
        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, dataset, filename, save_list, scale):
            if self.args.save_results:
                # 1. 获取最干净的物理目标文件夹路径（拒绝套娃）
                target_dir = self.get_path('results-{}'.format(dataset.dataset.name))
                
                # 👈 【核心绝杀】强行让操作系统立刻在硬盘上创建这个 results-FLIR 文件夹
                # 这样 imageio 写盘时就绝对不可能报 FileNotFoundError！
                os.makedirs(target_dir, exist_ok=True)

                # 2. 纯粹提取文件名本身，扔掉可能夹带的任何相对路径斜杠
                base_filename = os.path.basename(filename)
                
                # 3. 完美拼装最终物理文件的绝对路径
                full_path = os.path.join(target_dir, '{}_x{}_'.format(base_filename, scale))

                postfix = ('SR', 'LR', 'HR')
                for v, p in zip(save_list, postfix):
                    # 4. 高精度浮点数像素保持，防止过早截断黑图
                    normalized = v[0].float().mul(255.0 / self.args.rgb_range)
                    normalized = normalized.clamp(0, 255).round()
                    
                    # 5. 调整维度并转为 uint8 搬回 CPU
                    tensor_cpu = normalized.permute(1, 2, 0).to('cpu', torch.uint8)
                    
                    # 6. 单通道红外图降维补丁
                    if tensor_cpu.ndim == 3 and tensor_cpu.shape[2] == 1:
                        tensor_cpu = tensor_cpu.squeeze(2)
                        
                    # 7. 将拼好的绝对文件名和数据丢给子进程
                    self.queue.put(('{}{}.png'.format(full_path, p), tensor_cpu))

            if self.args.save_results:
                filename = self.get_path(
                    'results-{}'.format(dataset.dataset.name),
                    '{}_x{}_'.format(filename, scale)
                )

                postfix = ('SR', 'LR', 'HR')
                for v, p in zip(save_list, postfix):
                    normalized = v[0].mul(255 / self.args.rgb_range)
                    tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                    
                    # 👈 【双保险补丁】塞进多进程队列前，如果发现是单通道，直接降维
                    if tensor_cpu.ndim == 3 and tensor_cpu.shape[2] == 1:
                        tensor_cpu = tensor_cpu.squeeze(2)
                        
                    self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))
def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch
    
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer

