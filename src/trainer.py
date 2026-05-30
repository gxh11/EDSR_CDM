import os
import math
from decimal import Decimal

import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        
        self.loader_train.dataset.set_scale(0)
        
        # 🌟【核心修改 1】适配 FLIR 的双流解包。
        # 原版是 for batch, (lr, hr, _,) ... 
        # 现在多模态第一项是一个元组，我们直接用 (lr_thermal, hr_visible) 拆开承接
        for batch, ((lr_thermal, hr_visible), hr_thermal, _) in enumerate(self.loader_train):
            
            # 使用升级版的 prepare 函数，将所有输入端安全送入 GPU/CPU 
            lr_thermal, hr_visible, hr_thermal = self.prepare(lr_thermal, hr_visible, hr_thermal)
            
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            
            # 🌟【核心修改 2】前向传播喂入双流参数：红外低清输入 + 可见光高清引导
            sr = self.model(lr_thermal, hr_visible)
            
            # 损失函数对比的是超分出来的红外图（sr）和红外高清原图（hr_thermal）
            loss = self.loss(sr, hr_thermal)
            
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                
                # 🌟【核心修改 3】验证/测试循环内同步修改双流数据解包逻辑
                for (lr_thermal, hr_visible), hr_thermal, filename in tqdm(d, ncols=80):
                    
                    # 搬运至 GPU
                    lr_thermal, hr_visible, hr_thermal = self.prepare(lr_thermal, hr_visible, hr_thermal)
                    
                    # 双流前向传播推理
                    sr = self.model(lr_thermal, hr_visible)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    
                    # 评估 PSNR：对比 SR 结果和红外真值（hr_thermal）
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr_thermal, scale, self.args.rgb_range, dataset=d
                    )
                    if self.args.save_gt:
                        # 如果需要保存中间图像，把低清红外和高清红外真值存进去
                        save_list.extend([lr_thermal, hr_thermal])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        """🌟【核心修改 4】自适应全类型张量设备搬运工"""
        if self.args.cpu:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        def _prepare(tensor):
            if self.args.precision == 'half': 
                tensor = tensor.half()
            return tensor.to(device)

        # 完美兼容普通 Tensor 或包装在嵌套结构里的多模态数据流
        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs