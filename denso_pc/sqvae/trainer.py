import time

from trainer_base import TrainerBase
from util import *
# from third_party.semseg import SegmentationMetric
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from visual import show_quant_indices, show_anomaly
import torchvision

class GaussianSQVAETrainer(TrainerBase):
    def __init__(self, cfgs, flgs, train_loader, val_loader, test_loader):
        super(GaussianSQVAETrainer, self).__init__(
            cfgs, flgs, train_loader, val_loader, test_loader)
        self.plots = {
            "loss_train": [], "mse_train": [], "perplexity_train": [],
            "loss_val": [], "mse_val": [], "perplexity_val": [],
            "loss_test": [], "mse_test": [], "perplexity_test": []
        }
        
    def _train(self, epoch):
        train_loss = []
        ms_error = []
        perplexity = []
        self.model.train()
        start_time = time.time()
        for batch_idx, (x, _) in enumerate(self.train_loader):
            if self.flgs.decay:
                step = (epoch - 1) * len(self.train_loader) + batch_idx + 1
                temperature_current = self._set_temperature(
                    step, self.cfgs.quantization.temperature)
                self.model.module.quantizer.set_temperature(temperature_current)
            x = x.cuda()
            _, _, loss = self.model(x, flg_train=True, flg_quant_det=False)
            self.optimizer.zero_grad()
            loss["all"].backward()
            self.optimizer.step()

            train_loss.append(loss["all"].detach().cpu().item())
            ms_error.append(loss["mse"].detach().cpu().item())
            perplexity.append(loss["perplexity"].detach().cpu().item())

        result = {}
        result["loss"] = np.asarray(train_loss).mean(0)
        result["mse"] = np.array(ms_error).mean(0)
        result["perplexity"] = np.array(perplexity).mean(0)
        self.print_loss(result, "train", time.time()-start_time)
                
        return result    
    
    def _test(self, mode="validation"):
        self.model.eval()
        _ = self._test_sub(False, mode)
        result = self._test_sub(True, mode)
        self.scheduler.step(result["loss"])
        return result

    def _test_sub(self, flg_quant_det, mode="validation"):
        test_loss = []
        ms_error = []
        perplexity = []
        if mode == "validation":
            data_loader = self.val_loader
        elif mode == "test":
            data_loader = self.test_loader
        start_time = time.time()
        with torch.no_grad():
            for x, _ in data_loader:
                x = x.cuda()
                _, _, loss = self.model(x, flg_quant_det=flg_quant_det)
                test_loss.append(loss["all"].item())
                ms_error.append(loss["mse"].item())
                perplexity.append(loss["perplexity"].item())
        result = {}
        result["loss"] = np.asarray(test_loss).mean(0)
        result["mse"] = np.array(ms_error).mean(0)
        result["perplexity"] = np.array(perplexity).mean(0)
        self.print_loss(result, mode, time.time()-start_time)
        
        return result
    
    def generate_reconstructions(self, filename, nrows=4, ncols=8):
        self._generate_reconstructions_continuous(filename, nrows=nrows, ncols=ncols)
    
    def print_loss(self, result, mode, time_interval):
        myprint(mode.capitalize().ljust(16) +
            "Loss: {:5.4f}, MSE: {:5.4f}, Perplexity: {:5.4f}, Time: {:5.3f} sec"
            .format(
                result["loss"], result["mse"], result["perplexity"], time_interval
            ), self.flgs.noprint)

    
    @torch.no_grad()
    def visualization(self):
        self.model.eval()
        
        quant_indices_list = []
        quant_label_list = []
        
        for x, _ in tqdm(self.val_loader, total=len(self.val_loader)):
            x = x.cuda()
            x_reconst, latents, loss, indices = self.model(x, flg_train=False, flg_quant_det=True)
            # recon_imgs = torch.cat([x, x_reconst],dim=0)
            # torchvision.utils.save_image(recon_imgs,
            #                         "recon_imgs2.png",
            #                         normalize=True,
            #                         n_row=x.shape[0])
            # show_quant_indices(indices, quant_num=512)
            # input()
            quant = indices.cpu().numpy()
            quant_indices_list.append(quant)
            quant_label_list.append(np.zeros_like(quant))
        
        for data, targets in tqdm(self.test_loader, total=len(self.test_loader)):
            data, targets = data.cuda(), targets.cuda()
            x_reconst, latents, loss, indices = self.model(data, flg_train=False, flg_quant_det=True)
            # recon_imgs = torch.cat([data, x_reconst],dim=0)
            # torchvision.utils.save_image(recon_imgs,
            #                         "recon_imgs.png",
            #                         normalize=True,
            #                         n_row=data.shape[0])
            # show_quant_indices(indices, quant_num=512)
            # input()

            quant = indices.cpu().numpy()
            masks = F.interpolate(targets, size=quant.shape[-2:])[:, 0]
            masks = (masks > 0.99)*1.0
            quant_indices_list.append(quant)
            quant_label_list.append(masks.cpu().numpy())
        
        quant = np.concatenate(quant_indices_list, axis=0)
        label = np.concatenate(quant_label_list, axis=0)

        normaly = quant[label == 0]
        anomaly = quant[label == 1]
        bins = 128
        under = 0
        # plt.hist([normaly, anomaly], bins, range=[0, 511], color=["b", "r"], label=['normaly', 'anomaly'], log=True, stacked=True)
        plt.hist(normaly, bins, range=[under, under+bins], alpha = 0.5, color="b", label='normaly', log=True)
        plt.hist(anomaly, bins, range=[under, under+bins], alpha = 0.5, color="r", label='anomaly', log=True)
        plt.legend(loc='upper left')
        plt.savefig(f"quantize_indices.png")
        plt.close()
        normaly_idx, normaly_counts = np.unique(normaly, return_counts=True)
        anomaly_idx, anomaly_counts = np.unique(anomaly, return_counts=True)
        counts = np.zeros([2, 512])
        counts[0, normaly_idx] = normaly_counts
        counts[1, anomaly_idx] = anomaly_counts
        counts = counts[:, np.all(counts != 0, axis=0)]
        # counts /= len(normaly)
        plt.scatter(counts[0], counts[1], s=10, color="k")
        plt.savefig(f"scatter.png")
        print(f"共分散:{np.cov(counts[0], counts[1])}")
        plt.close()

class VmfSQVAETrainer(TrainerBase):
    def __init__(self, cfgs, flgs, train_loader, val_loader, test_loader):
        super(VmfSQVAETrainer, self).__init__(
            cfgs, flgs, train_loader, val_loader, test_loader)
        # self.metric_semseg = SegmentationMetric(cfgs.network.num_class)
        self.plots = {
            "loss_train": [], "acc_train": [], "perplexity_train": [],
            "loss_val": [], "acc_val": [], "perplexity_val": [], "miou_val": [],
            "loss_test": [], "acc_test": [], "perplexity_test": [], "miou_test": []
        }
    
    def _train(self, epoch):
        train_loss = []
        acc = []
        perplexity = []
        self.model.train()
        start_time = time.time()
        for batch_idx, (x, y) in enumerate(self.train_loader):
            y = self.preprocess(x, y)
            if self.flgs.decay:
                step = (epoch - 1) * len(self.train_loader) + batch_idx + 1
                temperature_current = self._set_temperature(
                    step, self.cfgs.quantization.temperature)
                self.model.module.quantizer.set_temperature(temperature_current)
            _, _, loss = self.model(y, flg_train=True, flg_quant_det=False)
            self.optimizer.zero_grad()
            loss["all"].backward()
            self.optimizer.step()

            train_loss.append(loss["all"].item())
            acc.append(loss["acc"].item())
            perplexity.append(loss["perplexity"].item())

        result = {}
        result["loss"] = np.asarray(train_loss).mean(0)
        result["acc"] = np.array(acc).mean(0)
        result["perplexity"] = np.array(perplexity).mean(0)
        self.print_loss(result, "train", time.time()-start_time)
        
        return result
    
    def _test(self, mode="val"):
        _ = self._test_sub(False)
        result = self._test_sub(True, mode)
        self.scheduler.step(result["loss"])
        return result
    
    def _test_sub(self, flg_quant_det, mode="val"):
        test_loss = []
        acc = []
        perplexity = []
        self.metric_semseg.reset()
        if mode == "val":
            data_loader = self.val_loader
        elif mode == "test":
            data_loader = self.test_loader
        start_time = time.time()
        with torch.no_grad():
            for x, y in data_loader:
                y = self.preprocess(x, y)
                x_reconst, _, loss = self.model(y, flg_quant_det=flg_quant_det)
                self.metric_semseg.update(x_reconst, y)
                pixAcc, mIoU, _ = self.metric_semseg.get()
                test_loss.append(loss["all"].item())
                acc.append(loss["acc"].item())
                perplexity.append(loss["perplexity"].item())
            pixAcc, mIoU, _ = self.metric_semseg.get()
        result = {}
        result["loss"] = np.asarray(test_loss).mean(0)
        result["acc"] = np.array(acc).mean(0)
        result["miou"] = mIoU
        result["perplexity"] = np.array(perplexity).mean(0)
        self.print_loss(result, mode, time.time()-start_time)
        myprint("%15s"%"PixAcc: {:5.4f} mIoU: {:5.4f}".format(
            pixAcc, mIoU
        ), self.flgs.noprint)
        
        return result
    
    def generate_reconstructions(self, filename, nrows=4, ncols=8):
        self._generate_reconstructions_discrete(filename, nrows=nrows, ncols=ncols)
    
    def print_loss(self, result, mode, time_interval):
        myprint(mode.capitalize().ljust(16) +
            "Loss: {:5.4f}, ACC: {:5.4f}, Perplexity: {:5.4f}, Time: {:5.3f} sec"
            .format(
            result["loss"], result["acc"], result["perplexity"], time_interval
            ), self.flgs.noprint)


