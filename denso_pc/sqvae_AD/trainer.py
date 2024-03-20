import time

import torch
import torch.nn.functional as F
from trainer_base import TrainerBase
from util import *
from metrics import compute_imagewise_retrieval_metrics, compute_pixelwise_retrieval_metrics, compute_pro

# from third_party.semseg import SegmentationMetric


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


class SQ_FastFlow_Trainer(TrainerBase):
    def __init__(self, cfgs, flgs, train_loader, val_loader, test_loader):
        super(SQ_FastFlow_Trainer, self).__init__(
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
        log_jac_dets = []
        self.model.train()
        start_time = time.time()
        for batch_idx, (x, _) in enumerate(self.train_loader):
            if self.flgs.decay:
                step = (epoch - 1) * len(self.train_loader) + batch_idx + 1
                temperature_current = self._set_temperature(
                    step, self.cfgs.quantization.temperature)
                for quant in self.model.module.quantizer:
                    quant.set_temperature(temperature_current)
            x = x.cuda()
            _, _, loss = self.model(x, flg_train=True, flg_quant_det=False)
            self.optimizer.zero_grad()
            loss["all"].backward()
            self.optimizer.step()

            train_loss.append(loss["all"].detach().cpu().item())
            ms_error.append(loss["mse"].detach().cpu().item())
            perplexity.append(loss["perplexity"].detach().cpu().item())
            log_jac_dets.append(loss["jac_loss"].detach().cpu().item())

        result = {}
        result["loss"] = np.asarray(train_loss).mean(0)
        result["mse"] = np.array(ms_error).mean(0)
        result["perplexity"] = np.array(perplexity).mean(0)
        result["jac_loss"] = np.array(log_jac_dets).mean(0)
        self.print_loss(result, "train", time.time()-start_time)
                
        return result    

    def main_loop(self, max_iter=None, timestamp=None):
        if timestamp == None:
            self._make_path()
        else:
            self.path = os.path.join(self.cfgs.path, timestamp)
        BEST_ACC = 0.0
        LAST_SAVED = -1

        if max_iter == None:
            max_iter = self.cfgs.train.epoch_max
        for epoch in range(1, max_iter+1):
            myprint("[Epoch={}]".format(epoch), self.flgs.noprint)
            res_train = self._train(epoch)
            if self.flgs.save:
                self._writer_train(res_train, epoch)
            res_test = self.detection()
            
            if self.flgs.save:
                if sum(list(res_test.values())) >= BEST_ACC:
                    BEST_ACC = sum(list(res_test.values()))
                    LAST_SAVED = epoch
                    myprint("----Saving model!", self.flgs.noprint)
                    torch.save(
                        self.model.state_dict(), os.path.join(self.path, "best.pt"))
                    self.generate_reconstructions(
                        os.path.join(self.path, "reconstrucitons_best"))
                else:
                    myprint("----Not saving model! Last saved: {}"
                        .format(LAST_SAVED), self.flgs.noprint)
                torch.save(
                    self.model.state_dict(), os.path.join(self.path, "current.pt"))
                self.generate_reconstructions(
                    os.path.join(self.path, "reconstructions_current"))

    def detection(self, mode="test"):
        self.model.eval()
        anomaly_map = []
        labels = []

        if mode == "validation":
            data_loader = self.val_loader
        elif mode == "test":
            data_loader = self.test_loader

        with torch.no_grad():
            for x, targets in data_loader:
                x = x.cuda()
                _, quant_info, _ = self.model(x, flg_quant_det=True)
                anomaly_map_list = []
                for z_from_encoder, z_to_decoder in zip(quant_info["z_from_encoder"], quant_info["z_to_decoder"]):
                    log_prob = -torch.mean((z_from_encoder - z_to_decoder)**2, dim=1, keepdim=True) * 0.5
                    prob = torch.exp(log_prob)
                    ano_map = F.interpolate(
                        -prob,
                        size=[x.shape[-2], x.shape[-1]],
                        mode="bilinear",
                        align_corners=False,
                    )
                    anomaly_map_list.append(ano_map)
                anomaly_map_list = torch.stack(anomaly_map_list, dim=-1)
                ano_map = torch.mean(anomaly_map_list, dim=-1)
                anomaly_map.append(ano_map.detach().cpu().numpy())
                labels.append(targets.numpy())
        
        predicts = np.concatenate(anomaly_map, axis=0)
        labels = np.concatenate(labels, axis=0)

        image_scores = predicts.reshape(predicts.shape[0], -1).max(axis=-1)
        image_labels = labels.reshape(labels.shape[0], -1).max(axis=-1)
        pixel_scores = predicts[:,0]
        pixel_labels = labels[:,0]

        imagewize_AUROC = compute_imagewise_retrieval_metrics(image_scores, image_labels)
        pixelwize_AUROC = compute_pixelwise_retrieval_metrics(pixel_scores, pixel_labels)
        PRO = compute_pro(pixel_labels, pixel_scores)

        print("Image level AUROC: {:.2%}".format(imagewize_AUROC["auroc"]))
        print("Pixel level AUROC: {:.2%}".format(pixelwize_AUROC["auroc"]))
        print("PRO: {:.2%}".format(PRO))

        result_dict = {"Image_level_AUROC":imagewize_AUROC["auroc"], "Pixel_level_AUROC":pixelwize_AUROC["auroc"], "PRO": PRO}

        result_path = os.path.join(self.path, "_result.txt")
        with open(result_path, "w") as f:
            for k, v in result_dict.items():
                f.write(f"{k}\t {v:.2%}\n")

        return result_dict
    
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
        log_jac_dets = []
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
                log_jac_dets.append(loss["jac_loss"].item())

        result = {}
        result["loss"] = np.asarray(test_loss).mean(0)
        result["mse"] = np.array(ms_error).mean(0)
        result["perplexity"] = np.array(perplexity).mean(0)
        result["jac_loss"] = np.array(log_jac_dets).mean(0)
        self.print_loss(result, mode, time.time()-start_time)
        
        return result
    
    def generate_reconstructions(self, filename, nrows=4, ncols=8):
        self._generate_reconstructions_continuous(filename, nrows=nrows, ncols=ncols)
    
    def print_loss(self, result, mode, time_interval):
        myprint(mode.capitalize().ljust(16) +
            "Loss: {:5.4f}, MSE: {:5.4f}, Perplexity: {:5.4f}, jac_loss: {:.2f}, Time: {:5.3f} sec"
            .format(
                result["loss"], result["mse"], result["perplexity"], result["jac_loss"], time_interval
            ), self.flgs.noprint)


