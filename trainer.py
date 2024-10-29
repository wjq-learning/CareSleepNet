import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.model import Model
from tqdm import tqdm
import numpy as np
import torch
from datasets.dataset import LoadDataset
from evaluator import Evaluator
from torch.nn import CrossEntropyLoss
from timeit import default_timer as timer
from models.loss import FocalLoss


def burnin_schedule(i):
    if i < 40:
        factor = 1
    else:
        factor = 0.1
    return factor


class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.data_loader = data_loader

        self.evaluator = Evaluator(params, self.data_loader['test'])

        self.model = model.cuda()
        if self.params.loss_function == "CrossEntropyLoss":
            self.criterion = CrossEntropyLoss().cuda()
        else:
            self.criterion = FocalLoss().cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.lr)
        self.data_length = len(self.data_loader['train'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.params.epochs * self.data_length
        )
        print(self.model)

    def train(self):
        f1_best = 0
        acc_best = 0
        cm_best = None
        kappa_best = 0
        wake_f1_best = 0
        n1_f1_best = 0
        n2_f1_best = 0
        n3_f1_best = 0
        rem_f1_best = 0
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)
                # print(self.model.epoch_encoder_eeg.encoder[14].attention_probs.shape)
                loss = self.criterion(pred.transpose(1, 2), y)
                # print(y)
                # print(predict)
                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, f1, cm, kappa, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1 = self.evaluator.get_accuracy(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, f1: {:.5f}, kappa: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        f1,
                        kappa,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print(cm)
                print(
                    "wake_f1: {:.5f}, n1_f1: {:.5f}, n2_f1: {:.5f}, n3_f1: {:.5f}, rem_f1: {:.5f}".format(
                        wake_f1,
                        n1_f1,
                        n2_f1,
                        n3_f1,
                        rem_f1,
                    )
                )
                if f1 > f1_best:
                    print("f1 increasing....saving weights !!")
                    best_f1_epoch = epoch + 1
                    f1_best = f1
                    acc_best = acc
                    cm_best = cm
                    kappa_best = kappa
                    wake_f1_best = wake_f1
                    n1_f1_best = n1_f1
                    n2_f1_best = n2_f1
                    n3_f1_best = n3_f1
                    rem_f1_best = rem_f1

                    model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_f1_{:.5f}_kappa_{:.5f}.pth".format(epoch + 1, acc, f1, kappa)
                    torch.save(self.model.state_dict(), model_path)
                    print("model save in " + model_path)


            if epoch + 1 == self.params.epochs:
                print("{} epoch get the best f1 {:.5f}".format(best_f1_epoch, f1_best))
                print("the model is save in " + model_path)
        evaluation_best = np.array([acc_best, f1_best, kappa_best, wake_f1_best, n1_f1_best, n2_f1_best, n3_f1_best, rem_f1_best])
        return evaluation_best, cm_best