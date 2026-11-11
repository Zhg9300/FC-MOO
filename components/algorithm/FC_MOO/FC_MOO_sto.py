import components as cn
import torch
import numpy as np
from components.algorithm.common.utils import get_d_fc_moo

class FC_MOO_sto(cn.Algorithm):
    def __init__(self,
                 name='FC-MOO-sto',
                 data_loader=None,
                 loader_name=None,
                 module=None,
                 device=None,
                 train_setting=None,
                 client_num=None,
                 client_list=None,
                 online_client_num=None,
                 metric_list=None,
                 max_comm_round=0,
                 max_training_num=0,
                 epochs=1,
                 save_name=None,
                 outFunc=None,
                 write_log=True,
                 dishonest=None,
                 test_conflicts=False,
                 params=None,
                 prefer=6e-3,
                 rho=5,
                 *args,
                 **kwargs):
        if params is not None:
            prefer = params['prefer']
            rho = params['rho']
        if save_name is None:
            save_name = (name + ' ' + module.name + ' E' + str(epochs) + ' lr' + str(
                train_setting['optimizer'].defaults['lr']) + ' decay' + str(
                train_setting['lr_decay']) + ' prefer' + str(prefer) + ' rho' + str(rho)
                         + ' momentum' + str(train_setting['momentum']))

        super().__init__(name, data_loader, loader_name, module, device, train_setting, client_num, client_list,
                         online_client_num,
                         metric_list, max_comm_round, max_training_num, epochs, save_name, outFunc, write_log,
                         dishonest, test_conflicts, params)
        self.prefer = prefer
        self.rho = rho

    def run(self):
        batch_num = np.mean(self.get_clinet_attr('training_batch_num'))
        round = 0
        accumulated_grads = None
        accumulated_loss = None
        momentum = self.train_setting['optimizer'].defaults['momentum']

        while not self.terminated():
            round += 1
            beta = momentum
            old_model = self.module.span_model_params_to_vec()

            # one gradient sample
            m_locals, l_locals1 = self.train()
            l1 = torch.tensor(l_locals1, dtype=torch.float32, device=self.device)
            current_grads = []
            for idx in range(len(m_locals)):
                current_g = (old_model - m_locals[idx]) / self.lr
                current_grads.append(current_g)
            current_grads = torch.stack(current_grads, dim=0)

            if round == 1:
                accumulated_grads = current_grads.clone()
                accumulated_loss = l1.clone()
            else:
                accumulated_grads = (1 - beta) * current_grads + beta * accumulated_grads
                accumulated_loss = (1 - beta) * l1 + beta * accumulated_loss

            n = self.client_num
            l_centered = accumulated_loss - accumulated_loss.mean()
            fairness = l_centered.pow(2).sum() / n
            current_g_fair = (2.0 / n) * (l_centered.unsqueeze(1) * accumulated_grads).sum(dim=0)

            phi = self.rho * (fairness - self.prefer)

            d, _ = get_d_fc_moo(
                task_grads=accumulated_grads,
                g_fair=current_g_fair,
                device=self.device,
                phi=phi,
                mu_cap=1000,
            )

            self.update_module(self.module, self.optimizer, self.lr, d)
            self.client_update()
            self.current_training_num += self.epochs * batch_num
