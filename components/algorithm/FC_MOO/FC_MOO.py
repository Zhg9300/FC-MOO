import components as cn
import torch
import numpy as np
from components.algorithm.common.utils import get_d_fc_moo


class FC_MOO(cn.Algorithm):
    def __init__(self,
                 name='FC-MOO',
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
                train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay']) + ' prefer' + str(prefer) + ' rho' + str(rho)
                + ' momentum' + str(train_setting['momentum']))

        super().__init__(name, data_loader, loader_name, module, device, train_setting, client_num, client_list, online_client_num,
                         metric_list, max_comm_round, max_training_num, epochs, save_name, outFunc, write_log, dishonest, test_conflicts, params)
        self.prefer = prefer
        self.rho = rho


    def run(self):
        batch_num = np.mean(self.get_clinet_attr('training_batch_num'))
        while not self.terminated():
            m_locals, l_locals = self.train()
            g_locals = []
            old_model = self.module.span_model_params_to_vec()
            for local_model in m_locals:
                current_g = (old_model - local_model) / self.lr
                g_locals.append(current_g)
            g_locals = torch.stack(g_locals, dim=0)
            l = torch.tensor(l_locals, dtype=torch.float32, device=self.device)
            fairness = l.var(unbiased=False)
            l = l - l.mean()
            g_fair = (2.0 / len(l_locals)) * (l.unsqueeze(1) * g_locals).sum(dim=0)
            phi = self.rho * (fairness - self.prefer)
            d, _ = get_d_fc_moo(
                task_grads=g_locals,
                g_fair=g_fair,
                device=self.device,
                phi=phi,
                mu_cap=1000,
            )
            self.update_module(self.module, self.optimizer, self.lr, d)
            self.client_update()
            self.current_training_num += self.epochs * batch_num
