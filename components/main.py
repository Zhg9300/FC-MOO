import components as cn
import numpy as np
import argparse
import torch


def outFunc(alg):
    loss_list = []
    for i, metric_history in enumerate(alg.comm_log['client_metric_history']):
        test_loss = metric_history['test_loss'][-1]
        if test_loss is None:
            continue
        loss_list.append(test_loss)
    loss_list = np.array(loss_list)

    local_acc_list = []
    for i, metric_history in enumerate(alg.comm_log['client_metric_history']):
        local_acc_list.append(metric_history['test_accuracy'][-1])
    local_acc_list = np.array(local_acc_list)
    print(str(alg.params))

    stream_log = ""
    stream_log += alg.save_name + ' ' + alg.data_loader.nickname + '\n'
    stream_log += 'round {}'.format(alg.current_comm_round) + '\n'
    stream_log += f'Mean Global Test loss: {format(np.mean(loss_list), ".3f")}' + \
        '\n' if len(loss_list) > 0 else ''
    stream_log += 'Global model test: \n'
    stream_log += f'Test Acc List: {[f"{x:.3f}" for x in local_acc_list]}\n'
    stream_log += f'Average: {format(np.mean(local_acc_list), ".3f")}. Variance: {format(np.var(local_acc_list), ".3f")}. Min: {format(np.min(local_acc_list), ".3f")}. Max: {format(np.max(local_acc_list), ".3f")}' + '\n'
    stream_log += '\n'
    alg.stream_log = stream_log + alg.stream_log
    print(stream_log)


def read_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', help='seed', type=int, default=1)

    parser.add_argument(
        '--device', help='device: -1, 0, 1, or ...', type=int, default=0)

    parser.add_argument('--module', help='module name;',
                        type=str, default='CNN')

    parser.add_argument('--algorithm', help='algorithm name',
                        choices=tuple(cn.ALGORITHMS), default='FedAvg')

    parser.add_argument('--dataloader', help='dataloader name;',
                        type=str, default='DataLoader_cifar10_pat')

    parser.add_argument('--B', help='batch size', type=int, default=50)

    parser.add_argument('--NC', help='client_class_num', type=int, default=1)

    parser.add_argument(
        '--balance', help='balance or not for pathological separation', type=str, default='True')

    parser.add_argument(
        '--Diralpha', help='alpha parameter for dirichlet', type=float, default=0.1)

    parser.add_argument('--N', help='client num', type=int, default=100)

    parser.add_argument(
        '--C', help='select client proportion', type=float, default=1.0)

    parser.add_argument('--R', help='communication round',
                        type=int, default=3000)

    parser.add_argument('--E', help='local epochs', type=int, default=1)

    parser.add_argument('--test_interval',
                        help='test interval', type=int, default=50)

    parser.add_argument('--test_conflicts',
                        help='test conflicts', type=str, default='False')

    parser.add_argument('--sgd_step', help='sgd training',
                        type=str, default='False')

    parser.add_argument('--lr', help='learning rate', type=float, default=0.1)
    parser.add_argument('--decay', help='learning rate decay',
                        type=float, default=0.999)
    parser.add_argument('--momentum', help='momentum', type=float, default=0.0)

    parser.add_argument(
        '--lam', help='gradient ascent step size in AFL', type=float, default=0.8)

    parser.add_argument(
        '--epsilon', help='parameter epsilon in FedMGDA+', type=float, default=0.1)
    parser.add_argument(
        '--theta', help='parameter theta in FedMDFG', type=float, default=11.25)
    parser.add_argument(
        '--s', help='parameter s in FedMDFG', type=float, default=5)
    parser.add_argument(
        '--pow', help='parameter pow in AdaFed', type=float, default=3)
    parser.add_argument(
        '--prefer', help='fairness threshold in FairMOO and FC-MOO', type=float, default=6e-3)
    parser.add_argument(
        '--rho', help='constraint coefficient in FC-MOO', type=float, default=5)
    parser.add_argument('--q', help='parameter q in qFedAvg',
                        type=float, default=0.1)

    parser.add_argument(
        '--pretrained', help='use ImageNet pretrained weights (for ResNet)',
        type=str, default='False')

    parser.add_argument('--dishonest_num',
                        help='dishonest number', type=int, default=0)
    parser.add_argument('--scaled_update',
                        help='scaled update attack', type=str, default='None')
    parser.add_argument('--random_update',
                        help='random update attack', type=str, default='None')
    parser.add_argument(
        '--zero_update', help='zero update attack', type=str, default='None')

    try:
        parsed = vars(parser.parse_args())
        return parsed
    except IOError as msg:
        parser.error(str(msg))


def initialize(params):
    cn.setup_seed(seed=params['seed'])
    device = torch.device(
        'cuda:' + str(params['device']) if torch.cuda.is_available() and params['device'] != -1 else "cpu")
    Module = getattr(cn, params['module'])
    module = Module(device)
    if hasattr(module, 'pretrained'):
        module.pretrained = eval(params['pretrained'])
    Dataloader = getattr(cn, params['dataloader'])
    data_loader = Dataloader(
        params=params, input_require_shape=module.input_require_shape, device=device)
    print(f"Inputs device: {data_loader.device}")

    module.generate_model(data_loader.input_data_shape,
                          data_loader.target_class_num)

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, module.model.parameters(
    )), lr=params['lr'], momentum=params['momentum'], weight_decay=5e-4)
    train_setting = {'criterion': torch.nn.CrossEntropyLoss(label_smoothing=0.1),
                     'optimizer': optimizer, 'lr_decay': params['decay'],
                     'sgd_step': eval(params['sgd_step']),
                     'momentum': params['momentum']}
    test_interval = params['test_interval']
    loader_name = params['dataloader'].split('_')[1]
    dishonest_num = params['dishonest_num']
    scaled_update = eval(params['scaled_update'])
    if scaled_update is not None:
        scaled_update = float(scaled_update)
    dishonest = {'dishonest_num': dishonest_num,
                 'scaled_update': scaled_update,
                 'random_update': eval(params['random_update']),
                 'zero_update': eval(params['zero_update'])}
    test_conflicts = eval(params['test_conflicts'])
    Algorithm = cn.ALGORITHMS[params['algorithm']]
    algorithm = Algorithm(data_loader=data_loader,
                          loader_name=loader_name,
                          module=module,
                          device=device,
                          train_setting=train_setting,
                          client_num=data_loader.pool_size,
                          online_client_num=int(
                              data_loader.pool_size * params['C']),
                          metric_list=[cn.Correct()],
                          max_comm_round=params['R'],
                          max_training_num=None,
                          epochs=params['E'],
                          outFunc=outFunc,
                          write_log=True,
                          dishonest=dishonest,
                          test_conflicts=test_conflicts,
                          params=params,)
    algorithm.test_interval = test_interval
    return data_loader, algorithm


if __name__ == '__main__':
    params = read_params()
    data_loader, algorithm = initialize(params)
    algorithm.run()
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
