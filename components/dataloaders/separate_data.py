import components as cn
import numpy as np
import random
import torch
CIFAR100_COARSE_LABELS = [
    [4, 30, 55, 72, 95],          # aquatic mammals
    [1, 32, 67, 73, 91],          # fish
    [54, 62, 70, 82, 92],         # flowers
    [9, 10, 16, 28, 61],          # food containers
    [0, 51, 53, 57, 83],          # fruit and vegetables
    [22, 39, 40, 86, 87],         # household electrical devices
    [5, 20, 25, 84, 94],          # household furniture
    [6, 7, 14, 18, 24],           # insects
    [3, 42, 43, 88, 97],          # large carnivores
    [12, 17, 37, 68, 76],         # large man-made outdoor things
    [23, 33, 49, 60, 71],         # large natural outdoor scenes
    [15, 19, 21, 31, 38],         # large omnivores and herbivores
    [34, 63, 64, 66, 75],         # medium-sized mammals
    [26, 45, 77, 79, 99],         # non-insect invertebrates
    [2, 11, 35, 46, 98],          # people
    [27, 29, 44, 78, 93],         # reptiles
    [36, 50, 65, 74, 80],         # small mammals
    [47, 52, 56, 59, 96],         # trees
    [8, 13, 48, 58, 90],          # vehicles 1
    [41, 69, 81, 85, 89]          # vehicles 2
]

SEMANTIC_GROUPS = [
    [0, 1],
    [2, 4],
    [5, 6],
    [7, 13],
    [8, 11],
    [9, 10],
    [12, 16],
    [14, 15],
    [17, 17],
    [18, 19]
]

def get_coarse_label(fine_label):
    for coarse_idx, fine_labels in enumerate(CIFAR100_COARSE_LABELS):
        if fine_label in fine_labels:
            return coarse_idx
    return -1

def separate_data(data, train_prob, num_clients, target_class_num, item_classes_num, batch_size, alpha, niid=False, balance=False, partition=None):
    least_samples = batch_size / (1-train_prob)
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data
    all_indices = np.arange(len(dataset_label))

    coarse_labels = np.array([get_coarse_label(fine_label) for fine_label in dataset_label])

    dataidx_map = {}

    if not niid:
        partition = 'pat'
        item_classes_num = target_class_num
    if partition == 'pat1':
        if num_clients != 10:
            print(f"[警告] pat1分区需要10个客户端，但当前设置为{num_clients}。使用IID代替。")
            partition = 'pat'

        random.shuffle(SEMANTIC_GROUPS)

        if len(SEMANTIC_GROUPS) != num_clients:
            raise ValueError(f"预定义分组数量({len(SEMANTIC_GROUPS)})与客户端数量({num_clients})不匹配")

        for client_id, group in enumerate(SEMANTIC_GROUPS):
            client_indices = []

            for coarse_label in group:
                mask = coarse_labels == coarse_label
                coarse_indices = all_indices[mask]
                client_indices.extend(coarse_indices.tolist())

            dataidx_map[client_id] = np.array(client_indices)

    elif partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(target_class_num):
            idx_for_each_class.append(idxs[dataset_label == i])

        class_num_per_client = [item_classes_num for _ in range(num_clients)]

        min_required_clients = int(target_class_num / item_classes_num)
        if num_clients < min_required_clients:
            print(f"Warning: 客户端数量太少（{num_clients}），无法完成 pathological 分区，退化为 IID 分配")
            np.random.shuffle(idxs)
            num_items = len(idxs) // num_clients
            for client in range(num_clients):
                start = client * num_items
                end = len(idxs) if client == num_clients - 1 else (client + 1) * num_items
                dataidx_map[client] = idxs[start:end]
        else:
            for i in range(target_class_num):
                selected_clients = []
                for client in range(num_clients):
                    if class_num_per_client[client] > 0:
                        selected_clients.append(client)

                selected_clients = selected_clients[:int(num_clients / target_class_num * item_classes_num)]
                num_selected_clients = len(selected_clients)

                if num_selected_clients == 0:
                    continue

                num_all_samples = len(idx_for_each_class[i])
                num_per = num_all_samples / num_selected_clients

                if balance:
                    num_samples = [int(num_per) for _ in range(num_selected_clients - 1)]
                else:
                    num_samples = np.random.randint(
                        max(num_per / 10, least_samples / target_class_num),
                        num_per,
                        num_selected_clients - 1
                    ).tolist()
                num_samples.append(num_all_samples - sum(num_samples))

                idx = 0
                for client, num_sample in zip(selected_clients, num_samples):
                    if client not in dataidx_map:
                        dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
                    else:
                        dataidx_map[client] = np.append(
                            dataidx_map[client],
                            idx_for_each_class[i][idx:idx + num_sample],
                            axis=0
                        )
                    idx += num_sample
                    class_num_per_client[client] -= 1


    elif partition == "dir":

        min_size = 0
        K = target_class_num
        N = len(dataset_label)

        while min_size < target_class_num:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(
                    np.repeat(alpha, num_clients))
                proportions = np.array(
                    [p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) *
                               len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,
                             idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_clients):
            np.random.shuffle(idx_batch[j])
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client] == i))))

    del data

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(
            y[client]))
        print("\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic

def separate_batch_data(input_data, target_data, batch_size):
    batch_data_indices_list = cn.DataLoader.separate_list(
        list(range(len(target_data))), batch_size)
    local_data = []
    for batch_data_indices in batch_data_indices_list:
        batch_input_data = input_data[batch_data_indices]

        batch_target_data = target_data[batch_data_indices]
        local_data.append((batch_input_data, batch_target_data))
    return local_data


def create_data_pool(X, y, pool_size, shuffle, train_prob, batch_size, target_class_num):

    data_pool = [{} for _ in range(pool_size)]
    for pool_idx in range(pool_size):
        input_data = torch.Tensor(X[pool_idx]).float()
        target_data = torch.Tensor(y[pool_idx]).long()#.long()
        if shuffle:
            indices = list(range(len(target_data)))
            random.shuffle(indices)
            input_data = input_data[indices]
            target_data = target_data[indices]

        training_input_data = input_data[:int(train_prob*len(target_data))]
        training_target_data = target_data[:int(train_prob*len(target_data))]
        test_input_data = input_data[int(train_prob*len(target_data)):]
        test_target_data = target_data[int(train_prob*len(target_data)):]


        data_pool[pool_idx]['local_training_data'] = separate_batch_data(
            training_input_data, training_target_data, batch_size)
        data_pool[pool_idx]['local_training_number'] = len(
            training_target_data)
        data_pool[pool_idx]['data_name'] = str(pool_idx)

        data_pool[pool_idx]['local_test_data'] = separate_batch_data(
            test_input_data, test_target_data, batch_size)
        data_pool[pool_idx]['local_test_number'] = len(test_target_data)


    return data_pool
