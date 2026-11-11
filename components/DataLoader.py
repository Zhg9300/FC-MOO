import numpy as np


class DataLoader:

    def __init__(self,
                 name='DataLoader',
                 nickname='DataLoader',
                 pool_size=0,
                 batch_size=0,
                 input_require_shape=None,
                 *args,
                 **kwargs):
        self.name = name
        self.nickname = nickname
        self.pool_size = pool_size
        self.batch_size = batch_size
        self.input_require_shape = input_require_shape
        self.input_data_shape = None
        self.target_class_num = None
        self.data_pool = None

    def allocate(self, client_list):

        choose_data_pool_item_indices = np.random.choice(
            list(range(self.pool_size)), len(client_list), replace=False)
        for idx, client in enumerate(client_list):
            data_pool_item = self.data_pool[choose_data_pool_item_indices[idx]]
            client.update_data(choose_data_pool_item_indices[idx],
                               data_pool_item['local_training_data'],
                               data_pool_item['local_training_number'],
                               data_pool_item['local_test_data'],
                               data_pool_item['local_test_number'],
                               )

    def cal_data_shape(self, raw_input_data_shape):
        def cal(require_shape, raw_shape):
            if len(require_shape) == len(raw_shape) - 1:
                data_shape = list(raw_shape[1:])
            else:
                data_shape = []
                for i in range(1, len(raw_shape)):
                    if i < len(require_shape) + 1:
                        data_shape.append(raw_shape[i])
                    else:
                        data_shape[-1] *= raw_shape[i]
            return data_shape
        self.input_data_shape = cal(
            self.input_require_shape, raw_input_data_shape)

    @staticmethod
    def separate_list(input_list, n):
        def separate(input_list, n):
            for i in range(0, len(input_list), n):
                yield input_list[i: i + n]

        return list(separate(input_list, n))
