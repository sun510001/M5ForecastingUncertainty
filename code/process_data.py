# -*- coding:utf-8 -*-
from __init__ import *
from load_data import LoadData


class ProcessDataCNN(object):
    def __init__(self, sales):
        global LAGS
        print("start process_data...")

        df_head = sales.head()

        if not PROC_CSV_EXIST:
            sales["x"] = sales["demand"] / sales["scale1"]
            self.FEATS = []
            for lag in tqdm(LAGS):
                sales[f"x_{lag}"] = sales.groupby("id")["x"].shift(lag)
                self.FEATS.append(f"x_{lag}")

            print(sales.shape)
            sales = sales.loc[sales.nb > sales.start]
            print(sales.shape)

            nb = sales['nb'].values
            MAX_LAG = max(LAGS)

            # SORRY THIS IS FAKE VALIDATION. I DIDN'T THINK IT WOULD HAVE HAD LIFTED UP MY SCORE LIKE THAT
            self.tr_mask = np.logical_and(nb > START + MAX_LAG, nb <= 1941)
            self.val_mask = np.logical_and(nb > 1913, nb <= 1941)
            self.te_mask = np.logical_and(nb > 1941, nb <= 1969)

            print('processing csv file...')

            # def preprocess_sales_2(sales):
            #     months_unq = sales['month'].unique().tolist()
            #     years_unq = sales['year'].unique().tolist()
            #     weeks_unq = sales['wday'].unique().tolist()
            #     # sales = sales.dropna(axis=0, subset=['x_28'])
            #     for i in years_unq:
            #         for y in months_unq:
            #             sales.loc[(sales['year'] == i) & (sales['month'] == y), 'x_28' + '_month_mean'] = \
            #                 sales.loc[(sales['month'] == y) & (sales['year'] == i)].groupby(['id'])['x_28'].transform(
            #                     lambda x: x.mean()).astype("float32")
            #             sales.loc[(sales['year'] == i) & (sales['month'] == y), 'x_28' + '_month_max'] = \
            #                 sales.loc[(sales['month'] == y) & (sales['year'] == i)].groupby(['id'])['x_28'].transform(
            #                     lambda x: x.max()).astype("float32")
            #             sales.loc[(sales['year'] == i) & (sales['month'] == y), 'x_28' + '_month_min'] = \
            #                 sales.loc[(sales['month'] == y) & (sales['year'] == i)].groupby(['id'])['x_28'].transform(
            #                     lambda x: x.min()).astype("float32")
            #             sales['x_28' + '_month_max_to_min_diff'] = (
            #                     sales['x_28' + '_month_max'] - sales['x_28' + '_month_min']).astype("float32")
            # 
            #             for z in weeks_unq:
            #                 sales.loc[(sales['year'] == i) & (sales['month'] == y) & (
            #                         sales['wday'] == z), 'x_28' + '_wk_mean'] = \
            #                     sales.loc[(sales['month'] == y) & (sales['year'] == i) & (sales['wday'] == z)].groupby(
            #                         ['id'])[
            #                         'x_28'].transform(lambda x: x.mean()).astype("float32")
            #                 sales.loc[(sales['year'] == i) & (sales['month'] == y) & (
            #                         sales['wday'] == z), 'x_28' + '_wk_median'] = \
            #                     sales.loc[(sales['month'] == y) & (sales['year'] == i) & (sales['wday'] == z)].groupby(
            #                         ['id'])[
            #                         'x_28'].transform(lambda x: x.median()).astype("float32")
            #                 sales.loc[(sales['year'] == i) & (sales['month'] == y) & (
            #                         sales['wday'] == z), 'x_28' + '_wk_max'] = \
            #                     sales.loc[(sales['month'] == y) & (sales['year'] == i) & (sales['wday'] == z)].groupby(
            #                         ['id'])[
            #                         'x_28'].transform(lambda x: x.max()).astype("float32")
            #                 sales.loc[(sales['year'] == i) & (sales['month'] == y) & (
            #                         sales['wday'] == z), 'x_28' + '_wk_min'] = \
            #                     sales.loc[(sales['month'] == y) & (sales['year'] == i) & (sales['wday'] == z)].groupby(
            #                         ['id'])[
            #                         'x_28'].transform(lambda x: x.min()).astype("float32")
            #                 sales['x_28' + '_wk_max_to_min_diff'] = (
            #                         sales['x_28' + '_wk_max'] - sales['x_28' + '_wk_min']).astype("float32")
            #     return sales
            # 
            # sales = preprocess_sales_2(sales)

            # sales_f9 = pd.read_csv("../input/sales_processed_f9.csv", index_col=0)
            # 
            # def preprocess_sales_3(df):
            #     months_unq = df['month'].unique().tolist()
            #     years_unq = df['year'].unique().tolist()
            #     weeks_unq = df['wday'].unique().tolist()
            # 
            #     for i in years_unq:
            #         for y in months_unq:
            #             df.loc[(df['year'] == i) & (df['month'] == y), 'x_28' + '_month_var'] = \
            #                 df.loc[(df['month'] == y) & (df['year'] == i)].groupby(['id'])[
            #                     'x_28'].transform(lambda x: x.var()).astype("float32")
            #             for z in weeks_unq:
            #                 df.loc[(df['year'] == i) & (df['month'] == y) & (
            #                         df['wday'] == z), 'x_28' + '_wk_var'] = df.loc[
            #                     (df['month'] == y) & (df['year'] == i) & (df['wday'] == z)].groupby(
            #                     ['id'])['x_28'].transform(lambda x: x.var()).astype("float32")
            #     return df
            # 
            # sales_f9 = preprocess_sales_3(sales_f9)

            sales_f11 = pd.read_csv("../input/sales_processed_f9.csv", index_col=0)

            def preprocess_sales_3(df):
                print('In process 3')
                months_unq = df['month'].unique().tolist()
                years_unq = df['year'].unique().tolist()

                for i in years_unq:
                    for y in months_unq:
                        df.loc[(df['year'] == i) & (df['month'] == y), 'x_28' + '_month_median'] = \
                            df.loc[(df['month'] == y) & (df['year'] == i)].groupby(['id'])[
                                'x_28'].transform(lambda x: x.median()).astype("float32")
                return df

            sales_f11 = preprocess_sales_3(sales_f11)

            # sales.to_csv("../input/sales_processed_f9.csv", index=True)
            # sales[LIST_OF_FEATURE].to_csv("../input/sales_processed_only_f9.csv", index=True)
            # sales_f9.to_csv("../input/sales_processed_f11.csv", index=True)

            sales_f11.to_csv("../input/sales_processed_f10.csv", index=True)
            print('csv file processed.')
            exit()

            # sales_f15 = pd.read_csv("../input/sales_processed_f15.csv", index_col=0)
            # sales_f15 = sales_f15.drop('x_28_month_var', axis=1)
            # sales_f15.to_csv("../input/sales_processed_f15.csv", index=True)
        else:
            print('Reading csv file...')
            sales = pd.read_csv("../input/sales_processed_f10.csv", index_col=0)
            sales = sales.loc[:, ~sales.columns.isin(
                ['x_28', 'x_30', 'x_35', 'x_42', 'x_49', 'x_56', 'x_63'])]
            # sales = pd.read_csv("../input/sales_processed.csv", index_col=0)
            sales = LoadData.reduce_mem_usage(sales)
            gc.collect()
            print('Csv file opened.')

            df_head_proc = sales.head()

            # sales['CA_w'] = sales.loc[:]['snap_CA'] * sales.loc[:]['x_28_wk_mean']
            # sales['TX_w'] = sales.loc[:]['snap_TX'] * sales.loc[:]['x_28_wk_mean']
            # sales['WI_w'] = sales.loc[:]['snap_WI'] * sales.loc[:]['x_28_wk_mean']

            self.FEATS = []
            for lag in tqdm(LAGS):
                sales[f"x_{lag}"] = sales.groupby("id")["x"].shift(lag)
                self.FEATS.append(f"x_{lag}")

            print(sales.shape)
            sales = sales.loc[sales.nb > sales.start]
            print(sales.shape)

            nb = sales['nb'].values
            MAX_LAG = max(LAGS)

            # SORRY THIS IS FAKE VALIDATION. I DIDN'T THINK IT WOULD HAVE HAD LIFTED UP MY SCORE LIKE THAT
            self.tr_mask = np.logical_and(nb > START + MAX_LAG, nb <= 1941)
            self.val_mask = np.logical_and(nb > 1913, nb <= 1941)
            self.te_mask = np.logical_and(nb > 1941, nb <= 1969)

        # print('#' * 40)
        # print("SALES:", sales.isnull().any())
        # self.scale2 = sales['scale1'].values
        self.scale = sales['scale1'].values
        self.ids = sales['id'].values
        # y = sales['demand'].values
        # ys = y / scale
        # self.ys = sales[['x', 'sales1']].values
        self.ys = sales['x'].values

        # feats_list = self.FEATS + LIST_OF_FEATURE
        self.feats_list = self.FEATS

        self.Z = sales[self.feats_list].values
        # self.Z = sales[self.FEATS].values.reshape((NITEMS, -1, len(self.FEATS)))
        print(self.scale.shape, self.ids.shape, self.ys.shape, self.Z.shape)

        self.sv = self.scale[self.val_mask]
        self.se = self.scale[self.te_mask]
        self.ids = self.ids[self.te_mask]
        self.ids = self.ids.reshape((-1, 28))

        self.ca = sales[['snap_CA']].values
        self.tx = sales[['snap_TX']].values
        self.wi = sales[['snap_WI']].values
        self.wday = sales[['wday']].values
        self.month = sales[['month']].values
        self.year = sales[['year']].values
        self.event = sales[['event_name']].values
        self.nday = sales[['nday']].values

        self.item = sales[['item_id']].values
        self.dept = sales[['dept_id']].values
        self.cat = sales[['cat_id']].values
        self.store = sales[['store_id']].values
        self.state = sales[['state_id']].values

        self.x_28_month_mean = sales[['x_28_month_mean']].values
        self.x_28_month_median = sales[['x_28_month_median']].values
        self.x_28_month_max = sales[['x_28_month_max']].values
        self.x_28_month_min = sales[['x_28_month_min']].values
        self.x_28_month_max_to_min_diff = sales[['x_28_month_max_to_min_diff']].values
        self.x_28_wk_mean = sales[['x_28_wk_mean']].values
        self.x_28_wk_median = sales[['x_28_wk_median']].values
        self.x_28_wk_max = sales[['x_28_wk_max']].values
        self.x_28_wk_min = sales[['x_28_wk_min']].values
        self.x_28_wk_max_to_min_diff = sales[['x_28_wk_max_to_min_diff']].values

    def make_data(self, mask):
        x = {'snap_CA': self.ca[mask], 'snap_TX': self.tx[mask], 'snap_WI': self.wi[mask], 'wday': self.wday[mask],
             'month': self.month[mask], 'year': self.year[mask], 'event': self.event[mask], 'nday': self.nday[mask],
             'item': self.item[mask], 'dept': self.dept[mask], 'cat': self.cat[mask], 'store': self.store[mask],
             'state': self.state[mask],
             'x_28_month_mean': self.x_28_month_mean[mask],
             'x_28_month_median': self.x_28_month_median[mask],
             'x_28_month_max': self.x_28_month_max[mask],
             'x_28_month_min': self.x_28_month_min[mask],
             'x_28_month_max_to_min_diff': self.x_28_month_max_to_min_diff[mask],
             'x_28_wk_mean': self.x_28_wk_mean[mask],
             'x_28_wk_median': self.x_28_wk_median[mask],
             'x_28_wk_max': self.x_28_wk_max[mask],
             'x_28_wk_min': self.x_28_wk_min[mask],
             'x_28_wk_max_to_min_diff': self.x_28_wk_max_to_min_diff[mask],
             'num': self.Z[mask]}

        t = self.ys[mask]
        return x, t

    def run(self):
        xt, yt = self.make_data(self.tr_mask)  # train
        xv, yv = self.make_data(self.val_mask)  # val
        xe, ye = self.make_data(self.te_mask)  # test

        return xt, yt, xv, yv, xe, ye


class ProcessDataRNN(object):
    def __init__(self, sales):
        global LAGS
        print("start process_data...")

        df_head = sales.head()

        if not PROC_CSV_EXIST:
            sales["x"] = sales["demand"] / sales["scale1"]
            self.FEATS = []
            for lag in tqdm(LAGS):
                sales[f"x_{lag}"] = sales.groupby("id")["x"].shift(lag)
                self.FEATS.append(f"x_{lag}")

            # print(sales.shape)
            # sales = sales.loc[sales.nb > sales.start]
            # print(sales.shape)

            # nb = sales['nb'].values
            # MAX_LAG = max(LAGS)

            # tr_mask = np.logical_and(nb>START + MAX_LAG, nb<=1913)
            # SORRY THIS IS FAKE VALIDATION. I DIDN'T THINK IT WOULD HAVE HAD LIFTED UP MY SCORE LIKE THAT
            # self.tr_mask = np.logical_and(nb > START + MAX_LAG, nb <= 1941)
            # self.val_mask = np.logical_and(nb > 1913, nb <= 1941)
            # self.te_mask = np.logical_and(nb > 1941, nb <= 1969)

            def preprocess_sales_2(sales):
                months_unq = sales['month'].unique().tolist()
                years_unq = sales['year'].unique().tolist()
                weeks_unq = sales['wday'].unique().tolist()
                # sales = sales.dropna(axis=0, subset=['x_28'])
                for i in years_unq:
                    for y in months_unq:
                        sales.loc[(sales['year'] == i) & (sales['month'] == y), 'x_28' + '_month_mean'] = \
                            sales.loc[(sales['month'] == y) & (sales['year'] == i)].groupby(['id'])['x_28'].transform(
                                lambda x: x.mean()).astype("float32")
                        sales.loc[(sales['year'] == i) & (sales['month'] == y), 'x_28' + '_month_max'] = \
                            sales.loc[(sales['month'] == y) & (sales['year'] == i)].groupby(['id'])['x_28'].transform(
                                lambda x: x.max()).astype("float32")
                        sales.loc[(sales['year'] == i) & (sales['month'] == y), 'x_28' + '_month_min'] = \
                            sales.loc[(sales['month'] == y) & (sales['year'] == i)].groupby(['id'])['x_28'].transform(
                                lambda x: x.min()).astype("float32")
                        sales['x_28' + '_month_max_to_min_diff'] = (
                                sales['x_28' + '_month_max'] - sales['x_28' + '_month_min']).astype("float32")

                        for z in weeks_unq:
                            sales.loc[(sales['year'] == i) & (sales['month'] == y) & (
                                    sales['wday'] == z), 'x_28' + '_wk_mean'] = \
                                sales.loc[(sales['month'] == y) & (sales['year'] == i) & (sales['wday'] == z)].groupby(
                                    ['id'])[
                                    'x_28'].transform(lambda x: x.mean()).astype("float32")
                            sales.loc[(sales['year'] == i) & (sales['month'] == y) & (
                                    sales['wday'] == z), 'x_28' + '_wk_median'] = \
                                sales.loc[(sales['month'] == y) & (sales['year'] == i) & (sales['wday'] == z)].groupby(
                                    ['id'])[
                                    'x_28'].transform(lambda x: x.median()).astype("float32")
                            sales.loc[(sales['year'] == i) & (sales['month'] == y) & (
                                    sales['wday'] == z), 'x_28' + '_wk_max'] = \
                                sales.loc[(sales['month'] == y) & (sales['year'] == i) & (sales['wday'] == z)].groupby(
                                    ['id'])[
                                    'x_28'].transform(lambda x: x.max()).astype("float32")
                            sales.loc[(sales['year'] == i) & (sales['month'] == y) & (
                                    sales['wday'] == z), 'x_28' + '_wk_min'] = \
                                sales.loc[(sales['month'] == y) & (sales['year'] == i) & (sales['wday'] == z)].groupby(
                                    ['id'])[
                                    'x_28'].transform(lambda x: x.min()).astype("float32")
                            sales['x_28' + '_wk_max_to_min_diff'] = (
                                    sales['x_28' + '_wk_max'] - sales['x_28' + '_wk_min']).astype("float32")
                return sales

            sales = preprocess_sales_2(sales)
            sales.to_csv("../input/sales_processed_f9.csv", index=True)
            sales[LIST_OF_FEATURE].to_csv("../input/sales_processed_only_f9.csv", index=True)
        else:
            sales = pd.read_csv("../input/sales_processed_f9.csv", index_col=0)
            sales = sales.loc[:, ~sales.columns.isin(
                ['x_28', 'x_30', 'x_35', 'x_42', 'x_49', 'x_56', 'x_63'])]
            # sales = pd.read_csv("../input/sales_processed.csv", index_col=0)
            sales = LoadData.reduce_mem_usage(sales)
            self.FEATS = []
            for lag in tqdm(LAGS):
                sales[f"x_{lag}"] = sales.groupby("id")["x"].shift(lag)
                self.FEATS.append(f"x_{lag}")

        # print('#' * 40)
        # print("SALES:", sales.isnull().any())
        # self.scale2 = sales['scale1'].values
        self.scale = sales['scale1'].values.reshape((NITEMS, -1))
        self.ids = sales['id'].values.reshape((NITEMS, -1))
        # y = sales['demand'].values
        # ys = y / scale
        self.ys = sales[['x', 'sales1']].values.reshape((NITEMS, -1, 2))
        # self.ys = sales['x'].values

        # feats_list = self.FEATS + LIST_OF_FEATURE
        feats_list = self.FEATS

        # arr_feat = sales[self.FEATS].values
        # arr_feature = sales[LIST_OF_FEATURE].values
        # arr_merge = np.concatenate([arr_feat, arr_feature], 1)

        # z_merge = sales[feats_list].values
        # print(np.array_equal(arr_merge, z_merge))
        # print(arr_merge == z_merge)
        # print(np.all((arr_merge == z_merge) | (np.isnan(arr_merge) & np.isnan(z_merge))))
        # print(type(arr_merge[0][0]), arr_merge[0][0])
        # print(type(z_merge[0][0]), z_merge[0][0])

        self.Z = sales[feats_list].values.reshape((NITEMS, -1, len(feats_list)))
        # self.Z = sales[self.FEATS].values.reshape((NITEMS, -1, len(self.FEATS)))
        print(self.scale.shape, self.ids.shape, self.ys.shape, self.Z.shape)

        self.sv = self.scale[:, LEN - 56:LEN - 28]
        self.se = self.scale[:, LEN - 28:LEN]

        # self.sv = self.scale[self.val_mask]
        # self.se = self.scale[self.te_mask]
        # self.ids = self.ids[self.te_mask]
        # self.ids = self.ids.reshape((-1, 28))
        #
        # self.ca = sales[['snap_CA']].values
        # self.tx = sales[['snap_TX']].values
        # self.wi = sales[['snap_WI']].values
        # self.wday = sales[['wday']].values
        # self.month = sales[['month']].values
        # self.year = sales[['year']].values
        # self.event = sales[['event_name']].values
        # self.nday = sales[['nday']].values
        #
        # self.item = sales[['item_id']].values
        # self.dept = sales[['dept_id']].values
        # self.cat = sales[['cat_id']].values
        # self.store = sales[['store_id']].values
        # self.state = sales[['state_id']].values

        self.C = sales[CATCOLS].values.reshape((NITEMS, -1, len(CATCOLS)))
        # self.C = sales[CATCOLS + LIST_OF_FEATURE].values.reshape((NITEMS, -1, len(CATCOLS + LIST_OF_FEATURE)))
        # print("C_nan:", np.argwhere(np.isnan(self.C)))
        # print("Z_nan:", np.argwhere(np.isnan(self.Z)))
        # print("ys_nan:", np.argwhere(np.isnan(self.ys)))
        # print()

    # def make_data(self, mask):
    @staticmethod
    def make_data(c, z, y):
        # x = {"snap_CA": self.ca[mask], "snap_TX": self.tx[mask], "snap_WI": self.wi[mask], "wday": self.wday[mask],
        #      "month": self.month[mask], "year": self.year[mask], "event": self.event[mask], "nday": self.nday[mask],
        #      "item": self.item[mask], "dept": self.dept[mask], "cat": self.cat[mask], "store": self.store[mask],
        #      "state": self.state[mask], "num": self.Z[mask]}
        x = {"snap_CA": c[:, :, 0], "snap_TX": c[:, :, 1], "snap_WI": c[:, :, 2], "wday": c[:, :, 3],
             "month": c[:, :, 4], "year": c[:, :, 5], "event": c[:, :, 6], "nday": c[:, :, 7],
             "item": c[:, :, 8], "dept": c[:, :, 9], "cat": c[:, :, 10], "store": c[:, :, 11],
             "state": c[:, :, 12], "num": z}
        # "x_28_month_mean": c[:, :, 13], "x_28_month_max": c[:, :, 14],
        # "x_28_month_min": c[:, :, 15], "x_28_month_max_to_min_diff": c[:, :, 16],
        # 'x_28_wk_mean': c[:, :, 17], 'x_28_wk_median': c[:, :, 18], 'x_28_wk_max': c[:, :, 19],
        # 'x_28_wk_min': c[:, :, 20], 'x_28_wk_max_to_min_diff': c[:, :, 21],

        # t = self.ys[mask]
        t = y
        return x, t

    def run(self):
        # xt, yt = self.make_data(self.tr_mask)  # train
        # xv, yv = self.make_data(self.val_mask)  # val
        # xe, ye = self.make_data(self.te_mask)  # test

        # xt, yt = ProcessData.make_data(self.C[:, MAX_LAG:LEN - 56, :], self.Z[:, MAX_LAG:LEN - 56, :],
        #                         self.ys[:, MAX_LAG:LEN - 56])  # train
        xv, yv = ProcessDataRNN.make_data(self.C[:, LEN - 56:LEN - 28, :], self.Z[:, LEN - 56:LEN - 28, :],
                                          self.ys[:, LEN - 56:LEN - 28])  # val

        # xt, yt = ProcessData.make_data(self.C[:, MAX_LAG:LEN - 28, :], self.Z[:, MAX_LAG:LEN - 28, :],
        #                         self.ys[:, MAX_LAG:LEN - 28])  # train
        xe, ye = ProcessDataRNN.make_data(self.C[:, LEN - 28:LEN, :], self.Z[:, LEN - 28:LEN, :],
                                          self.ys[:, LEN - 28:LEN])  # test

        # return xt, yt, xv, yv, xe, ye
        return xv, yv, xe, ye
        # return xt, yt, xe, ye


class DataGenerator(tf.keras.utils.Sequence):
    # 'Generates data for Keras'''

    def __init__(self, x, brks, batch_size=32, shuffle=True):
        # 'Initialization'
        self.batch_size = batch_size
        self.c = x[0]
        self.z = x[1]
        self.y = x[2]
        self.brks = brks.copy()
        self.list_IDs = np.array(range(42840))
        self.shuffle = shuffle
        self.nb_batch = int(np.ceil(len(self.list_IDs) / self.batch_size))
        self.n_windows = brks.shape[0]
        self.on_epoch_end()
        # B+H <= idx <= LEN - 28

    def __len__(self):
        # 'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs) / self.batch_size)) * (self.n_windows)

    def __getitem__(self, index):
        # 'Generate one batch of data'
        # Generate indexes of the batch
        idx, kx = self.ids[index]
        batch_ids = self.list_IDs[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(batch_ids, kx)

        return X, y

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.ids = list(it.product(np.arange(0, self.nb_batch), self.brks))
        if self.shuffle == True:
            np.random.shuffle(self.ids)

    def __data_generation(self, batch_ids, kx):
        # 'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        x_batch, y_batch = ProcessDataRNN.make_data(self.c[batch_ids, kx - 28:kx], self.z[batch_ids, kx - 28:kx],
                                                    self.y[batch_ids, kx - 28:kx])
        return x_batch, y_batch

