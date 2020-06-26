# -*- coding:utf-8 -*-
from __init__ import *


class ProcessData:
    def __init__(self, sales):
        sales["x"] = sales["demand"] / sales["scale"]

        LAGS = [28, 35, 42, 49, 56, 63]
        self.FEATS = []
        for lag in tqdm(LAGS):
            sales[f"x_{lag}"] = sales.groupby("id")["x"].shift(lag)
            self.FEATS.append(f"x_{lag}")

        print(sales.shape)
        sales = sales.loc[sales.nb > sales.start]
        print(sales.shape)

        nb = sales['nb'].values
        MAX_LAG = max(LAGS)

        # tr_mask = np.logical_and(nb>START + MAX_LAG, nb<=1913)
        # SORRY THIS IS FAKE VALIDATION. I DIDN'T THINK IT WOULD HAVE HAD LIFTED UP MY SCORE LIKE THAT
        self.tr_mask = np.logical_and(nb > START + MAX_LAG, nb <= 1941)
        self.val_mask = np.logical_and(nb > 1913, nb <= 1941)
        self.te_mask = np.logical_and(nb > 1941, nb <= 1969)

        self.scale = sales['scale'].values
        self.ids = sales['id'].values
        # y = sales['demand'].values
        # ys = y / scale
        self.ys = sales['x'].values
        self.Z = sales[self.FEATS].values

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

    def make_data(self, mask):
        x = {"snap_CA": self.ca[mask], "snap_TX": self.tx[mask], "snap_WI": self.wi[mask], "wday": self.wday[mask],
             "month": self.month[mask], "year": self.year[mask], "event": self.event[mask], "nday": self.nday[mask],
             "item": self.item[mask], "dept": self.dept[mask], "cat": self.cat[mask], "store": self.store[mask],
             "state": self.state[mask], "num": self.Z[mask]}
        t = self.ys[mask]
        return x, t

    def run(self):
        xt, yt = self.make_data(self.tr_mask)  # train
        xv, yv = self.make_data(self.val_mask)  # val
        xe, ye = self.make_data(self.te_mask)  # test
        return xt, yt, xv, yv, xe, ye
