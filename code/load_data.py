# -*- coding:utf-8 -*-
from __init__ import *


class LoadData(object):
    @staticmethod
    def reduce_mem_usage(df, verbose=True):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024 ** 2
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        end_mem = df.memory_usage().sum() / 1024 ** 2
        if verbose:
            print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                    start_mem - end_mem) / start_mem))
        return df

    # def autocorrelation(ys, t=1):
    #     return np.corrcoef(ys[:-t], ys[t:])

    def preprocess_calendar(self, calendar):
        global maps, mods
        calendar["event_name"] = calendar["event_name_1"]
        calendar["event_type"] = calendar["event_type_1"]

        map1 = {mod: i for i, mod in enumerate(calendar['event_name'].unique())}
        calendar['event_name'] = calendar['event_name'].map(map1)
        map2 = {mod: i for i, mod in enumerate(calendar['event_type'].unique())}
        calendar['event_type'] = calendar['event_type'].map(map2)
        calendar['nday'] = calendar['date'].str[-2:].astype(int)
        maps["event_name"] = map1
        maps["event_type"] = map2
        mods["event_name"] = len(map1)
        mods["event_type"] = len(map2)
        calendar["wday"] -= 1
        calendar["month"] -= 1
        calendar["year"] -= 2011
        mods["month"] = 12
        mods["year"] = 6
        mods["wday"] = 7
        mods['snap_CA'] = 2
        mods['snap_TX'] = 2
        mods['snap_WI'] = 2
        # mods['x_28_month_mean'] = 1
        # mods['x_28_month_max'] = 1
        # mods['x_28_month_min'] = 1
        # mods['x_28_month_max_to_min_diff'] = 1
        # mods['x_28_wk_mean'] = 1
        # mods['x_28_wk_median'] = 1
        # mods['x_28_wk_max'] = 1
        # mods['x_28_wk_min'] = 1
        # mods['x_28_wk_max_to_min_diff'] = 1

        calendar.drop(["event_name_1", "event_name_2", "event_type_1", "event_type_2", "date", "weekday"],
                      axis=1, inplace=True)
        return calendar

    def preprocess_sales(self, sales, start=1400, upper=1970):
        if start is not None:
            print("dropping...")
            to_drop = [f"d_{i + 1}" for i in range(start - 1)]
            print(sales.shape)
            sales.drop(to_drop, axis=1, inplace=True)
            print(sales.shape)

        print("adding...")
        new_columns = ['d_%i' % i for i in range(1942, upper, 1)]
        for col in new_columns:
            sales[col] = np.nan
        print("melting...")
        sales = sales.melt(
            id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "scale1", "sales1", "start",
                     "sales2", "scale2"],
            var_name='d', value_name='demand')

        print("generating order")
        if start is not None:
            skip = start
        else:
            skip = 1
        sales["nb"] = sales.index // 42840 + skip
        return sales

    def make_dataset(self, categorize=False, start=1400, upper=1970):
        global maps, mods
        print("loading calendar...")
        calendar = pd.read_csv("../input/m5-forecasting-uncertainty/calendar.csv")
        print("loading sales...")
        sales = pd.read_csv("../input/walmartadd/sales_aug.csv")
        # sales = sales.sample(n=2)
        cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
        if categorize:
            for col in cols:
                temp_dct = {mod: i for i, mod in enumerate(sales[col].unique())}
                mods[col] = len(temp_dct)
                maps[col] = temp_dct
            for col in cols:
                sales[col] = sales[col].map(maps[col])

        sales = self.preprocess_sales(sales, start=start, upper=upper)
        calendar = self.preprocess_calendar(calendar)
        calendar = LoadData.reduce_mem_usage(calendar)
        print("merge with calendar...")
        sales = sales.merge(calendar, on='d', how='left')
        del calendar

        print("reordering...")
        sales.sort_values(by=["id", "nb"], inplace=True)
        print("re-indexing..")
        sales.reset_index(inplace=True, drop=True)
        gc.collect()

        sales['n_week'] = (sales['nb'] - 1) // 7
        sales["nday"] -= 1
        mods['nday'] = 31
        sales = LoadData.reduce_mem_usage(sales)
        gc.collect()
        return sales


if __name__ == '__main__':
    load = LoadData()
    sales = load.make_dataset(categorize=CATEGORIZE, start=START, upper=UPPER)
    print()
