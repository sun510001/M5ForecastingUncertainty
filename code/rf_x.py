# -*- coding:utf-8 -*-

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


sales_1 = pd.read_csv("../input/sales_processed_f9.csv", index_col=0)
sales_1.sample(frac=0.3, replace=True, random_state=1)

LIST_OF_FEATURE_2 = ["x_28_month_mean", "x_28_month_max", "x_28_month_min", "x_28_month_max_to_min_diff",
                   'x_28_wk_mean', 'x_28_wk_median', 'x_28_wk_max', 'x_28_wk_min',
                   'x_28_wk_max_to_min_diff', 'scale1']

sales_1 = sales_1.dropna()
rf = RandomForestRegressor()
rf.fit(sales_1[LIST_OF_FEATURE_2], sales_1['x'])
f_i = pd.DataFrame(data=rf.feature_importances_, index=sales_1[LIST_OF_FEATURE_2].columns, columns=['score'])
f_i = f_i.sort_values(by=['score'], ascending=False)
print(f_i)
print()