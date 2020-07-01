# -*- coding:utf-8 -*-
from __init__ import *


def wqloss(y_true, y_pred):
    # Pinball loss for multiple quantiles
    qs = [0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true[:, :, :1] - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    psl = tf.reduce_mean(v, axis=[1, 2])
    weights = y_true[:, 0, 1] / K.sum(y_true[:, 0, 1])
    return tf.reduce_sum(psl * weights)


def qloss(y_true, y_pred):
    # Pinball loss for multiple quantiles
    qs = [0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    return K.mean(v)


def make_model_cnn(n_in):
    num = L.Input((n_in,), name="num")
    ca = L.Input((1,), name="snap_CA")
    tx = L.Input((1,), name="snap_TX")
    wi = L.Input((1,), name="snap_WI")
    wday = L.Input((1,), name="wday")
    month = L.Input((1,), name="month")
    year = L.Input((1,), name="year")
    event = L.Input((1,), name="event")
    nday = L.Input((1,), name="nday")
    item = L.Input((1,), name="item")
    dept = L.Input((1,), name="dept")
    cat = L.Input((1,), name="cat")
    store = L.Input((1,), name="store")
    state = L.Input((1,), name="state")

    x_28_month_mean = L.Input((1,), name="x_28_month_mean")
    x_28_month_median = L.Input((1,), name="x_28_month_median")
    x_28_month_max = L.Input((1,), name="x_28_month_max")
    x_28_month_min = L.Input((1,), name="x_28_month_min")
    x_28_month_max_to_min_diff = L.Input((1,), name="x_28_month_max_to_min_diff")

    x_28_wk_mean = L.Input((1,), name="x_28_wk_mean")
    x_28_wk_median = L.Input((1,), name="x_28_wk_median")
    x_28_wk_max = L.Input((1,), name="x_28_wk_max")
    x_28_wk_min = L.Input((1,), name="x_28_wk_min")
    x_28_wk_max_to_min_diff = L.Input((1,), name="x_28_wk_max_to_min_diff")

    # ca_w = L.Input((1,), name='CA_w')
    # tx_w = L.Input((1,), name='TX_w')
    # wi_w = L.Input((1,), name='WI_w')

    # inp = {"snap_CA": ca, "snap_TX": tx, "snap_WI": wi, "wday": wday, "month": month, "year": year, "event": event,
    #        "nday": nday, "item": item, "dept": dept, "cat": cat, "store": store, "state": state,
    #        "x_28_month_mean": x_28_month_mean, "x_28_month_max": x_28_month_max,
    #        "x_28_month_min": x_28_month_min, "x_28_month_max_to_min_diff": x_28_month_max_to_min_diff,
    #        'x_28_wk_mean': x_28_wk_mean, 'x_28_wk_median': x_28_wk_median, 'x_28_wk_max': x_28_wk_max,
    #        'x_28_wk_min': x_28_wk_min, 'x_28_wk_max_to_min_diff': x_28_wk_max_to_min_diff,
    #        "num": num}
    inp = {"snap_CA": ca, "snap_TX": tx, "snap_WI": wi, "wday": wday, "month": month, "year": year, "event": event,
           "nday": nday, "item": item, "dept": dept, "cat": cat, "store": store, "state": state,
           "x_28_month_mean": x_28_month_mean,
           'x_28_month_median': x_28_month_median,
           "x_28_month_max": x_28_month_max,
           "x_28_month_min": x_28_month_min,
           "x_28_month_max_to_min_diff": x_28_month_max_to_min_diff,
           'x_28_wk_mean': x_28_wk_mean,
           'x_28_wk_median': x_28_wk_median,
           'x_28_wk_max': x_28_wk_max,
           'x_28_wk_min': x_28_wk_min,
           'x_28_wk_max_to_min_diff': x_28_wk_max_to_min_diff,
           "num": num}

    ca_ = L.Embedding(mods["snap_CA"], mods["snap_CA"], name="ca_3d")(ca)
    tx_ = L.Embedding(mods["snap_TX"], mods["snap_TX"], name="tx_3d")(tx)
    wi_ = L.Embedding(mods["snap_WI"], mods["snap_WI"], name="wi_3d")(wi)
    wday_ = L.Embedding(mods["wday"], mods["wday"], name="wday_3d")(wday)
    month_ = L.Embedding(mods["month"], mods["month"], name="month_3d")(month)
    year_ = L.Embedding(mods["year"], mods["year"], name="year_3d")(year)
    event_ = L.Embedding(mods["event_name"], mods["event_name"], name="event_3d")(event)
    nday_ = L.Embedding(mods["nday"], mods["nday"], name="nday_3d")(nday)
    item_ = L.Embedding(mods["item_id"], 10, name="item_3d")(item)
    dept_ = L.Embedding(mods["dept_id"], mods["dept_id"], name="dept_3d")(dept)
    cat_ = L.Embedding(mods["cat_id"], mods["cat_id"], name="cat_3d")(cat)
    store_ = L.Embedding(mods["store_id"], mods["store_id"], name="store_3d")(store)
    state_ = L.Embedding(mods["state_id"], mods["state_id"], name="state_3d")(state)

    p = [ca_, tx_, wi_, wday_, month_, year_, event_, nday_, item_, dept_, cat_, store_, state_]

    emb = L.Concatenate(name="embds")(p)
    context = L.Flatten(name="context")(emb)

    x = L.Concatenate(name="x1")(
        [context, num, x_28_month_mean,
         x_28_month_max, x_28_month_min,
         x_28_month_max_to_min_diff,
         x_28_wk_mean,
         x_28_wk_median,
         x_28_wk_max,
         x_28_wk_min,
         x_28_wk_max_to_min_diff])
    x = L.Dense(840, activation='relu', name="d1")(x)  # original nodes: 500; ori act: relu
    x = L.Dropout(0.25)(x)
    x = L.Concatenate(name="m1")([x, context])
    x = L.Dense(840, activation='relu', name="d2")(x)
    x = L.Dropout(0.25)(x)
    x = L.Concatenate(name="m2")([x, context])
    x = L.Dense(840, activation='relu', name="d3")(x)
    x = L.Dropout(0.25)(x)
    x = L.Concatenate(name="m3")([x, context])
    x = L.Dense(840, activation='relu', name="d4")(x)

    preds = L.Dense(9, activation="linear", name="preds")(x)
    model = M.Model(inp, preds, name="M1")
    model.compile(loss=qloss, optimizer="adam")

    return model


def make_model_rnn(n_in):
    seq_len = 28
    num = L.Input((seq_len, n_in,), name="num")

    ca = L.Input((seq_len,), name="snap_CA")
    tx = L.Input((seq_len,), name="snap_TX")
    wi = L.Input((seq_len,), name="snap_WI")
    wday = L.Input((seq_len,), name="wday")
    month = L.Input((seq_len,), name="month")
    year = L.Input((seq_len,), name="year")
    event = L.Input((seq_len,), name="event")
    nday = L.Input((seq_len,), name="nday")
    item = L.Input((seq_len,), name="item")
    dept = L.Input((seq_len,), name="dept")
    cat = L.Input((seq_len,), name="cat")
    store = L.Input((seq_len,), name="store")
    state = L.Input((seq_len,), name="state")

    # x_28_month_mean = L.Input((seq_len,), name="x_28_month_mean")
    # x_28_month_max = L.Input((seq_len,), name="x_28_month_max")
    # x_28_month_min = L.Input((seq_len,), name="x_28_month_min")
    # x_28_month_max_to_min_diff = L.Input((seq_len,), name="x_28_month_max_to_min_diff")
    # x_28_wk_mean = L.Input((seq_len,), name="x_28_wk_mean")
    # x_28_wk_median = L.Input((seq_len,), name="x_28_wk_median")
    # x_28_wk_max = L.Input((seq_len,), name="x_28_wk_max")
    # x_28_wk_min = L.Input((seq_len,), name="x_28_wk_min")
    # x_28_wk_max_to_min_diff = L.Input((seq_len,), name="x_28_wk_max_to_min_diff")

    inp = {"snap_CA": ca, "snap_TX": tx, "snap_WI": wi, "wday": wday,
           "month": month, "year": year, "event": event, "nday": nday,
           "item": item, "dept": dept, "cat": cat, "store": store, "state": state, "num": num}

    # "x_28_month_mean": x_28_month_mean, "x_28_month_max": x_28_month_max,
    # "x_28_month_min": x_28_month_min, "x_28_month_max_to_min_diff": x_28_month_max_to_min_diff,
    # 'x_28_wk_mean': x_28_wk_mean, 'x_28_wk_median': x_28_wk_median, 'x_28_wk_max': x_28_wk_max,
    # 'x_28_wk_min': x_28_wk_min, 'x_28_wk_max_to_min_diff': x_28_wk_max_to_min_diff,

    ca_ = L.Embedding(mods["snap_CA"], mods["snap_CA"], name="ca_3d")(ca)
    tx_ = L.Embedding(mods["snap_TX"], mods["snap_TX"], name="tx_3d")(tx)
    wi_ = L.Embedding(mods["snap_WI"], mods["snap_WI"], name="wi_3d")(wi)
    wday_ = L.Embedding(mods["wday"], mods["wday"], name="wday_3d")(wday)
    month_ = L.Embedding(mods["month"], mods["month"], name="month_3d")(month)
    year_ = L.Embedding(mods["year"], mods["year"], name="year_3d")(year)
    event_ = L.Embedding(mods["event_name"], mods["event_name"], name="event_3d")(event)
    nday_ = L.Embedding(mods["nday"], mods["nday"], name="nday_3d")(nday)
    item_ = L.Embedding(mods["item_id"], 10, name="item_3d")(item)
    dept_ = L.Embedding(mods["dept_id"], mods["dept_id"], name="dept_3d")(dept)
    cat_ = L.Embedding(mods["cat_id"], mods["cat_id"], name="cat_3d")(cat)
    store_ = L.Embedding(mods["store_id"], mods["store_id"], name="store_3d")(store)
    state_ = L.Embedding(mods["state_id"], mods["state_id"], name="state_3d")(state)
    # x_28_month_mean_ = L.Embedding(mods["x_28_month_mean"], mods["x_28_month_mean"], name="state_3d")(state)

    p = [ca_, tx_, wi_, wday_, month_, year_, event_, nday_, item_, dept_, cat_, store_, state_]
    context = L.Concatenate(name="context")(p)

    x = L.Concatenate(name="x1")([context, num])
    x = L.Bidirectional(L.LSTM(128, return_sequences=True, name="d1"))(x)
    x = L.Dropout(0.2)(x)
    x = L.Concatenate(name="m1")([x, context])
    x = L.Bidirectional(L.LSTM(128, return_sequences=True, name="d2"))(x)
    x = L.Dropout(0.2)(x)
    x = L.Concatenate(name="m2")([x, context])
    x = L.Bidirectional(L.LSTM(128, return_sequences=True, name="d3"))(x)
    preds = L.Dense(9, activation="linear", name="preds")(x)
    model = M.Model(inp, preds, name="M1")
    model.compile(loss=wqloss, optimizer="adam")
    return model
# dropout=0.2, recurrent_dropout=0.2,
