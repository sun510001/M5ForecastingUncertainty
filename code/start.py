# -*- coding:utf-8 -*-
from __init__ import *
from process_data import ProcessDataCNN, ProcessDataRNN, DataGenerator
from load_data import LoadData
from model import make_model_cnn, make_model_rnn


def start_cnn():
    load_data = LoadData()
    process_data = ProcessDataCNN(sales=load_data.make_dataset(categorize=CATEGORIZE, start=START, upper=UPPER))
    xt, yt, xv, yv, xe, ye = process_data.run()

    if not ONLY_LOAD_MODEL:
        ckpt = ModelCheckpoint(OUTPUT_MODEL, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=MIN_LR, verbose=1)
        es = EarlyStopping(monitor='val_loss', patience=3)
        net = make_model_cnn(len(process_data.FEATS))
        plot_model(net, to_file='model.png')
        print(net.summary())
        # exit()

        net.fit(xt, yt, batch_size=BATCH_SIZE, epochs=EPOCH, validation_data=(xv, yv), callbacks=[reduce_lr, ckpt, es])

    nett = make_model_cnn(len(process_data.feats_list))
    nett.load_weights(OUTPUT_MODEL)

    pv = nett.predict(xv, batch_size=BATCH_SIZE, verbose=1)
    pe = nett.predict(xe, batch_size=BATCH_SIZE, verbose=1)
    print("Eva result:", nett.evaluate(xv, yv, batch_size=BATCH_SIZE))

    pv = pv.reshape((-1, 28, 9))
    pe = pe.reshape((-1, 28, 9))
    sv = process_data.sv.reshape((-1, 28))
    se = process_data.se.reshape((-1, 28))
    Yv = yv.reshape((-1, 28))

    return process_data, Yv, pv, pe, sv, se


def start_rnn():
    load_data = LoadData()
    process_data = ProcessDataRNN(sales=load_data.make_dataset(categorize=CATEGORIZE, start=START, upper=UPPER))
    # xt, yt, xv, yv, xe, ye = process_data.run()
    xv, yv, xe, ye = process_data.run()

    if not ONLY_LOAD_MODEL:
        ckpt = ModelCheckpoint(OUTPUT_MODEL, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=MIN_LR, verbose=1)
        es = EarlyStopping(monitor='val_loss', patience=3)
        # net = make_model(len(process_data.FEATS))
        net = make_model_rnn(process_data.Z.shape[2])
        print(net.summary())

        n_slices = LEN // 28
        brks = np.array([LEN - (n_slices - i) * 28 for i in range(n_slices + 1)])
        brks = brks[brks >= max(LAGS) + 28]
        print("#" * 30)
        print(LEN, process_data.C.shape, process_data.Z.shape)
        print(brks)
        print(process_data.C.min(), process_data.ys.min(), process_data.Z[:, 66:].min())
        print("#" * 30)
        net.fit_generator(
            DataGenerator((process_data.C, process_data.Z, process_data.ys), brks[:-1], batch_size=BATCH_SIZE),
            epochs=EPOCH, validation_data=(xv, yv), callbacks=[ckpt, reduce_lr, es])

        # net.fit(xt, yt, batch_size=BATCH_SIZE, epochs=EPOCH, validation_data=(xv, yv), callbacks=[ckpt, reduce_lr, es])

    # nett = make_model(len(process_data.FEATS))
    nett = make_model_rnn(process_data.Z.shape[2])
    nett.load_weights(OUTPUT_MODEL)

    pv = nett.predict(xv, batch_size=BATCH_SIZE, verbose=1)
    pe = nett.predict(xe, batch_size=BATCH_SIZE, verbose=1)
    print("Eva result:", nett.evaluate(xv, yv, batch_size=BATCH_SIZE))

    # pv = pv.reshape((-1, 28, 9))
    # pe = pe.reshape((-1, 28, 9))
    sv = process_data.sv.reshape((-1, 28))
    se = process_data.se.reshape((-1, 28))
    # Yv = yv.reshape((-1, 28))
    return process_data, yv, pv, pe, sv, se


if __name__ == '__main__':
    # process_data, yv, pv, pe, sv, se = start_rnn()
    process_data, yv, pv, pe, sv, se = start_cnn()
    for i in range(5):
        k = np.random.randint(0, 42840)
        # k = np.random.randint(0, 200)
        print(process_data.ids[k, 0])
        plt.plot(np.arange(28, 56), yv[k], label="true")
        plt.plot(np.arange(28, 56), pv[k, :, 3], label="q25")
        plt.plot(np.arange(28, 56), pv[k, :, 4], label="q50")
        plt.plot(np.arange(28, 56), pv[k, :, 5], label="q75")
        # plt.plot(np.arange(28, 56), pv[k, :, 8], label="q99.5")
        # plt.plot(np.arange(0, 28), process_data.ys[k, -56:-28, 0], label="past")
        plt.legend(loc="best")
        # plt.show()
        plt.savefig(OUTPUT_IMAGE.format(VER, i))
        plt.clf()

    names = [f"F{i + 1}" for i in range(28)]
    piv = pd.DataFrame(process_data.ids[:, 0], columns=["id"])

    for i, quantile in tqdm(enumerate(QUANTILES)):
        t1 = pd.DataFrame(pv[:, :, i] * sv, columns=names)
        t1 = piv.join(t1)
        t1["id"] = t1["id"] + f"_{quantile}_validation"
        t2 = pd.DataFrame(pe[:, :, i] * se, columns=names)
        t2 = piv.join(t2)
        t2["id"] = t2["id"] + f"_{quantile}_evaluation"
        VALID.append(t1)
        EVAL.append(t2)

    sub = pd.DataFrame()
    sub = sub.append(VALID + EVAL)
    del VALID, EVAL, t1, t2

    print(sub.head())
    sub.to_csv("submission_{}.csv".format(VER), index=False)
