# -*- coding:utf-8 -*-
from __init__ import *
from process_data import ProcessData
from load_data import LoadData
from model import make_model

load_data = LoadData()
process_data = ProcessData(sales=load_data.make_dataset(categorize=CATEGORIZE, start=START, upper=UPPER))
xt, yt, xv, yv, xe, ye = process_data.run()
net = make_model(len(process_data.FEATS))

ckpt = ModelCheckpoint(OUTPUT_MODEL, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=1e-4)  # original min_lr=0.001
es = EarlyStopping(monitor='val_loss', patience=3)
print(net.summary())

batch_size = 12_000  # batch_size=50_000
net.fit(xt, yt, batch_size=batch_size, epochs=20, validation_data=(xv, yv),
        callbacks=[ckpt, reduce_lr, es])  # ori epochs=20

nett = make_model(len(process_data.FEATS))
nett.load_weights(OUTPUT_MODEL)

pv = nett.predict(xv, batch_size=batch_size, verbose=1)  # ori batchsize=50_000
pe = nett.predict(xe, batch_size=batch_size, verbose=1)
nett.evaluate(xv, yv, batch_size=batch_size)  # ori batchsize=50_000

pv = pv.reshape((-1, 28, 9))
pe = pe.reshape((-1, 28, 9))
sv = process_data.sv.reshape((-1, 28))
se = process_data.se.reshape((-1, 28))
Yv = yv.reshape((-1, 28))

k = np.random.randint(0, 42840)
# k = np.random.randint(0, 200)
print(process_data.ids[k, 0])
plt.plot(np.arange(28, 56), Yv[k], label="true")
plt.plot(np.arange(28, 56), pv[k, :, 3], label="q25")
plt.plot(np.arange(28, 56), pv[k, :, 4], label="q50")
plt.plot(np.arange(28, 56), pv[k, :, 5], label="q75")
plt.legend(loc="best")
plt.savefig(OUTPUT_IMAGE)

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

sub.head()
sub.to_csv("submission.csv", index=False)
