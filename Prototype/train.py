"""
train_fullspec_transformer_v3.py   —   versión corregida
========================================================
• Dataset balanceado 1 limpio + 5 “light” + 5 “heavy” por modelo
• Modelo Transformer grande (128-8-4, FF-512) + atención
• CE con label_smoothing 0.1, ReduceLROnPlateau (+ early-stop)
• Inferencia:
      – top-10 candidatos
      – se elige el de mayor similitud de edición
      – compuertas:   similarity ≥ 0.80   y   prob ≥ 0.90
• Esta versión corrige el bug del cálculo de `similarity`
"""

# ───────────── Imports ──────────────────────────────────────────────
import os, random, json, numpy as np, psycopg2, torch, onnx, onnxruntime as ort
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from collections import OrderedDict, defaultdict
from torch.utils.data import Dataset, DataLoader
from difflib import SequenceMatcher
import warnings; warnings.filterwarnings("ignore", category=UserWarning,
    message="Exporting a model to ONNX")

# ───────────── Config ───────────────────────────────────────────────
DB = dict(dbname=os.getenv("PG_DB","hvac"),
          user=os.getenv("PG_USER","postgres"),
          password=os.getenv("PG_PW","supersecret"),
          host=os.getenv("PG_HOST","localhost"),
          port=os.getenv("PG_PORT","5432"))

OUT_DIR, ONNX_FILE = "../AI-Models", "hvac_fullspec.onnx"

MAXLEN           = 20
EPOCHS           = 400
PATIENCE_ES      = 40
BATCH            = 64

EMB_DIM          = 128
N_HEADS          = 8
N_LAYERS         = 4
HID_FF           = 512

TEMP             = 0.55
PROB_TH          = 0.5
SIM_TH           = 0.75
TOP_K            = 10

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ───────────── 1 · Leer tabla SQL ───────────────────────────────────
def fetch_rows():
    q = "SELECT * FROM hvac.spec_model;"
    with psycopg2.connect(**DB) as conn, conn.cursor() as cur:
        cur.execute(q); cols = [c.name for c in cur.description]
        rows = []
        for tup in cur.fetchall():
            row = OrderedDict(zip(cols, tup))
            rows.append((row["model_code"], row))
        return rows

# ───────────── 2 · Aumentación OCR ──────────────────────────────────
CONF = [
    ('0','O'),('O','0'),('1','I'),('I','1'),('5','S'),('S','5'),
    ('8','B'),('B','8'),('G','H'),('H','G'),('G','C'),('C','G'),
    ('P','B'),('B','P'),('6','G'),('G','6')
]
def add_noise(txt: str, light=False) -> str:
    s = list(txt)

    # flip de 1-2 may/min (30 %)
    if random.random() < .30:
        for _ in range(random.randint(1, 2)):
            i = random.randint(0, len(s)-1)
            s[i] = s[i].swapcase()

    # confusiones OCR
    for _ in range(random.randint(1, 2 if light else 3)):
        i = random.randint(0, len(s)-1)
        for a, b in CONF:
            if s[i] == a and random.random() < (.60 if light else .55):
                s[i] = b

    if not light:
        # swaps
        for _ in range(random.randint(0, 2)):
            if len(s) > 2 and random.random() < .25:
                i = random.randint(0, len(s)-2)
                s[i], s[i+1] = s[i+1], s[i]
        # duplicar
        if random.random() < .10:
            i = random.randint(0, len(s)-1); s.insert(i, s[i])
        # guion / espacio
        if random.random() < .15 and len(s) > 3:
            s.insert(random.randint(1, len(s)-1), random.choice([' ', '-']))

    # mezcla global de caso
    s = [c.upper() if random.random() < .5 else c.lower() for c in s]
    return ''.join(s)

def augment(base):
    out = []
    for code, row in base:
        out.append((code, code))                          # 1 limpio
        for _ in range(5): out.append((add_noise(code, True),  code))
        for _ in range(5): out.append((add_noise(code, False), code))
    return out

# ───────────── 3 · Vocab y splits ──────────────────────────────────
def build_vocab(samples):
    ch = {c for t,_ in samples for c in t}
    v  = {c:i+1 for i,c in enumerate(sorted(ch))}; v['<PAD>'] = 0
    return v
def encode(t, v): return [v.get(c,0) for c in t[:MAXLEN]] + [0]*(MAXLEN-len(t))
def make_maps(samples):
    codes = sorted({c for _,c in samples})
    m2i   = {c:i for i,c in enumerate(codes)}
    i2m   = {i:c for c,i in m2i.items()}
    return m2i, i2m
class TxtDS(Dataset):
    def __init__(self, samples, v, m2i):
        self.data = [(encode(t,v), m2i[c]) for t,c in samples]
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        x, y = self.data[i]; return torch.tensor(x), torch.tensor(y)
def strat_split(samples, r=.85):
    b = defaultdict(list)
    for t,c in samples: b[c].append((t,c))
    tr, va = [], []
    for lst in b.values():
        random.shuffle(lst); k = max(1, int(len(lst)*r))
        tr += lst[:k]; va += lst[k:]
    random.shuffle(tr); random.shuffle(va)
    return tr, va

# ───────────── 4 · Modelo Transformer ───────────────────────────────
class PosEnc(nn.Module):
    def __init__(self, D, L):
        super().__init__()
        pe = torch.zeros(L,D)
        pos = torch.arange(0,L).unsqueeze(1)
        div = torch.exp(torch.arange(0,D,2)*(-np.log(10000.0)/D))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self,x): return x + self.pe[:,:x.size(1)]

class Net(nn.Module):
    def __init__(self, V, C):
        super().__init__()
        self.emb = nn.Embedding(V, EMB_DIM, padding_idx=0)
        self.pos = PosEnc(EMB_DIM, MAXLEN)
        layer = nn.TransformerEncoderLayer(
            EMB_DIM, N_HEADS, HID_FF, batch_first=True, activation="gelu")
        self.enc  = nn.TransformerEncoder(layer, N_LAYERS)
        self.att  = nn.Linear(EMB_DIM, 1)
        self.fc   = nn.Linear(EMB_DIM, C)
    def forward(self,x):
        mask = (x==0)
        h = self.enc(self.pos(self.emb(x)), src_key_padding_mask=mask)
        α = torch.softmax(self.att(h).masked_fill(mask.unsqueeze(-1), -1e4), dim=1)
        return self.fc((h*α).sum(1))

# ───────────── 5 · Entrenamiento ───────────────────────────────────
def train(net, tr, val, dev):
    opt = optim.Adam(net.parameters(), 1e-3)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=10)
    ce  = nn.CrossEntropyLoss(label_smoothing=0.1)
    best = bad = 0
    for ep in range(1, EPOCHS+1):
        net.train(); tot = cor = 0
        for X,y in tr:
            X,y = X.to(dev), y.to(dev)
            opt.zero_grad(); ce(net(X), y).backward(); opt.step()
            cor += (net(X).argmax(1)==y).sum().item(); tot += y.size(0)
        acc = cor/tot
        net.eval(); vtot=vcor=0; vloss=0.0
        with torch.no_grad():
            for X,y in val:
                X,y = X.to(dev), y.to(dev); out = net(X)
                vloss += ce(out,y).item()*y.size(0)
                vcor  += (out.argmax(1)==y).sum().item(); vtot += y.size(0)
        vacc = vcor/vtot; vloss /= vtot
        sch.step(vloss)
        print(f"E{ep:03d} acc {acc:.2%} val {vacc:.2%}  lr {opt.param_groups[0]['lr']:.1e}")
        if vacc > best: best=vacc; torch.save(net.state_dict(),"best.pt"); bad=0
        else: bad += 1
        if bad >= PATIENCE_ES: print("Early stop!"); break
    net.load_state_dict(torch.load("best.pt")); print("Best val:",best)

# ───────────── MAIN ────────────────────────────────────────────────
if __name__ == "__main__":
    base   = fetch_rows()
    samples = augment(base)
    tr_s, val_s = strat_split(samples)
    vocab = build_vocab(samples)
    m2i, i2m = make_maps(samples)

    train_dl = DataLoader(TxtDS(tr_s, vocab, m2i), BATCH, shuffle=True)
    val_dl   = DataLoader(TxtDS(val_s, vocab, m2i), BATCH*2)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net(len(vocab), len(m2i)).to(dev)
    train(net, train_dl, val_dl, dev)

    # ── Export ONNX ────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    dummy = torch.zeros(1, MAXLEN, dtype=torch.long, device=dev)
    path  = os.path.join(OUT_DIR, ONNX_FILE)
    torch.onnx.export(net, dummy, path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0:"batch", 1:"seq"}}, opset_version=17)

    mdl = onnx.load(path)
    for k,v in [("vocab_json", json.dumps(vocab)),
                ("idx2model_json", json.dumps(i2m)),
                ("spec_lookup_json", json.dumps({c:r for c,r in base}, default=str))]:
        e = mdl.metadata_props.add(); e.key, e.value = k, v
    onnx.save(mdl, path)
    print("\n✅ ONNX autocontenido listo:", path)

    # ── Inference util ─────────────────────────────────────────────
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    vocab = json.loads(meta["vocab_json"])
    idx2m = {int(k):v for k,v in json.loads(meta["idx2model_json"]).items()}
    spec  = json.loads(meta["spec_lookup_json"])

    enc = lambda t: np.asarray([[vocab.get(c,0) for c in t[:MAXLEN]]+[0]*(MAXLEN-len(t))], np.int64)
    sim = lambda a,b: SequenceMatcher(None,a,b).ratio()

    def predict(txt: str):
        logits = sess.run(None, {"input": enc(txt)})[0][0]
        probs  = F.softmax(torch.from_numpy(logits)/TEMP, dim=0).numpy()
        top    = probs.argsort()[-TOP_K:][::-1]

        best_idx, best_sim = None, 0.0
        norm = txt.upper().replace(' ','').replace('-','')
        for idx in top:
            cand = idx2m[int(idx)]
            s    = sim(cand, norm)
            if s > best_sim:
                best_sim = s            # ← similitud real 0-1
                best_idx = int(idx)     # ← índice de clase

        if best_sim < SIM_TH:
            return {"status":"low_confidence", "similarity": round(best_sim,3)}

        p_hat = float(probs[best_idx])
        if p_hat < PROB_TH:
            return {"status":"low_confidence",
                    "similarity": round(best_sim,3),
                    "confidence": round(p_hat,3)}

        rec = OrderedDict(spec[idx2m[best_idx]])
        rec.update(similarity=round(best_sim,3), confidence=round(p_hat,3))
        return rec

    # ── Prueba rápida ─────────────────────────────────────────────
    import pprint; pp = pprint.PrettyPrinter(indent=2, width=120)
    for s in ["GPG1461080M41", "NM0AMIOOCXVAFH"]:
        print(f"\nOCR input: {s}"); pp.pprint(predict(s))
