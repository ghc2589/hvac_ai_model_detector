import os, json, re, random, numpy as np
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          DataCollatorForTokenClassification, TrainingArguments, Trainer)
from sklearn.metrics import precision_recall_fscore_support
from pathlib import Path
from transformers.onnx import export
from transformers.onnx.features import FeaturesManager
import onnxruntime as ort

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ---------- 1. cargar sets LIMPIOS ----------
train_data = json.load(open("hvac_ner_dataset_clean_v10.json"))
val_data   = json.load(open("hvac_ner_validation_clean.json"))

# ---------- 2. generar refuerzos ----------
brands  = ["Carrier","Trane","Daikin","Samsung","York","Lennox","Rheem"]
models  = ["AX-100Z","RZQ200","YVFE60","PRO88-XL","AM100CXVAFH"]
refrig  = ["R-410A","R-32","R-134a"]
volts   = ["220V","208-230V","460V"]
freqs   = ["50Hz","60Hz"]; phases=["1Ph","3Ph","3N~"]

def tok(s): return re.findall(r"[A-Za-z0-9\-]+|[()°/]|kW|kg|Btu/h|Hz|Ph|~|uses|and|power|input|feed", s)

def make_free():
    b,m,r,v = random.choice(brands),random.choice(models),random.choice(refrig),random.choice(volts)
    sent=f"{b} {m} uses {r} and {v} power"
    t,tags=[],[]
    for w in tok(sent):
        t.append(w)
        tags.append("B-BRAND" if w==b else
                    "B-MODEL" if w==m else
                    "B-REFRIGERANT_TYPE" if w==r else
                    "B-VOLTAGE" if w==v else "O")
    return {"tokens":t,"ner_tags":tags}

def make_layout():
    header,label,val = random.choice([
      ("POWER SUPPLY","VOLTAGE",lambda:f"{random.choice(volts)}{random.choice(freqs)}{random.choice(phases)}"),
      ("REFRIGERANT CHARGE","REFRIGERANT_KG",lambda:f"{random.uniform(1.5,4):.1f} kg {random.choice(refrig)}")
    ])
    tokens=tok(header)+tok(val())
    tags  = ["B-HEADER"]+["I-HEADER"]*(len(tok(header))-1)
    first=True
    for w in tokens[len(tok(header)):]:
        if w in {"kg","Hz","Ph","~"}: tags.append("B-UNIT")
        else: tags.append(("B-" if first else "I-")+label); first=False
    return {"tokens":tokens,"ner_tags":tags}

train_data += [make_free()   for _ in range(1500)]
train_data += [make_layout() for _ in range(300)]

train_ds, val_ds = Dataset.from_list(train_data), Dataset.from_list(val_data)

# ---------- 3. etiquetas ----------
labs = sorted({l for ex in (train_data+val_data) for l in ex["ner_tags"]})
lid  = {l:i for i,l in enumerate(labs)}; idl={i:l for l,i in lid.items()}

# ---------- 4. tokenizador ----------
tokz=AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
def encode(ex):
    enc=tokz(ex["tokens"],is_split_into_words=True, truncation=True)
    enc["labels"]=[-100 if w is None else lid[ex["ner_tags"][w]] for w in enc.word_ids()]
    return enc
train_tok=train_ds.map(encode); val_tok=val_ds.map(encode)

ckpt ="huawei-noah/TinyBERT_General_4L_312D"          # carpeta local
model = AutoModelForTokenClassification.from_pretrained(
    ckpt,
    num_labels=len(labs),
    id2label=labs,
    label2id=lid,
    use_safetensors=True,
)

def metrics(p):
    logits,labels=p
    preds=np.argmax(logits,2)
    pl=[[idl[p] for p,l in zip(ps,ls) if l!=-100] for ps,ls in zip(preds,labels)]
    tl=[[idl[l] for p,l in zip(ps,ls) if l!=-100] for ps,ls in zip(preds,labels)]
    pr,rc,f1,_=precision_recall_fscore_support(sum(tl,[]),sum(pl,[]),average="macro")
    return {"precision":pr,"recall":rc,"f1":f1}

args=TrainingArguments(
    "./hvac_ner_model",  
    per_device_train_batch_size=8, 
    num_train_epochs=15,
    learning_rate=1e-5, 
    weight_decay=0.01, 
    logging_dir="./logs",
    do_eval=True, 
    eval_steps=1000, 
    save_steps=1000, 
    warmup_ratio=0.1,
    
)

trainer=Trainer(model=model,args=args,train_dataset=train_tok,eval_dataset=val_tok,
                tokenizer=tokz,data_collator=DataCollatorForTokenClassification(tokz),
                compute_metrics=metrics)
trainer.train()
trainer.save_model("./hvac_ner_model"); tokz.save_pretrained("./hvac_ner_model")

# ----- export ONNX & prueba rápida  -----
model.cpu()
onnx_path=Path("../AI-Models/ner_model.onnx")
_,cfg=FeaturesManager.check_supported_model_or_raise(model,"token-classification")
export(preprocessor=tokz,model=model,config=cfg(model.config),opset=14,output=onnx_path)

print("✅ ONNX listo:",onnx_path)
session=ort.InferenceSession(str(onnx_path),providers=["CPUExecutionProvider"])
test=["Samsung","AX-100Z","uses","R-410A","and","220V","power"]
enc=tokz(test,is_split_into_words=True,return_tensors="np")
outs=session.run(None,{"input_ids":enc["input_ids"],
                       "attention_mask":enc["attention_mask"],
                       "token_type_ids":np.zeros_like(enc["input_ids"])})[0]
preds=outs.argmax(-1)[0]
print("🧾 NER:",[(t,idl[int(p)]) for t,p in zip(test,preds[1:len(test)+1])])

# ---------- 10. Construir JSON estructurado ----------------------

# 1. de etiquetas a tokens agrupados
tokens     = test                                               # lista: Samsung …
pred_tags  = [idl[int(p)] for p in preds[1:len(tokens)+1]]      # salta el [CLS]

entities, curr_lbl, curr_toks = {}, None, []
for tok, tag in zip(tokens, pred_tags):
    if tag.startswith("B-"):
        if curr_lbl:                                            # cierra la anterior
            entities.setdefault(curr_lbl, []).append(" ".join(curr_toks))
        curr_lbl, curr_toks = tag[2:], [tok]                    # nueva entidad
    elif tag.startswith("I-") and curr_lbl == tag[2:]:
        curr_toks.append(tok)                                   # continúa
    else:                                                       # "O" o cambio de etiqueta
        if curr_lbl:
            entities.setdefault(curr_lbl, []).append(" ".join(curr_toks))
        curr_lbl, curr_toks = None, []
if curr_lbl:                                                    # cierra la última
    entities.setdefault(curr_lbl, []).append(" ".join(curr_toks))

# 2. mapa NER → campos de salida
FIELD_MAP = {
    "BRAND":"Brand","MODEL":"Model","CATALOG_NUMBER":"Catalog Number",
    "REFRIGERANT_TYPE":"Refrigerant Type","FACTORY_CHARGE":"Factory Charge (lbs)",
    "VOLTAGE":"Voltage","PHASE":"Phase","FREQUENCY":"Frequency",
    "RLA":"RLA","FLA":"FLA",
    "HIGH_SIDE":"Preasure Ratings High Side(PSI)",
    "LOW_SIDE":"Preasure Ratings Low Side(PSI)",
    "INPUT_BTU":"Heathing Specifications Input (BTU/hr)",
    "OUTPUT_BTU":"Heathing Specifications Output(BTU/hr)",
    "EFFICIENCY":"Heathing Specifications Efficiency(%)",
    "GAS_TYPE":"Heathing Specifications Gas Type",
    "GAS_SUPPLY_MIN":"Gas suply Pressure Minimun (in WC)",
    "GAS_SUPPLY_MAX":"Gas suply Pressure Maximun(In WC)",
    "INPUT_MIN":"Gas Supply Information Input Min (btu/hr)",
    "INPUT_MAX":"Gas Supply Information  Input Max (BTU/hr)",
    "GAS_SUPPLY":"Gas Supply Information  Gas supply (In WC)",
    "MANIFOLD_PRESSURE":"Gas Supply Information Manifold Pressure(in WC)",
    "AIR_TEMP_RISE":"Airflow and Temperature  Air temperature rise (ºF)",
    "MAX_STATIC_PRESSURE":"Airflow and Temperature Max external static Pressure (in WC)",
    "COOLING":"Cooling Data Cooling (BTU/h)",
    "IEER":"Cooling Data IEER","EER":"Cooling Data EER",
    "COMPRESSOR_QTY":"Compressor Data  Quantity",
    "COMPRESSOR_HZ":"Compressor Data HZ",
    "REFRIGERANT_KG":"Compressor Data Refrigerant charge (kg)",
    "TEST_PRESSURE":"Compressor Data Test pressure Gage(PSI)",
    "MIN_AMBIENT_TEMP":"Min Operating Ambient Temperatures (ºF)",
    "MAX_AMBIENT_TEMP":"Max operating Ambient Temperatures (ºF)",
    "INSTALLATION_TYPE":"Installation type",
    "MAX_AIR_TEMP":"Maximun Air Temperature(ºF)"
}

# 3. JSON final
result_json = {v: "" for v in FIELD_MAP.values()}
for ner_key, vals in entities.items():
    if ner_key in FIELD_MAP and vals:
        result_json[FIELD_MAP[ner_key]] = vals[0]               # primer match

print("\n📦 JSON extraído:")
print(json.dumps(result_json, indent=2, ensure_ascii=False))
