"""
Loan Approval Predictor - Flask app serving logistic.pkl
Run locally :  python app.py
Production  :  gunicorn -w 2 -b 0.0.0.0:8000 app:app
AWS EB      :  exposes `application` (alias of `app`)
"""
import os
import pickle
import warnings
from datetime import datetime

import pandas as pd
from flask import Flask, jsonify, request, Response

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "logistic.pkl"))

app = Flask(__name__)
application = app  # Elastic Beanstalk looks for `application`

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURES = list(model.feature_names_in_)

# Label encodings (alphabetical LabelEncoder order, standard Loan Prediction dataset)
MAPS = {
    "Gender": {"Female": 0, "Male": 1},
    "Married": {"No": 0, "Yes": 1},
    "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3},
    "Education": {"Graduate": 0, "Not Graduate": 1},
    "Self_Employed": {"No": 0, "Yes": 1},
    "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
}
NUMERIC = {
    "ApplicantIncome": (0, 1_000_000),
    "CoapplicantIncome": (0, 1_000_000),
    "LoanAmount": (1, 10_000),
    "Loan_Amount_Term": (6, 480),
    "Credit_History": (0, 1),
}
NICE = {
    "Gender": "Gender", "Married": "Marital status", "Dependents": "Dependents",
    "Education": "Education", "Self_Employed": "Self employed",
    "ApplicantIncome": "Applicant income", "CoapplicantIncome": "Co-applicant income",
    "LoanAmount": "Loan amount", "Loan_Amount_Term": "Loan term",
    "Credit_History": "Credit history", "Property_Area": "Property area",
}


def parse(payload):
    row = {}
    for k, mp in MAPS.items():
        v = str(payload.get(k, "")).strip()
        if v not in mp:
            raise ValueError(f"Invalid value for {k}: '{v}'")
        row[k] = mp[v]
    for k, (lo, hi) in NUMERIC.items():
        try:
            v = float(payload.get(k))
        except (TypeError, ValueError):
            raise ValueError(f"{k} must be a number")
        if not lo <= v <= hi:
            raise ValueError(f"{k} must be between {lo} and {hi}")
        row[k] = v
    return row


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.route("/health")
def health():
    return jsonify(status="ok", model=type(model).__name__, features=FEATURES)


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        row = parse(request.get_json(force=True, silent=True) or {})
        X = pd.DataFrame([row])[FEATURES]
        proba = float(model.predict_proba(X)[0][1])
        pred = int(model.predict(X)[0])
        factors = sorted(
            [{"name": NICE[f], "impact": round(float(c * row[f]), 3)}
             for f, c in zip(FEATURES, model.coef_[0])],
            key=lambda d: abs(d["impact"]), reverse=True)[:6]
        risk = "Low" if proba >= 0.75 else "Medium" if proba >= 0.5 else "High"
        return jsonify(approved=bool(pred), probability=round(proba * 100, 2),
                       risk=risk, factors=factors,
                       time=datetime.now().strftime("%H:%M:%S"))
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:  # pragma: no cover
        return jsonify(error=f"Server error: {e}"), 500


PAGE = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LoanIQ · Smart Loan Approval Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root{--bg:#f4f6fb;--bg2:#e9eefb;--card:rgba(255,255,255,.85);--text:#12172b;--muted:#6b7491;--border:rgba(20,30,70,.1);
--accent:#6366f1;--accent2:#8b5cf6;--good:#10b981;--bad:#ef4444;--warn:#f59e0b;--input:#fff;--shadow:0 10px 40px rgba(60,70,140,.12)}
[data-theme=dark]{--bg:#0b1020;--bg2:#141a35;--card:rgba(24,30,58,.75);--text:#eef1ff;--muted:#8d97bd;--border:rgba(255,255,255,.09);
--input:#0f1530;--shadow:0 10px 40px rgba(0,0,0,.45)}
*{box-sizing:border-box;margin:0}
body{font-family:Inter,system-ui,sans-serif;background:radial-gradient(1200px 600px at 10% -10%,var(--bg2),transparent),var(--bg);
color:var(--text);min-height:100vh;transition:background .3s,color .3s}
body::before{content:"";position:fixed;inset:0;background:radial-gradient(500px 300px at 90% 10%,rgba(139,92,246,.18),transparent),
radial-gradient(500px 300px at 0% 90%,rgba(99,102,241,.15),transparent);pointer-events:none;z-index:-1}
header{display:flex;justify-content:space-between;align-items:center;padding:18px 5vw;position:sticky;top:0;
backdrop-filter:blur(14px);background:color-mix(in srgb,var(--bg) 70%,transparent);border-bottom:1px solid var(--border);z-index:10}
.logo{display:flex;gap:10px;align-items:center;font-weight:800;font-size:1.25rem}
.logo i{width:36px;height:36px;border-radius:11px;background:linear-gradient(135deg,var(--accent),var(--accent2));display:grid;place-items:center;font-style:normal;color:#fff}
.logo span{background:linear-gradient(90deg,var(--accent),var(--accent2));-webkit-background-clip:text;color:transparent}
.hbtns{display:flex;gap:10px}
.icon-btn{border:1px solid var(--border);background:var(--card);color:var(--text);border-radius:12px;padding:9px 14px;cursor:pointer;font-weight:600;font-size:.85rem;transition:.2s}
.icon-btn:hover{transform:translateY(-2px);border-color:var(--accent)}
main{max-width:1200px;margin:0 auto;padding:34px 5vw 60px}
.hero{text-align:center;margin-bottom:30px}
.hero h1{font-size:clamp(1.8rem,4vw,2.8rem);font-weight:800;letter-spacing:-.02em}
.hero h1 b{background:linear-gradient(90deg,var(--accent),var(--accent2),#ec4899);-webkit-background-clip:text;color:transparent}
.hero p{color:var(--muted);margin-top:8px}
.grid{display:grid;grid-template-columns:1.25fr 1fr;gap:24px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--border);border-radius:22px;padding:26px;box-shadow:var(--shadow);backdrop-filter:blur(12px);animation:rise .6s both}
@keyframes rise{from{opacity:0;transform:translateY(18px)}}
.card h2{font-size:1.05rem;margin-bottom:18px;display:flex;align-items:center;gap:8px}
.fields{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}
@media(max-width:560px){.fields{grid-template-columns:1fr}}
.f{display:flex;flex-direction:column;gap:6px}.f.full{grid-column:1/-1}
label{font-size:.78rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;display:flex;justify-content:space-between}
label em{font-style:normal;color:var(--accent);text-transform:none;letter-spacing:0}
select,input[type=number]{background:var(--input);border:1px solid var(--border);color:var(--text);padding:12px 14px;border-radius:12px;font:inherit;outline:none;transition:.2s;width:100%}
select:focus,input:focus{border-color:var(--accent);box-shadow:0 0 0 4px rgba(99,102,241,.18)}
input[type=range]{accent-color:var(--accent);width:100%}
.seg{display:flex;background:var(--input);border:1px solid var(--border);border-radius:12px;padding:4px}
.seg button{flex:1;border:0;background:transparent;color:var(--muted);padding:9px;border-radius:9px;cursor:pointer;font:inherit;font-weight:600;transition:.2s}
.seg button.on{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff;box-shadow:0 4px 14px rgba(99,102,241,.4)}
.actions{display:flex;gap:12px;margin-top:22px;flex-wrap:wrap}
.btn{flex:1;border:0;border-radius:14px;padding:14px 20px;font:inherit;font-weight:700;cursor:pointer;transition:.2s;min-width:130px}
.btn.primary{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff;box-shadow:0 8px 24px rgba(99,102,241,.4)}
.btn.primary:hover{transform:translateY(-2px)}
.btn.ghost{background:transparent;border:1px solid var(--border);color:var(--text)}
.btn:disabled{opacity:.6;cursor:wait}
.result{text-align:center}
.gauge{position:relative;width:220px;height:220px;margin:6px auto 12px}
.gauge svg{transform:rotate(-90deg)}
.gauge circle{fill:none;stroke-width:16;stroke-linecap:round}
.gauge .bgc{stroke:var(--border)}
.gauge .fg{stroke:var(--muted);stroke-dasharray:565;stroke-dashoffset:565;transition:stroke-dashoffset 1.2s cubic-bezier(.2,.8,.2,1),stroke .4s}
.gauge .mid{position:absolute;inset:0;display:grid;place-content:center}
.gauge .pct{font-size:2.6rem;font-weight:800}
.gauge .sub{font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
.badge{display:inline-block;padding:8px 20px;border-radius:99px;font-weight:700;margin:6px 0}
.badge.ok{background:rgba(16,185,129,.15);color:var(--good)}.badge.no{background:rgba(239,68,68,.15);color:var(--bad)}
.placeholder{color:var(--muted);padding:30px 0;font-size:.95rem}
.chips{display:flex;justify-content:center;gap:8px;margin:10px 0;flex-wrap:wrap}
.chip{font-size:.78rem;padding:6px 12px;border-radius:99px;border:1px solid var(--border);color:var(--muted)}
.chip b{color:var(--text)}
.factors{text-align:left;margin-top:18px}
.factors h3{font-size:.8rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px}
.fr{display:grid;grid-template-columns:130px 1fr 50px;gap:10px;align-items:center;font-size:.82rem;margin:7px 0}
.bar{height:8px;background:var(--border);border-radius:9px;position:relative;overflow:hidden}
.bar i{position:absolute;top:0;bottom:0;border-radius:9px;transition:width 1s}
.pos{background:var(--good)}.neg{background:var(--bad)}
.fr span:last-child{text-align:right;color:var(--muted)}
.tools{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-top:24px}
@media(max-width:900px){.tools{grid-template-columns:1fr}}
.emi{font-size:2rem;font-weight:800;margin:8px 0;background:linear-gradient(90deg,var(--accent),var(--accent2));-webkit-background-clip:text;color:transparent}
.kv{display:flex;justify-content:space-between;font-size:.85rem;color:var(--muted);padding:5px 0;border-bottom:1px dashed var(--border)}
.kv b{color:var(--text)}
table{width:100%;border-collapse:collapse;font-size:.82rem}
th,td{padding:9px 6px;text-align:left;border-bottom:1px solid var(--border)}
th{color:var(--muted);font-weight:600;font-size:.72rem;text-transform:uppercase}
.tag{padding:3px 9px;border-radius:99px;font-size:.72rem;font-weight:700}
.tag.ok{background:rgba(16,185,129,.15);color:var(--good)}.tag.no{background:rgba(239,68,68,.15);color:var(--bad)}
.toast{position:fixed;bottom:24px;left:50%;transform:translate(-50%,100px);background:var(--bad);color:#fff;padding:12px 22px;border-radius:12px;transition:.4s;z-index:50;font-weight:600}
.toast.show{transform:translate(-50%,0)}
footer{text-align:center;color:var(--muted);font-size:.8rem;padding:20px}
.spin{width:16px;height:16px;border:2px solid #fff6;border-top-color:#fff;border-radius:50%;display:inline-block;animation:sp .7s linear infinite;vertical-align:-3px;margin-right:6px}
@keyframes sp{to{transform:rotate(360deg)}}
canvas{position:fixed;inset:0;pointer-events:none;z-index:40}
</style>
</head>
<body>
<header>
  <div class="logo"><i>◆</i><span>LoanIQ</span></div>
  <div class="hbtns">
    <button class="icon-btn" id="sample">✨ Sample</button>
    <button class="icon-btn" id="theme">🌙 Dark</button>
  </div>
</header>
<main>
  <div class="hero">
    <h1>Predict your <b>loan approval</b> in seconds</h1>
    <p>Powered by a Logistic Regression model. Fill in the applicant details and get an instant decision with explanation.</p>
  </div>

  <div class="grid">
    <section class="card">
      <h2>📝 Applicant details</h2>
      <div class="fields" id="form">
        <div class="f"><label>Gender</label><div class="seg" data-k="Gender"><button data-v="Male" class="on">Male</button><button data-v="Female">Female</button></div></div>
        <div class="f"><label>Married</label><div class="seg" data-k="Married"><button data-v="Yes" class="on">Yes</button><button data-v="No">No</button></div></div>
        <div class="f"><label>Dependents</label><select id="Dependents"><option>0</option><option>1</option><option>2</option><option>3+</option></select></div>
        <div class="f"><label>Education</label><select id="Education"><option>Graduate</option><option>Not Graduate</option></select></div>
        <div class="f"><label>Self employed</label><div class="seg" data-k="Self_Employed"><button data-v="No" class="on">No</button><button data-v="Yes">Yes</button></div></div>
        <div class="f"><label>Property area</label><select id="Property_Area"><option>Urban</option><option>Semiurban</option><option>Rural</option></select></div>
        <div class="f"><label>Applicant income <em id="v_ApplicantIncome"></em></label>
          <input type="number" id="ApplicantIncome" min="0" value="5000"><input type="range" id="r_ApplicantIncome" min="0" max="50000" step="100"></div>
        <div class="f"><label>Co-applicant income <em id="v_CoapplicantIncome"></em></label>
          <input type="number" id="CoapplicantIncome" min="0" value="1500"><input type="range" id="r_CoapplicantIncome" min="0" max="30000" step="100"></div>
        <div class="f"><label>Loan amount (in thousands) <em id="v_LoanAmount"></em></label>
          <input type="number" id="LoanAmount" min="1" value="128"><input type="range" id="r_LoanAmount" min="10" max="700" step="1"></div>
        <div class="f"><label>Loan term (months)</label><select id="Loan_Amount_Term">
          <option>12</option><option>36</option><option>60</option><option>84</option><option>120</option><option>180</option><option>240</option><option>300</option><option selected>360</option><option>480</option></select></div>
        <div class="f full"><label>Credit history</label><div class="seg" data-k="Credit_History"><button data-v="1" class="on">✅ Good (meets guidelines)</button><button data-v="0">⚠️ Poor / none</button></div></div>
      </div>
      <div class="actions">
        <button class="btn primary" id="go">Predict approval</button>
        <button class="btn ghost" id="reset">Reset</button>
      </div>
    </section>

    <section class="card result" id="resCard">
      <h2>🎯 Prediction</h2>
      <div class="gauge">
        <svg width="220" height="220" viewBox="0 0 200 200"><circle class="bgc" cx="100" cy="100" r="90"/><circle class="fg" id="fg" cx="100" cy="100" r="90"/></svg>
        <div class="mid"><div class="pct" id="pct">--</div><div class="sub">approval chance</div></div>
      </div>
      <div id="out"><div class="placeholder">Fill in the form and press <b>Predict approval</b>.</div></div>
    </section>
  </div>

  <div class="tools">
    <section class="card">
      <h2>🧮 EMI calculator</h2>
      <label>Interest rate (% p.a.) <em id="rateV">8.5%</em></label>
      <input type="range" id="rate" min="1" max="20" step="0.1" value="8.5">
      <div class="emi" id="emi">–</div>
      <div class="kv"><span>Principal</span><b id="k1">–</b></div>
      <div class="kv"><span>Total interest</span><b id="k2">–</b></div>
      <div class="kv"><span>Total payable</span><b id="k3">–</b></div>
      <div class="kv"><span>EMI vs total income</span><b id="k4">–</b></div>
    </section>
    <section class="card">
      <h2>🕘 Recent predictions <button class="icon-btn" id="clr" style="margin-left:auto;padding:5px 10px">Clear</button></h2>
      <div id="hist"></div>
    </section>
  </div>
</main>
<footer>LoanIQ · for educational use only — not financial advice.</footer>
<div class="toast" id="toast"></div>
<canvas id="cv"></canvas>

<script>
const $=id=>document.getElementById(id);
const state={Gender:"Male",Married:"Yes",Self_Employed:"No",Credit_History:"1"};
const DEF={ApplicantIncome:5000,CoapplicantIncome:1500,LoanAmount:128};
const inr=n=>"₹"+Math.round(n).toLocaleString("en-IN");

/* theme */
function setTheme(t){document.documentElement.dataset.theme=t;$("theme").textContent=t==="dark"?"🌙 Dark":"☀️ Light";try{localStorage.setItem("theme",t)}catch(e){}}
let saved=null;try{saved=localStorage.getItem("theme")}catch(e){}
setTheme(saved||(matchMedia("(prefers-color-scheme: light)").matches?"light":"dark"));
$("theme").onclick=()=>setTheme(document.documentElement.dataset.theme==="dark"?"light":"dark");

/* segmented buttons */
document.querySelectorAll(".seg").forEach(s=>s.querySelectorAll("button").forEach(b=>b.onclick=()=>{
  s.querySelectorAll("button").forEach(x=>x.classList.remove("on"));b.classList.add("on");state[s.dataset.k]=b.dataset.v;calc();}));
function setSeg(k,v){const s=document.querySelector(`.seg[data-k="${k}"]`);s.querySelectorAll("button").forEach(b=>b.classList.toggle("on",b.dataset.v==v));state[k]=String(v)}

/* number + range sync */
["ApplicantIncome","CoapplicantIncome","LoanAmount"].forEach(k=>{
  const n=$(k),r=$("r_"+k);
  const upd=()=>{$("v_"+k).textContent=k==="LoanAmount"?inr((+n.value||0)*1000):inr(+n.value||0);calc()};
  n.oninput=()=>{r.value=n.value;upd()};r.oninput=()=>{n.value=r.value;upd()};r.value=n.value;upd();});
["Loan_Amount_Term"].forEach(k=>$(k).onchange=calc);

/* EMI */
function calc(){
  const P=(+$("LoanAmount").value||0)*1000,n=+$("Loan_Amount_Term").value,r=+$("rate").value/1200;
  $("rateV").textContent=$("rate").value+"%";
  const emi=r?P*r*Math.pow(1+r,n)/(Math.pow(1+r,n)-1):P/n;
  const tot=emi*n,inc=(+$("ApplicantIncome").value||0)+(+$("CoapplicantIncome").value||0);
  $("emi").textContent=inr(emi)+" / month";$("k1").textContent=inr(P);$("k2").textContent=inr(tot-P);$("k3").textContent=inr(tot);
  const ratio=inc?emi/inc*100:0;$("k4").textContent=inc?ratio.toFixed(1)+"%"+(ratio>50?" ⚠️ high":""):"–";
}
$("rate").oninput=calc;

/* history */
function loadH(){try{return JSON.parse(localStorage.getItem("hist")||"[]")}catch(e){return[]}}
function saveH(h){try{localStorage.setItem("hist",JSON.stringify(h.slice(0,8)))}catch(e){}}
function renderH(){const h=loadH();
  $("hist").innerHTML=h.length?`<table><tr><th>Time</th><th>Loan</th><th>Chance</th><th>Result</th></tr>${h.map(x=>
  `<tr><td>${x.t}</td><td>${inr(x.l*1000)}</td><td>${x.p}%</td><td><span class="tag ${x.a?"ok":"no"}">${x.a?"Approved":"Rejected"}</span></td></tr>`).join("")}</table>`
  :'<div class="placeholder">No predictions yet.</div>'}
$("clr").onclick=()=>{saveH([]);renderH()};renderH();

/* predict */
function payload(){return{...state,Dependents:$("Dependents").value,Education:$("Education").value,Property_Area:$("Property_Area").value,
  ApplicantIncome:$("ApplicantIncome").value,CoapplicantIncome:$("CoapplicantIncome").value,LoanAmount:$("LoanAmount").value,
  Loan_Amount_Term:$("Loan_Amount_Term").value}}
function toast(m){const t=$("toast");t.textContent=m;t.classList.add("show");setTimeout(()=>t.classList.remove("show"),3500)}
$("go").onclick=async()=>{
  const b=$("go");b.disabled=true;b.innerHTML='<span class="spin"></span>Analysing…';
  try{
    const res=await fetch("/api/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload())});
    const d=await res.json();if(!res.ok)throw new Error(d.error||"Request failed");show(d);
  }catch(e){toast(e.message)}finally{b.disabled=false;b.textContent="Predict approval"}
};
function show(d){
  const fg=$("fg"),col=d.approved?"var(--good)":"var(--bad)";
  fg.style.stroke=col;fg.style.strokeDashoffset=565*(1-d.probability/100);
  let c=0;const tgt=d.probability,step=()=>{c=Math.min(tgt,c+tgt/40);$("pct").textContent=c.toFixed(1)+"%";if(c<tgt)requestAnimationFrame(step)};step();
  const mx=Math.max(...d.factors.map(f=>Math.abs(f.impact)),.01);
  const rc=d.risk==="Low"?"var(--good)":d.risk==="Medium"?"var(--warn)":"var(--bad)";
  $("out").innerHTML=`<div class="badge ${d.approved?"ok":"no"}">${d.approved?"✅ Likely to be APPROVED":"❌ Likely to be REJECTED"}</div>
   <div class="chips"><span class="chip">Risk: <b style="color:${rc}">${d.risk}</b></span><span class="chip">Checked <b>${d.time}</b></span></div>
   <div class="factors"><h3>What influenced this result</h3>${d.factors.map(f=>`<div class="fr"><span>${f.name}</span>
   <div class="bar"><i class="${f.impact>=0?"pos":"neg"}" style="width:${Math.abs(f.impact)/mx*100}%"></i></div><span>${f.impact>0?"+":""}${f.impact}</span></div>`).join("")}</div>`;
  const h=loadH();h.unshift({t:d.time,l:+$("LoanAmount").value,p:d.probability,a:d.approved});saveH(h);renderH();
  if(d.approved)confetti();
  if(innerWidth<900)$("resCard").scrollIntoView({behavior:"smooth"});
}

/* sample / reset */
function fill(o){setSeg("Gender",o.Gender);setSeg("Married",o.Married);setSeg("Self_Employed",o.Self_Employed);setSeg("Credit_History",o.Credit_History);
  ["Dependents","Education","Property_Area","Loan_Amount_Term"].forEach(k=>$(k).value=o[k]);
  ["ApplicantIncome","CoapplicantIncome","LoanAmount"].forEach(k=>{$(k).value=o[k];$("r_"+k).value=o[k];$("v_"+k).textContent=k==="LoanAmount"?inr(o[k]*1000):inr(o[k])});calc()}
$("sample").onclick=()=>{const r=a=>a[Math.floor(Math.random()*a.length)];
  fill({Gender:r(["Male","Female"]),Married:r(["Yes","No"]),Self_Employed:r(["Yes","No"]),Credit_History:r(["1","1","0"]),Dependents:r(["0","1","2","3+"]),
  Education:r(["Graduate","Not Graduate"]),Property_Area:r(["Urban","Semiurban","Rural"]),Loan_Amount_Term:r(["120","240","360","360"]),
  ApplicantIncome:2000+Math.floor(Math.random()*16000),CoapplicantIncome:Math.floor(Math.random()*6000),LoanAmount:60+Math.floor(Math.random()*300)})};
$("reset").onclick=()=>{fill({Gender:"Male",Married:"Yes",Self_Employed:"No",Credit_History:"1",Dependents:"0",Education:"Graduate",Property_Area:"Urban",Loan_Amount_Term:"360",...DEF});
  $("fg").style.strokeDashoffset=565;$("fg").style.stroke="var(--muted)";$("pct").textContent="--";$("out").innerHTML='<div class="placeholder">Fill in the form and press <b>Predict approval</b>.</div>'};

/* confetti */
function confetti(){const cv=$("cv"),x=cv.getContext("2d");cv.width=innerWidth;cv.height=innerHeight;
  const P=Array.from({length:110},()=>({x:innerWidth/2,y:innerHeight/2.5,vx:(Math.random()-.5)*14,vy:Math.random()*-13-3,s:Math.random()*7+4,c:`hsl(${Math.random()*360},85%,60%)`,r:Math.random()*6}));
  let f=0;(function a(){x.clearRect(0,0,cv.width,cv.height);P.forEach(p=>{p.x+=p.vx;p.y+=p.vy;p.vy+=.35;p.r+=.2;x.save();x.translate(p.x,p.y);x.rotate(p.r);x.fillStyle=p.c;x.fillRect(-p.s/2,-p.s/2,p.s,p.s*.6);x.restore()});
  if(++f<120)requestAnimationFrame(a);else x.clearRect(0,0,cv.width,cv.height)})()}
calc();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)
