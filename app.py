# =============================================================================
# app.py — PredictaStock v5.0
# Sistema de Predicción de Demanda con Machine Learning para MyPEs — Perú
# =============================================================================
import warnings; warnings.filterwarnings("ignore")
import io, hashlib
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, PageBreak, HRFlowable, Image as RLImage)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import streamlit as st

# =============================================================================
st.set_page_config(page_title="PredictaStock v5.0", page_icon="📦",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;}
.stApp{background-color:#f0f2f6;}
.main-header{background:linear-gradient(135deg,#0f3460 0%,#16213e 100%);padding:1.4rem 2rem;border-radius:12px;margin-bottom:1.2rem;color:white;box-shadow:0 4px 20px rgba(15,52,96,.3);}
.main-header h1{font-size:1.6rem;font-weight:700;margin:0 0 .2rem 0;}
.main-header p{font-size:.85rem;color:#a8b8d8;margin:0;}
.metric-card{background:white;border-radius:10px;padding:1.1rem 1.3rem;border-left:5px solid #0f3460;box-shadow:0 2px 10px rgba(0,0,0,.06);margin-bottom:.5rem;}
.metric-card.g{border-left-color:#0d9e6e;}.metric-card.o{border-left-color:#e05c2d;}.metric-card.p{border-left-color:#7c3aed;}.metric-card.r{border-left-color:#dc2626;}
.ml{font-size:.73rem;text-transform:uppercase;letter-spacing:.08em;color:#6b7a90;font-weight:600;margin-bottom:.25rem;}
.mv{font-size:1.7rem;font-weight:700;color:#1a1a2e;font-family:'IBM Plex Mono',monospace;}
.ms{font-size:.7rem;color:#9aaabf;margin-top:.1rem;}
.ar{background:#fff1f1;border:1.5px solid #fca5a5;border-radius:10px;padding:.8rem 1.1rem;margin:.35rem 0;color:#7f1d1d;}
.ay{background:#fffbeb;border:1.5px solid #fcd34d;border-radius:10px;padding:.8rem 1.1rem;margin:.35rem 0;color:#78350f;}
.ag{background:#f0fdf4;border:1.5px solid #86efac;border-radius:10px;padding:.8rem 1.1rem;margin:.35rem 0;color:#14532d;}
.ai-box{background:linear-gradient(135deg,#eef3ff,#f5f0ff);border:1.5px solid #c5d3f5;border-radius:12px;padding:1.2rem 1.4rem;margin:.8rem 0;color:#1a1a2e;font-size:.9rem;line-height:1.7;}
.ai-box .ai-label{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#5b21b6;margin-bottom:.4rem;}
.onboarding-step{background:white;border-radius:12px;padding:1.2rem;border:1.5px solid #e2e8f0;margin:.5rem 0;display:flex;gap:1rem;align-items:flex-start;}
.step-num{background:#0f3460;color:white;border-radius:50%;width:28px;height:28px;min-width:28px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.85rem;}
.trend-up{color:#0d9e6e;font-weight:700;} .trend-dn{color:#dc2626;font-weight:700;} .trend-eq{color:#6b7a90;font-weight:600;}
.sim-box{background:#f0fdf4;border:1.5px solid #86efac;border-radius:10px;padding:1rem 1.2rem;margin:.5rem 0;}
section[data-testid="stSidebar"]{background-color:#1a1a2e!important;}
section[data-testid="stSidebar"] *{color:#c8d6e8!important;}
#MainMenu,footer{visibility:hidden;}
.badge-basic{background:#dbeafe;color:#1e40af;padding:3px 10px;border-radius:20px;font-size:.73rem;font-weight:600;}
.badge-pro{background:#ede9fe;color:#5b21b6;padding:3px 10px;border-radius:20px;font-size:.73rem;font-weight:600;}
.badge-experto{background:#fef3c7;color:#92400e;padding:3px 10px;border-radius:20px;font-size:.73rem;font-weight:600;}
</style>""", unsafe_allow_html=True)

# =============================================================================
# AUTH
# =============================================================================
PLANES = {
    "Basic":   {"graficos":2, "alertas":False, "prediccion":False, "pdf":False, "ia":False, "multi":False},
    "Pro":     {"graficos":7, "alertas":True,  "prediccion":False, "pdf":False, "ia":False, "multi":True},
    "Experto": {"graficos":7, "alertas":True,  "prediccion":True,  "pdf":True,  "ia":True,  "multi":True},
}

def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()
def init_users():
    if "users_db" not in st.session_state:
        st.session_state.users_db = {"Admin": {"password": hash_pw("Deam5412#"), "plan": "Experto"}}

def login_user(u, pw):
    db = st.session_state.users_db
    if u in db and db[u]["password"] == hash_pw(pw):
        st.session_state.logged_in = True
        st.session_state.username  = u
        st.session_state.user_plan = db[u]["plan"]
        return True
    return False

def register_user(u, pw, plan):
    db = st.session_state.users_db
    if not u or not pw: return False,"Usuario y contraseña requeridos."
    if len(pw) < 6: return False,"Contraseña mínimo 6 caracteres."
    if u in db:       return False,f"El usuario '{u}' ya existe."
    db[u] = {"password": hash_pw(pw), "plan": plan}
    return True,"¡Cuenta creada! Ahora puedes iniciar sesión."

# =============================================================================
# ML
# =============================================================================
@st.cache_data
def ejemplo_df():
    d=pd.date_range("2024-01-01",periods=10,freq="D")
    p=["Polo Básico","Jean Slim","Vestido","Polo Básico","Jean Slim","Vestido","Polo Básico","Jean Slim","Vestido","Polo Básico"]
    return pd.DataFrame({"Fecha":d.strftime("%Y-%m-%d"),"Producto":p,"Cantidad":[12,8,15,10,7,18,14,9,20,11]})

def limpiar(df):
    df.columns=df.columns.str.strip()
    if "Producto" in df.columns: df["Producto"]=df["Producto"].astype(str).str.strip().str.title()
    if "Fecha"    in df.columns: df["Fecha"]=pd.to_datetime(df["Fecha"],dayfirst=True,errors="coerce"); df.dropna(subset=["Fecha"],inplace=True)
    if "Cantidad" in df.columns: df["Cantidad"]=pd.to_numeric(df["Cantidad"],errors="coerce"); df.dropna(subset=["Cantidad"],inplace=True); df["Cantidad"]=df["Cantidad"].astype(float)
    return df

def validar(df): return {"Fecha","Producto","Cantidad"}.issubset(set(df.columns))

def feats(serie):
    d=serie.copy().sort_values("Fecha").reset_index(drop=True)
    d["mes"]=d["Fecha"].dt.month; d["dia"]=d["Fecha"].dt.dayofweek
    d["anio"]=d["Fecha"].dt.year; d["lag1"]=d["Cantidad"].shift(1)
    d.dropna(inplace=True); return d

def entrenar(df_p):
    f=feats(df_p[["Fecha","Cantidad"]])
    X=f[["mes","dia","anio","lag1"]]; y=f["Cantidad"]
    n=min(15,len(f)-1)
    m=RandomForestRegressor(n_estimators=200,max_depth=6,random_state=42,n_jobs=-1)
    m.fit(X.iloc[:-n],y.iloc[:-n])
    return m, mean_absolute_error(y.iloc[-n:],m.predict(X.iloc[-n:])), f

def predecir(m,f,n=7):
    ul=f["Fecha"].max(); lag=f["Cantidad"].iloc[-1]; rows=[]
    for i in range(1,n+1):
        fp=ul+timedelta(days=i)
        qty=max(0.0,round(m.predict([[fp.month,fp.dayofweek,fp.year,lag]])[0],2))
        rows.append({"Fecha":fp.strftime("%Y-%m-%d"),"Cantidad_Predicha":qty}); lag=qty
    return pd.DataFrame(rows)

def metricas(dp,mae):
    t=dp["Cantidad_Predicha"].sum(); p=dp["Cantidad_Predicha"].mean()
    return round(t,1),round(mae*1.65,1),round(p*3+mae*1.65,1)

def alertas(df):
    rows=[]; MESES={1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}
    um=df["Fecha"].max().to_period("M"); ma=um-1
    for pr in df["Producto"].unique():
        dp=df[df["Producto"]==pr]
        vu=dp[dp["Fecha"].dt.to_period("M")==um]["Cantidad"].sum()
        vp=dp[dp["Fecha"].dt.to_period("M")==ma]["Cantidad"].sum()
        var=(vu-vp)/vp*100 if vp>0 else 0
        mp=dp.groupby(dp["Fecha"].dt.month)["Cantidad"].sum().idxmax()
        nv="CRÍTICO" if var<=-20 else("ATENCIÓN" if var<=-5 else "ESTABLE")
        rows.append({"Producto":pr,"Nivel":nv,"Var%":round(var,1),"Ult":int(vu),"Ant":int(vp),"Pico":MESES.get(mp,str(mp)),"Prom":round(dp["Cantidad"].mean(),1)})
    return pd.DataFrame(rows).sort_values("Var%")

# =============================================================================
# TENDENCIA YoY
# =============================================================================
def calcular_yoy(df):
    """Calcula crecimiento año a año por producto."""
    df["Anio"]=df["Fecha"].dt.year
    anios=sorted(df["Anio"].unique())
    if len(anios)<2: return None
    a1,a2=anios[-2],anios[-1]
    v1=df[df["Anio"]==a1].groupby("Producto")["Cantidad"].sum()
    v2=df[df["Anio"]==a2].groupby("Producto")["Cantidad"].sum()
    resultado=[]
    for pr in df["Producto"].unique():
        c1=v1.get(pr,0); c2=v2.get(pr,0)
        yoy=((c2-c1)/c1*100) if c1>0 else 0
        resultado.append({"Producto":pr,f"Und. {a1}":int(c1),f"Und. {a2}":int(c2),"YoY%":round(yoy,1)})
    return pd.DataFrame(resultado).sort_values("YoY%",ascending=False), a1, a2

# =============================================================================
# RESUMEN IA (Claude API)
# =============================================================================
def generar_resumen_ia(df, df_alertas, prod_sel, demanda, ss, pp, mae, yoy_df=None):
    """Llama a la API de Claude para generar un resumen ejecutivo en lenguaje natural."""
    criticos  = df_alertas[df_alertas["Nivel"]=="CRÍTICO"]["Producto"].tolist()
    atencion  = df_alertas[df_alertas["Nivel"]=="ATENCIÓN"]["Producto"].tolist()
    top_prod  = df.groupby("Producto")["Cantidad"].sum().idxmax()
    total_und = int(df["Cantidad"].sum())
    anios     = sorted(df["Fecha"].dt.year.unique())

    yoy_info = ""
    if yoy_df is not None:
        top_crecimiento = yoy_df.iloc[0]
        top_caida       = yoy_df.iloc[-1]
        yoy_info = f"Tendencia YoY: '{top_crecimiento['Producto']}' creció {top_crecimiento['YoY%']:+.1f}% y '{top_caida['Producto']}' cayó {top_caida['YoY%']:+.1f}%."

    prompt = f"""Eres el analista de negocios de PredictaStock. Analiza estos datos de una MyPE retail peruana y genera un resumen ejecutivo en español claro y directo (máximo 5 oraciones cortas, sin listas, sin markdown, en prosa natural):

Datos del negocio:
- Total unidades históricas: {total_und:,}
- Años de datos: {', '.join(map(str,anios))}
- Producto estrella: {top_prod}
- Productos CRÍTICOS (caída >20%): {', '.join(criticos) if criticos else 'ninguno'}
- Productos en ATENCIÓN: {', '.join(atencion) if atencion else 'ninguno'}
- Producto analizado: {prod_sel}
- Demanda predicha 7 días: {demanda:.1f} unidades
- Stock de seguridad recomendado: {ss:.1f} unidades
- Punto de pedido: {pp:.1f} unidades
- Precisión del modelo (MAE): {mae:.2f} unidades
{yoy_info}

Escribe el resumen como si hablaras directamente con el dueño de la tienda. Menciona los puntos más importantes: qué producto necesita atención inmediata, cuándo reponer y si el negocio crece o decrece."""

    try:
        import urllib.request, json as jsonlib
        payload = jsonlib.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}]
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = jsonlib.loads(resp.read())
            return data["content"][0]["text"]
    except Exception as e:
        return f"(Resumen IA no disponible en este entorno. Error: {str(e)[:60]})"

# =============================================================================
# LAYOUT BASE GRÁFICOS
# =============================================================================
LB = dict(plot_bgcolor="white",paper_bgcolor="white",
          font=dict(family="IBM Plex Sans,Arial",color="#1a1a2e",size=12),
          title_font=dict(size=13,color="#0f3460"),
          legend=dict(font=dict(color="#1a1a2e",size=11)),
          xaxis=dict(tickfont=dict(color="#1a1a2e",size=11),title_font=dict(color="#1a1a2e"),showgrid=True,gridcolor="#f0f0f0"),
          yaxis=dict(tickfont=dict(color="#1a1a2e",size=11),title_font=dict(color="#1a1a2e"),showgrid=True,gridcolor="#f0f0f0"),
          margin=dict(l=10,r=10,t=45,b=10),height=380)
MESES={1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}

# =============================================================================
# GRÁFICOS
# =============================================================================
def g_ranking(df):
    d=df.groupby("Producto")["Cantidad"].sum().reset_index().sort_values("Cantidad",ascending=True)
    fig=px.bar(d,x="Cantidad",y="Producto",orientation="h",color="Cantidad",
               color_continuous_scale="Blues",text="Cantidad",title="🏆 Ranking de Productos")
    fig.update_traces(texttemplate="%{text:,.0f}",textposition="outside",textfont=dict(color="#1a1a2e",size=11))
    fig.update_layout(**LB,coloraxis_showscale=False); return fig

def g_pie(df):
    d=df.groupby("Producto")["Cantidad"].sum().reset_index()
    fig=px.pie(d,names="Producto",values="Cantidad",color_discrete_sequence=px.colors.sequential.Blues_r,
               hole=.38,title="🥧 Participación de Ventas")
    fig.update_traces(textposition="outside",textinfo="percent+label",textfont=dict(color="#1a1a2e",size=11))
    fig.update_layout(**{**LB,"showlegend":False}); return fig

def g_anio(df):
    df=df.copy(); df["Anio"]=df["Fecha"].dt.year
    d=df.groupby(["Anio","Producto"])["Cantidad"].sum().reset_index().rename(columns={"Anio":"Año"})
    fig=px.bar(d,x="Producto",y="Cantidad",color="Año",barmode="group",
               color_discrete_sequence=["#0f3460","#e05c2d","#0d9e6e","#7c3aed"],
               text_auto=".0f",title="📅 Comparativo por Año")
    fig.update_traces(textfont=dict(color="#1a1a2e",size=10))
    fig.update_layout(**LB); fig.update_xaxes(tickangle=-25); return fig

def g_mensual(df):
    df=df.copy(); df["Anio"]=df["Fecha"].dt.year; df["Mes"]=df["Fecha"].dt.month
    d=df.groupby(["Anio","Mes"])["Cantidad"].sum().reset_index().rename(columns={"Anio":"Año"})
    d["MN"]=d["Mes"].map(MESES)
    fig=px.line(d,x="MN",y="Cantidad",color="Año",markers=True,
                color_discrete_sequence=["#0f3460","#e05c2d","#0d9e6e","#7c3aed"],
                category_orders={"MN":list(MESES.values())},title="📆 Ventas Mensuales por Año")
    fig.update_traces(line_width=2.5,marker_size=7); fig.update_layout(**LB); return fig

def g_heat(df,anio):
    df=df.copy(); df["Anio"]=df["Fecha"].dt.year; df["Mes"]=df["Fecha"].dt.month
    d=df[df["Anio"]==anio].groupby(["Producto","Mes"])["Cantidad"].sum().reset_index()
    piv=d.pivot(index="Producto",columns="Mes",values="Cantidad").fillna(0)
    piv.columns=[MESES.get(c,c) for c in piv.columns]
    fig=px.imshow(piv,color_continuous_scale="Blues",text_auto=".0f",aspect="auto",
                  title=f"🌡️ Mapa de Calor — {anio}")
    fig.update_traces(textfont=dict(color="#1a1a2e",size=10))
    fig.update_layout(**{**LB,"coloraxis_showscale":False}); return fig

def g_var(df_al):
    col=df_al["Nivel"].map({"CRÍTICO":"#dc2626","ATENCIÓN":"#d97706","ESTABLE":"#0d9e6e"})
    fig=go.Figure(go.Bar(x=df_al["Producto"],y=df_al["Var%"],marker_color=col.tolist(),
        text=[f"{v:+.1f}%" for v in df_al["Var%"]],textposition="outside",
        textfont=dict(color="#1a1a2e",size=11)))
    fig.add_hline(y=0,line_dash="dash",line_color="#6b7a90",line_width=1.5)
    fig.update_layout(**{**LB,"title":"📊 Variación % vs. Mes Anterior","yaxis_title":"Variación %"})
    fig.update_xaxes(tickangle=-20); return fig

def g_pred(df_h,df_p,prod):
    dh=df_h.tail(30).copy(); dh["Tipo"]="Ventas Reales"
    dh["Fs"]=dh["Fecha"].dt.strftime("%Y-%m-%d"); dh=dh.rename(columns={"Cantidad":"Unidades"})
    dp=df_p.copy(); dp["Tipo"]="Predicción ML"; dp=dp.rename(columns={"Cantidad_Predicha":"Unidades"}); dp["Fs"]=dp["Fecha"]
    union=pd.DataFrame([{"Fs":dh["Fs"].iloc[-1],"Unidades":dh["Unidades"].iloc[-1],"Tipo":"Predicción ML"}])
    todo=pd.concat([dh[["Fs","Unidades","Tipo"]],union,dp[["Fs","Unidades","Tipo"]]],ignore_index=True)
    fig=px.line(todo,x="Fs",y="Unidades",color="Tipo",markers=True,
                color_discrete_map={"Ventas Reales":"#0f3460","Predicción ML":"#e05c2d"},
                labels={"Fs":"Fecha","Tipo":""},title=f"📈 Histórico y Proyección — {prod}")
    fig.add_vrect(x0=dp["Fs"].iloc[0],x1=dp["Fs"].iloc[-1],fillcolor="rgba(224,92,45,.07)",
                  layer="below",line_width=0,annotation_text="Período predicho",
                  annotation_position="top left",annotation_font_color="#e05c2d",annotation_font_size=10)
    fig.update_traces(line_width=2.5,marker_size=6)
    fig.update_layout(**LB); fig.update_layout(legend=dict(orientation="h",y=1.05))
    fig.update_xaxes(tickangle=-30); return fig

def g_yoy(yoy_df,a1,a2):
    """Gráfico de tendencia YoY."""
    col=yoy_df["YoY%"].apply(lambda x:"#0d9e6e" if x>0 else("#dc2626" if x<-5 else "#d97706"))
    fig=go.Figure(go.Bar(x=yoy_df["Producto"],y=yoy_df["YoY%"],
        marker_color=col.tolist(),
        text=[f"{v:+.1f}%" for v in yoy_df["YoY%"]],
        textposition="outside",textfont=dict(color="#1a1a2e",size=11)))
    fig.add_hline(y=0,line_dash="dash",line_color="#6b7a90",line_width=1.5)
    fig.update_layout(**{**LB,"title":f"📈 Tendencia YoY: {a1} → {a2}","yaxis_title":"Crecimiento %"})
    fig.update_xaxes(tickangle=-20); return fig

def g_multi(df,productos,mae_dict):
    """Gráfico comparativo de predicciones multi-producto."""
    rows=[]
    for pr,dp in productos.items():
        for _,r in dp.iterrows():
            rows.append({"Fecha":r["Fecha"],"Producto":pr,"Predicha":r["Cantidad_Predicha"]})
    d=pd.DataFrame(rows)
    fig=px.line(d,x="Fecha",y="Predicha",color="Producto",markers=True,
                title="🔮 Predicción 7 días — Todos los productos")
    fig.update_traces(line_width=2,marker_size=5)
    fig.update_layout(**LB); fig.update_xaxes(tickangle=-25); return fig

# =============================================================================
# PANTALLA LOGIN
# =============================================================================
def pantalla_login():
    init_users()
    if "show_reg" not in st.session_state: st.session_state.show_reg=False
    if "ok_msg"   not in st.session_state: st.session_state.ok_msg=""

    _,col,_=st.columns([1,1.6,1])
    with col:
        st.markdown("""<div style='text-align:center;margin-bottom:1.4rem;'>
        <div style='font-size:2.8rem;'>📦</div>
        <div style='font-size:1.7rem;font-weight:700;color:#0f3460;'>PredictaStock</div>
        <div style='font-size:.82rem;color:#6b7a90;margin-top:.2rem;'>
        Sistema de Predicción de Demanda · MyPEs del sector retail · Perú</div></div>""",
        unsafe_allow_html=True)

        if not st.session_state.show_reg:
            st.markdown("#### 🔐 Iniciar sesión")
            if st.session_state.ok_msg:
                st.success(st.session_state.ok_msg); st.session_state.ok_msg=""
            u=st.text_input("Usuario",placeholder="Ej: Admin",key="li_u")
            p=st.text_input("Contraseña",type="password",placeholder="••••••••",key="li_p")
            if st.button("Ingresar",type="primary",use_container_width=True):
                if login_user(u,p): st.rerun()
                else: st.error("❌ Usuario o contraseña incorrectos.")
            st.markdown("---")
            st.markdown("<div style='text-align:center;font-size:.83rem;color:#6b7a90;'>¿No tienes cuenta?</div>",unsafe_allow_html=True)
            if st.button("Crear una cuenta",use_container_width=True):
                st.session_state.show_reg=True; st.rerun()
        else:
            st.markdown("#### 🆕 Crear cuenta nueva")
            c1,c2,c3=st.columns(3)
            with c1: st.markdown("""<div style='background:#eff6ff;border:2px solid #60a5fa;border-radius:10px;padding:.9rem;text-align:center;'>
            <div style='font-weight:700;color:#1e40af;font-size:1rem;'>🔵 Basic</div>
            <div style='font-size:.72rem;color:#6b7a90;'>Gratis</div>
            <div style='font-size:.75rem;color:#374151;margin-top:.4rem;line-height:1.7;'>✅ 2 gráficos<br>❌ Alertas<br>❌ Predicción<br>❌ PDF</div></div>""",unsafe_allow_html=True)
            with c2: st.markdown("""<div style='background:#f5f3ff;border:2px solid #a78bfa;border-radius:10px;padding:.9rem;text-align:center;'>
            <div style='font-weight:700;color:#5b21b6;font-size:1rem;'>🟣 Pro</div>
            <div style='font-size:.72rem;color:#6b7a90;'>S/. 29/mes</div>
            <div style='font-size:.75rem;color:#374151;margin-top:.4rem;line-height:1.7;'>✅ Todos gráficos<br>✅ Alertas<br>✅ Multi-producto<br>❌ PDF/IA</div></div>""",unsafe_allow_html=True)
            with c3: st.markdown("""<div style='background:#fffbeb;border:2px solid #f59e0b;border-radius:10px;padding:.9rem;text-align:center;'>
            <div style='font-weight:700;color:#92400e;font-size:1rem;'>🌟 Experto</div>
            <div style='font-size:.72rem;color:#6b7a90;'>S/. 59/mes</div>
            <div style='font-size:.75rem;color:#374151;margin-top:.4rem;line-height:1.7;'>✅ Todo lo anterior<br>✅ Predicción ML<br>✅ Resumen IA<br>✅ PDF</div></div>""",unsafe_allow_html=True)
            st.markdown("<br>",unsafe_allow_html=True)
            nu=st.text_input("Nombre de usuario",key="ru"); np1=st.text_input("Contraseña",type="password",key="rp1"); np2=st.text_input("Confirmar contraseña",type="password",key="rp2")
            pl=st.selectbox("Plan",["Basic","Pro","Experto"],key="rpl")
            if st.button("Registrarme",type="primary",use_container_width=True):
                if np1!=np2: st.error("❌ Las contraseñas no coinciden.")
                else:
                    ok,msg=register_user(nu,np1,pl)
                    if ok: st.session_state.ok_msg=f"✅ {msg}"; st.session_state.show_reg=False; st.rerun()
                    else: st.error(f"❌ {msg}")
            st.markdown("---")
            if st.button("← Volver al login",use_container_width=True):
                st.session_state.show_reg=False; st.rerun()

# =============================================================================
# TUTORIAL ONBOARDING
# =============================================================================
def mostrar_onboarding():
    st.markdown("### 🎓 ¿Cómo usar PredictaStock?")
    pasos = [
        ("1", "Sube tu archivo", "Ve al panel lateral y carga tu historial de ventas en formato CSV o Excel. Necesitas 3 columnas: Fecha, Producto y Cantidad."),
        ("2", "Explora los gráficos", "En la pestaña Análisis elige qué gráficos quieres ver con el selector. Puedes ver desde 1 hasta 7 gráficos simultáneos según tu plan."),
        ("3", "Revisa las alertas", "En la pestaña Alertas verás qué productos necesitan atención inmediata (semáforo rojo, amarillo o verde)."),
        ("4", "Obtén la predicción", "En la pestaña Predicción selecciona un producto y el sistema calculará automáticamente la demanda de los próximos 7 días, tu stock de seguridad y cuándo reponer."),
        ("5", "Lee el resumen IA", "El sistema genera automáticamente un análisis en lenguaje natural que te explica qué acción tomar (Plan Experto)."),
        ("6", "Descarga el PDF", "Obtén un reporte profesional de 4 páginas con todo el análisis para compartir con tu contador o socio (Plan Experto)."),
    ]
    for num,titulo,desc in pasos:
        st.markdown(f"""<div class='onboarding-step'>
        <div class='step-num'>{num}</div>
        <div><div style='font-weight:600;color:#0f3460;margin-bottom:.2rem;'>{titulo}</div>
        <div style='font-size:.87rem;color:#374151;'>{desc}</div></div></div>""",unsafe_allow_html=True)
    st.markdown("---")
    if st.button("✅ Entendido, comenzar",type="primary",use_container_width=True):
        st.session_state.onboarding_done=True; st.rerun()

# =============================================================================
# PDF
# =============================================================================
def generar_pdf(df,dfa,prod,df_prod,df_pred,dem,ss,pp,mae,fp,fpie,fanio):
    buf=io.BytesIO()
    AZ=colors.HexColor("#0f3460"); NA=colors.HexColor("#e05c2d")
    VE=colors.HexColor("#0d9e6e"); GR=colors.HexColor("#f5f7fa"); GG=colors.HexColor("#e2e8f0"); GS=colors.HexColor("#6b7a90")
    doc=SimpleDocTemplate(buf,pagesize=A4,leftMargin=2*cm,rightMargin=2*cm,topMargin=2.5*cm,bottomMargin=2*cm)
    est=getSampleStyleSheet()
    eTit=ParagraphStyle("t",parent=est["Title"],fontSize=20,textColor=colors.white,fontName="Helvetica-Bold",alignment=TA_CENTER,spaceAfter=4)
    eSub=ParagraphStyle("s",parent=est["Normal"],fontSize=9,textColor=colors.HexColor("#a8b8d8"),fontName="Helvetica",alignment=TA_CENTER,spaceAfter=4)
    eH1=ParagraphStyle("h",parent=est["Heading1"],fontSize=12,textColor=AZ,fontName="Helvetica-Bold",spaceBefore=10,spaceAfter=4)
    eN=ParagraphStyle("n",parent=est["Normal"],fontSize=8.5,textColor=colors.HexColor("#374151"),fontName="Helvetica",leading=13)
    eSm=ParagraphStyle("sm",parent=est["Normal"],fontSize=7,textColor=GS,fontName="Helvetica",leading=11)
    def ib(fig,w,h):
        try:
            d=fig.to_image(format="png",width=int(w/cm*37.8),height=int(h/cm*37.8),scale=2)
            return RLImage(io.BytesIO(d),width=w,height=h)
        except: return Paragraph("[Gráfico no disponible]",eSm)
    W=17*cm; hr=datetime.now().strftime("%d/%m/%Y %H:%M"); hist=[]
    banner=Table([[Paragraph("📦 PredictaStock v5.0",eTit)],[Paragraph("Reporte de Predicción de Demanda",eSub)],[Paragraph(f"Generado: {hr}",eSub)]],colWidths=[W])
    banner.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#16213e")),("ROWPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,0),18),("BOTTOMPADDING",(0,-1),(-1,-1),18)]))
    hist.append(banner); hist.append(Spacer(1,.4*cm))
    res=[["Indicador","Valor"],["Total unidades históricas",f"{int(df['Cantidad'].sum()):,}"],
         ["Período",f"{df['Fecha'].min().strftime('%d/%m/%Y')} — {df['Fecha'].max().strftime('%d/%m/%Y')}"],
         [f"Demanda predicha 7 días — {prod}",f"{dem:.1f} und."],["Stock de seguridad",f"{ss:.1f} und."],
         ["Punto de pedido",f"{pp:.1f} und."],["MAE del modelo",f"{mae:.2f} und."]]
    t=Table(res,colWidths=[10*cm,7*cm])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),AZ),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8.5),("FONTNAME",(0,1),(-1,-1),"Helvetica"),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,GR]),("GRID",(0,0),(-1,-1),.4,GG),("LEFTPADDING",(0,0),(-1,-1),8),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
    hist.append(t); hist.append(PageBreak())
    hist.append(Paragraph("Alertas de Inventario",eH1)); hist.append(HRFlowable(width=W,thickness=1.5,color=AZ,spaceAfter=5))
    for _,r in dfa.iterrows():
        cc={"CRÍTICO":colors.HexColor("#dc2626"),"ATENCIÓN":colors.HexColor("#d97706"),"ESTABLE":VE}.get(r["Nivel"],GS)
        hist.append(Paragraph(f"<b>{r['Nivel']} — {r['Producto']}</b>: {r['Var%']:+.1f}% · {r['Ult']} und. último mes vs {r['Ant']} und. anterior · Mes pico: {r['Pico']}",ParagraphStyle("al",parent=eN,textColor=cc,spaceBefore=3)))
    hist.append(PageBreak())
    hist.append(Paragraph(f"Predicción — {prod}",eH1)); hist.append(HRFlowable(width=W,thickness=1.5,color=NA,spaceAfter=5))
    pd_=[["Día","Fecha","Predicha","Acumulado"]]; ac=0
    for i,row in df_pred.iterrows():
        ac+=row["Cantidad_Predicha"]; pd_.append([str(i+1),row["Fecha"],f"{row['Cantidad_Predicha']:.1f}",f"{ac:.1f}"])
    pd_.append(["—","TOTAL",f"{dem:.1f}",""])
    tp=Table(pd_,colWidths=[1.5*cm,4*cm,6*cm,5.5*cm])
    tp.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),NA),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8.5),("FONTNAME",(0,1),(-1,-1),"Helvetica"),("ROWBACKGROUNDS",(0,1),(-1,-2),[colors.white,GR]),("GRID",(0,0),(-1,-1),.4,GG),("ALIGN",(0,0),(-1,-1),"CENTER"),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
    hist.append(tp); hist.append(PageBreak())
    hist.append(Paragraph("Visualizaciones",eH1)); hist.append(HRFlowable(width=W,thickness=1.5,color=AZ,spaceAfter=6))
    for ttl,fig,ww,hh in [(f"Histórico y Proyección — {prod}",fp,17,8),("Participación de Ventas",fpie,14,7),("Comparativo por Año",fanio,17,7)]:
        hist.append(Paragraph(ttl,eN)); hist.append(Spacer(1,.25*cm)); hist.append(ib(fig,ww*cm,hh*cm)); hist.append(Spacer(1,.4*cm))
    hist.append(Spacer(1,.8*cm)); hist.append(HRFlowable(width=W,thickness=.5,color=GG,spaceAfter=4))
    hist.append(Paragraph(f"PredictaStock v5.0 · {hr} · RandomForestRegressor · Nivel servicio 95% · Lead time 3d",eSm))
    doc.build(hist); buf.seek(0); return buf

# =============================================================================
# APP PRINCIPAL
# =============================================================================
def app_principal():
    plan=st.session_state.user_plan; usuario=st.session_state.username; P=PLANES[plan]
    badge={"Basic":"badge-basic","Pro":"badge-pro","Experto":"badge-experto"}[plan]

    # Onboarding primera vez
    if "onboarding_done" not in st.session_state:
        st.session_state.onboarding_done=False
    if not st.session_state.onboarding_done:
        mostrar_onboarding(); return

    # ── Sidebar ──
    with st.sidebar:
        st.markdown(f"### 👤 {usuario}")
        st.markdown(f"<span class='{badge}'>Plan {plan}</span>",unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("## 📁 Cargar Datos")
        archivo=st.file_uploader("CSV o Excel (.xlsx)",type=["csv","xlsx"])
        st.markdown("---")
        st.markdown("### ℹ️ Estructura requerida")
        st.markdown("| Columna | Ejemplo |\n|---|---|\n| `Fecha` | 2024-03-15 |\n| `Producto` | Polo Básico |\n| `Cantidad` | 12 |")
        st.markdown("---")
        if st.button("🎓 Ver tutorial",use_container_width=True):
            st.session_state.onboarding_done=False; st.rerun()
        if st.button("🚪 Cerrar sesión",use_container_width=True):
            for k in ["logged_in","username","user_plan","onboarding_done"]:
                st.session_state.pop(k,None)
            st.rerun()
        st.caption("PredictaStock v5.0 · 2025")

    # ── Header ──
    st.markdown(f"""<div class="main-header">
    <h1>📦 PredictaStock v5.0</h1>
    <p>Predicción de Demanda con ML para MyPEs · Sesión: <b>{usuario}</b> &nbsp;·&nbsp;
    <span class='{badge}' style='font-size:.73rem;'>Plan {plan}</span></p></div>""",unsafe_allow_html=True)

    if archivo is None:
        c1,c2=st.columns([3,2])
        with c1:
            st.markdown("""<div style='background:#eef3ff;border:1px solid #c5d3f5;border-radius:10px;padding:1.1rem 1.4rem;color:#2a3a6e;font-size:.88rem;line-height:1.65;'>
            👋 <b>Bienvenido a PredictaStock v5.0</b><br><br>Sube tu historial de ventas para comenzar.</div>""",unsafe_allow_html=True)
            st.markdown("**📋 Estructura del archivo:**")
            st.dataframe(ejemplo_df(),use_container_width=True,hide_index=True)
        with c2:
            st.markdown("**✅ Tu plan incluye:**")
            st.markdown(f"- Hasta **{P['graficos']}** gráfico(s)\n- {'✅' if P['alertas'] else '❌'} Alertas de stock\n- {'✅' if P['multi'] else '❌'} Pronóstico multi-producto\n- {'✅' if P['prediccion'] else '❌'} Predicción ML individual\n- {'✅' if P['ia'] else '❌'} Resumen IA automático\n- {'✅' if P['pdf'] else '❌'} Reporte PDF")
        return

    # ── Carga ──
    try:
        df_raw=pd.read_csv(archivo) if archivo.name.endswith(".csv") else pd.read_excel(archivo)
    except Exception as e:
        st.error(f"❌ {e}"); return
    df=limpiar(df_raw)
    if not validar(df):
        st.error(f"❌ Columnas encontradas: {list(df.columns)}. Se necesitan: Fecha, Producto, Cantidad."); return
    df["Anio"]=df["Fecha"].dt.year; df["Mes"]=df["Fecha"].dt.month
    anios=sorted(df["Anio"].unique()); dfa=alertas(df)
    yoy_res=calcular_yoy(df)

    # ── Tabs ──
    tabs_list=["📊 Análisis"]
    if P["alertas"]:    tabs_list.append("🚨 Alertas")
    if P["multi"]:      tabs_list.append("🔮 Multi-Producto")
    if P["prediccion"]: tabs_list.append("📍 Predicción Individual")
    tabs=st.tabs(tabs_list)

    # ════════════════════════════════════════════════════════════
    # TAB ANÁLISIS
    # ════════════════════════════════════════════════════════════
    with tabs[0]:
        # Métricas rápidas
        tot=int(df["Cantidad"].sum()); top=df.groupby("Producto")["Cantidad"].sum().idxmax()
        c1,c2,c3,c4=st.columns(4)
        with c1: st.markdown(f"<div class='metric-card'><div class='ml'>Total Unidades</div><div class='mv'>{tot:,}</div><div class='ms'>historial</div></div>",unsafe_allow_html=True)
        with c2: st.markdown(f"<div class='metric-card g'><div class='ml'>Productos</div><div class='mv'>{df['Producto'].nunique()}</div><div class='ms'>distintos</div></div>",unsafe_allow_html=True)
        with c3: st.markdown(f"<div class='metric-card o'><div class='ml'>Años de datos</div><div class='mv'>{len(anios)}</div><div class='ms'>{' · '.join(map(str,anios))}</div></div>",unsafe_allow_html=True)
        with c4: st.markdown(f"<div class='metric-card p'><div class='ml'>Producto estrella</div><div class='mv' style='font-size:.9rem'>{top}</div><div class='ms'>mayor volumen</div></div>",unsafe_allow_html=True)
        st.markdown("---")

        # ── Tendencia YoY ──
        if yoy_res:
            yoy_df,a1,a2=yoy_res
            st.markdown(f'<div style="font-size:1rem;font-weight:600;color:#1a1a2e;margin-bottom:.5rem;">📈 Tendencia de Crecimiento {a1} → {a2}</div>',unsafe_allow_html=True)
            cc1,cc2=st.columns([3,2])
            with cc1:
                # FIX: key único para evitar StreamlitDuplicateElementId
                st.plotly_chart(g_yoy(yoy_df,a1,a2),use_container_width=True,key="chart_yoy")
            with cc2:
                st.markdown("**Detalle por producto:**")
                for _,r in yoy_df.iterrows():
                    pct=r["YoY%"]
                    cls="trend-up" if pct>0 else("trend-dn" if pct<-5 else "trend-eq")
                    arrow="↑" if pct>0 else("↓" if pct<0 else "→")
                    st.markdown(f"**{r['Producto']}** &nbsp; <span class='{cls}'>{arrow} {pct:+.1f}%</span>",unsafe_allow_html=True)
            st.markdown("---")

        # ── Selector de gráficos ──
        OPCIONES=["🏆 Ranking de Productos","🥧 Participación (Pie)","📅 Comparativo por Año",
                  "📆 Ventas Mensuales","🌡️ Mapa de Calor","📊 Variación % Alertas"]
        BASIC=OPCIONES[:2]; MAX=P["graficos"]
        opts=BASIC if plan=="Basic" else OPCIONES
        default=opts[:min(2,MAX)]

        cs,ci=st.columns([3,1])
        with cs:
            sel=st.multiselect(f"🎛️ Elige hasta {MAX} gráfico(s):",options=opts,default=default,
                               max_selections=MAX,key="gsel",
                               help=f"Plan {plan}: hasta {MAX} gráficos simultáneos.")
        with ci:
            st.markdown(f"""<div style='background:#f0fdf4;border:1px solid #86efac;border-radius:8px;
            padding:.7rem;margin-top:1.4rem;text-align:center;'>
            <div style='font-size:.7rem;color:#14532d;font-weight:600;'>SELECCIONADOS</div>
            <div style='font-size:1.7rem;font-weight:700;color:#0d9e6e;'>{len(sel)}/{MAX}</div></div>""",
            unsafe_allow_html=True)

        if not sel: st.info("👆 Selecciona al menos un gráfico."); return

        # FIX: keys únicos por posición en el grid para cada plotly_chart
        for i in range(0,len(sel),2):
            grupo=sel[i:i+2]; cols=st.columns(len(grupo))
            for j,g in enumerate(grupo):
                with cols[j]:
                    if g=="🏆 Ranking de Productos":
                        st.plotly_chart(g_ranking(df),use_container_width=True,key=f"chart_ranking_{i}_{j}")
                    elif g=="🥧 Participación (Pie)":
                        st.plotly_chart(g_pie(df),use_container_width=True,key=f"chart_pie_{i}_{j}")
                    elif g=="📅 Comparativo por Año":
                        st.plotly_chart(g_anio(df),use_container_width=True,key=f"chart_anio_{i}_{j}")
                    elif g=="📆 Ventas Mensuales":
                        st.plotly_chart(g_mensual(df),use_container_width=True,key=f"chart_mensual_{i}_{j}")
                    elif g=="🌡️ Mapa de Calor":
                        ah=st.selectbox("Año",anios,index=len(anios)-1,key=f"ht{i}{j}")
                        st.plotly_chart(g_heat(df.copy(),ah),use_container_width=True,key=f"chart_heat_{i}_{j}")
                    elif g=="📊 Variación % Alertas":
                        # FIX: key distinto al del tab Alertas para evitar duplicado
                        st.plotly_chart(g_var(dfa),use_container_width=True,key=f"chart_var_analisis_{i}_{j}")

        # Filtro mes
        st.markdown("---")
        st.markdown('<div style="font-size:1rem;font-weight:600;color:#1a1a2e;margin-bottom:.5rem;">🔍 ¿Qué producto lideró en un mes?</div>',unsafe_allow_html=True)
        fa,fb=st.columns(2)
        with fa: anf=st.selectbox("Año",anios,key="fa")
        with fb: mf=st.selectbox("Mes",list(MESES.values()),key="fm")
        mn={v:k for k,v in MESES.items()}[mf]; dff=df[(df["Anio"]==anf)&(df["Mes"]==mn)]
        if len(dff)>0:
            tm=dff.groupby("Producto")["Cantidad"].sum().reset_index().sort_values("Cantidad",ascending=False)
            gn=tm.iloc[0]
            st.markdown(f"""<div style='background:#eef3ff;border:1px solid #c5d3f5;border-radius:10px;
            padding:.9rem 1.3rem;color:#2a3a6e;font-size:.88rem;'>
            🥇 <b>{gn['Producto']}</b> lideró en <b>{mf} {anf}</b> con <b>{int(gn['Cantidad']):,} unidades</b>.</div>""",
            unsafe_allow_html=True)
            fm2=px.bar(tm,x="Producto",y="Cantidad",color="Cantidad",
                       color_continuous_scale="Blues",text="Cantidad",title=f"Ventas — {mf} {anf}")
            fm2.update_traces(texttemplate="%{text:,.0f}",textposition="outside",textfont=dict(color="#1a1a2e",size=11))
            fm2.update_layout(**LB,coloraxis_showscale=False)
            st.plotly_chart(fm2,use_container_width=True,key="chart_filtro_mes")

    # ════════════════════════════════════════════════════════════
    # TAB ALERTAS
    # ════════════════════════════════════════════════════════════
    if P["alertas"] and "🚨 Alertas" in tabs_list:
        with tabs[tabs_list.index("🚨 Alertas")]:
            crit=dfa[dfa["Nivel"]=="CRÍTICO"]; ate=dfa[dfa["Nivel"]=="ATENCIÓN"]; est=dfa[dfa["Nivel"]=="ESTABLE"]
            sa,sb,sc=st.columns(3)
            with sa: st.markdown(f"<div class='metric-card r'><div class='ml'>🔴 Crítico</div><div class='mv'>{len(crit)}</div><div class='ms'>acción urgente</div></div>",unsafe_allow_html=True)
            with sb: st.markdown(f"<div class='metric-card o'><div class='ml'>🟡 Atención</div><div class='mv'>{len(ate)}</div><div class='ms'>monitorear</div></div>",unsafe_allow_html=True)
            with sc: st.markdown(f"<div class='metric-card g'><div class='ml'>🟢 Estable</div><div class='mv'>{len(est)}</div><div class='ms'>sin cambios</div></div>",unsafe_allow_html=True)
            st.markdown("---")
            for _,r in crit.iterrows():
                st.markdown(f"<div class='ar'><b>{r['Producto']}</b> · Caída <b>{abs(r['Var%'])}%</b> · {r['Ult']} und. vs {r['Ant']} und. · Mes pico: <b>{r['Pico']}</b><br><em>→ Revisar precio y visibilidad. Stock antes de {r['Pico']}.</em></div>",unsafe_allow_html=True)
            for _,r in ate.iterrows():
                st.markdown(f"<div class='ay'><b>{r['Producto']}</b> · {r['Var%']}% · Mes pico: <b>{r['Pico']}</b><br><em>→ Reducir próximo pedido 10-15%.</em></div>",unsafe_allow_html=True)
            for _,r in est.iterrows():
                st.markdown(f"<div class='ag'><b>{r['Producto']}</b> · <b>+{r['Var%']}%</b> · Mes pico: <b>{r['Pico']}</b><br><em>→ Mantener política actual.</em></div>",unsafe_allow_html=True)
            st.markdown("---")
            # FIX: key distinto al del tab Análisis
            st.plotly_chart(g_var(dfa),use_container_width=True,key="chart_var_alertas")

    # ════════════════════════════════════════════════════════════
    # TAB MULTI-PRODUCTO (Pro + Experto)
    # ════════════════════════════════════════════════════════════
    if P["multi"] and "🔮 Multi-Producto" in tabs_list:
        with tabs[tabs_list.index("🔮 Multi-Producto")]:
            st.markdown("### 🔮 Pronóstico simultáneo de todos los productos")
            st.caption("El modelo se entrena por separado para cada producto y calcula la predicción de los próximos 7 días en paralelo.")

            prods_disponibles=sorted(df["Producto"].unique().tolist())
            prods_sel=st.multiselect("Selecciona productos a predecir:",options=prods_disponibles,
                                     default=prods_disponibles[:min(5,len(prods_disponibles))],key="multi_sel")
            if not prods_sel: st.info("Selecciona al menos un producto."); return

            if st.button("▶ Calcular pronóstico multi-producto",type="primary"):
                resultados={}; mae_dict={}
                pb=st.progress(0,"Entrenando modelos...")
                for idx,pr in enumerate(prods_sel):
                    dp=df[df["Producto"]==pr][["Fecha","Cantidad"]].sort_values("Fecha").reset_index(drop=True)
                    if len(dp)>=10:
                        m,mae,ff=entrenar(dp); dpred=predecir(m,ff)
                        resultados[pr]=dpred; mae_dict[pr]=mae
                    pb.progress((idx+1)/len(prods_sel),f"Procesando {pr}...")
                pb.empty()
                st.session_state.multi_result=resultados
                st.session_state.multi_mae=mae_dict

            if "multi_result" in st.session_state and st.session_state.multi_result:
                resultados=st.session_state.multi_result; mae_dict=st.session_state.multi_mae
                # FIX: key único para el gráfico multi
                st.plotly_chart(g_multi(df,resultados,mae_dict),use_container_width=True,key="chart_multi")
                st.markdown("---")
                st.markdown("**📋 Tabla comparativa de demanda predicha (próximos 7 días):**")
                rows_comp=[]
                for pr,dpred in resultados.items():
                    tot_p=dpred["Cantidad_Predicha"].sum(); mae_p=mae_dict.get(pr,0)
                    ss_p=round(mae_p*1.65,1); pp_p=round(tot_p/7*3+ss_p,1)
                    rows_comp.append({"Producto":pr,"Demanda 7d":f"{tot_p:.1f}","Prom/día":f"{tot_p/7:.1f}","Stock Seg.":f"{ss_p:.1f}","Pto.Pedido":f"{pp_p:.1f}","MAE":f"{mae_p:.2f}"})
                df_comp=pd.DataFrame(rows_comp)
                st.dataframe(df_comp,use_container_width=True,hide_index=True)
                # Exportar CSV
                csv_buf=df_comp.to_csv(index=False).encode()
                st.download_button("📥 Exportar tabla como CSV",data=csv_buf,
                                   file_name=f"prediccion_multi_{datetime.now().strftime('%Y%m%d')}.csv",
                                   mime="text/csv",use_container_width=True)

    # ════════════════════════════════════════════════════════════
    # TAB PREDICCIÓN INDIVIDUAL + IA + SIMULADOR (Experto)
    # ════════════════════════════════════════════════════════════
    if P["prediccion"] and "📍 Predicción Individual" in tabs_list:
        with tabs[tabs_list.index("📍 Predicción Individual")]:
            prods=sorted(df["Producto"].unique().tolist())
            prod_sel=st.selectbox("🛒 Producto a predecir:",prods)
            df_prod=df[df["Producto"]==prod_sel][["Fecha","Cantidad"]].sort_values("Fecha").reset_index(drop=True)
            if len(df_prod)<10:
                st.warning(f"⚠️ Solo {len(df_prod)} registros. Mínimo 10."); return
            with st.spinner("🤖 Entrenando modelo..."):
                modelo,mae,df_f=entrenar(df_prod)
            df_pred=predecir(modelo,df_f); dem,ss,pp=metricas(df_pred,mae)

            m1,m2,m3,m4=st.columns(4)
            with m1: st.markdown(f"<div class='metric-card'><div class='ml'>Demanda 7 días</div><div class='mv'>{dem:,.1f}</div><div class='ms'>unidades</div></div>",unsafe_allow_html=True)
            with m2: st.markdown(f"<div class='metric-card g'><div class='ml'>Stock Seguridad</div><div class='mv'>{ss:,.1f}</div><div class='ms'>nivel 95%</div></div>",unsafe_allow_html=True)
            with m3: st.markdown(f"<div class='metric-card o'><div class='ml'>Punto de Pedido</div><div class='mv'>{pp:,.1f}</div><div class='ms'>lead time 3d</div></div>",unsafe_allow_html=True)
            with m4: st.markdown(f"<div class='metric-card'><div class='ml'>MAE del modelo</div><div class='mv'>{mae:.2f}</div><div class='ms'>und. promedio</div></div>",unsafe_allow_html=True)

            fig_pred_plot=g_pred(df_prod,df_pred,prod_sel)
            # FIX: key único para el gráfico de predicción individual
            st.plotly_chart(fig_pred_plot,use_container_width=True,key="chart_pred_individual")

            # ── Simulador de escenarios ──────────────────────────
            st.markdown("---")
            st.markdown("### 🎲 Simulador de Escenarios")
            st.caption("¿Qué pasa con el stock de seguridad y el punto de pedido si la demanda varía?")
            sc1,sc2=st.columns([2,1])
            with sc1:
                delta=st.slider("% de variación en la demanda:",-50,100,0,5,
                                 format="%d%%",key="sim_delta")
            with sc2:
                lt=st.number_input("Lead time proveedor (días):",min_value=1,max_value=30,value=3,key="sim_lt")
            dem_sim=dem*(1+delta/100); ss_sim=round(mae*1.65,1); pp_sim=round(dem_sim/7*lt+ss_sim,1)
            arrow_dem="↑" if delta>0 else("↓" if delta<0 else "→"); arrow_pp="↑" if pp_sim>pp else("↓" if pp_sim<pp else "→")
            st.markdown(f"""<div class='sim-box'>
            <b>Escenario: demanda {arrow_dem} {delta:+d}%</b><br>
            Demanda estimada 7 días: <b>{dem_sim:,.1f} und.</b> &nbsp;|&nbsp;
            Stock de seguridad: <b>{ss_sim:.1f} und.</b> &nbsp;|&nbsp;
            Punto de pedido: <b>{pp_sim:.1f} und.</b> {arrow_pp} (antes: {pp:.1f})
            </div>""",unsafe_allow_html=True)

            # ── Resumen IA ───────────────────────────────────────
            st.markdown("---")
            st.markdown("### 🤖 Resumen IA Automático")
            if st.button("✨ Generar resumen con IA",type="primary",use_container_width=True):
                with st.spinner("Generando análisis con Claude..."):
                    yoy_df_ia=yoy_res[0] if yoy_res else None
                    resumen=generar_resumen_ia(df,dfa,prod_sel,dem,ss,pp,mae,yoy_df_ia)
                st.markdown(f"""<div class='ai-box'>
                <div class='ai-label'>🤖 Análisis generado por Claude Sonnet</div>
                {resumen}</div>""",unsafe_allow_html=True)

            # ── Tabla + CSV ──────────────────────────────────────
            st.markdown("---")
            ct,cr=st.columns([2,1])
            with ct:
                df_tbl=df_pred.copy(); df_tbl.columns=["Fecha","Cantidad Predicha (und.)"]
                df_tbl["Cantidad Predicha (und.)"]=df_tbl["Cantidad Predicha (und.)"].apply(lambda x:f"{x:.1f}")
                st.dataframe(df_tbl,use_container_width=True,hide_index=True)
                csv_pred=df_pred.to_csv(index=False).encode()
                st.download_button("📥 Exportar predicción como CSV",data=csv_pred,
                                   file_name=f"prediccion_{prod_sel.replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                                   mime="text/csv",use_container_width=True)
            with cr:
                st.markdown(f"**Producto:** {prod_sel}\n\n**Total:** {dem:,.1f} und.\n\n**Prom/día:** {dem/7:.1f} und.\n\n**Pico:** {df_pred['Cantidad_Predicha'].max():.1f} und.")
                st.markdown("---")
                st.markdown(f"**🔔 Reponer cuando stock = {pp:,.1f} und.**")

            # ── PDF ──────────────────────────────────────────────
            if P["pdf"]:
                st.markdown("---")
                st.markdown("### 📄 Reporte PDF")
                if st.button("📥 Generar y Descargar PDF",type="primary",use_container_width=True):
                    with st.spinner("Generando PDF..."):
                        pie_d=df.groupby("Producto")["Cantidad"].sum().reset_index()
                        fpie=px.pie(pie_d,names="Producto",values="Cantidad",color_discrete_sequence=px.colors.sequential.Blues_r,hole=.35)
                        fpie.update_traces(textfont=dict(color="#1a1a2e")); fpie.update_layout(showlegend=False,paper_bgcolor="white",font=dict(color="#1a1a2e"))
                        va=df.copy(); va["Anio"]=va["Fecha"].dt.year
                        vad=va.groupby(["Anio","Producto"])["Cantidad"].sum().reset_index().rename(columns={"Anio":"Año"})
                        fanio=px.bar(vad,x="Producto",y="Cantidad",color="Año",barmode="group",color_discrete_sequence=["#0f3460","#e05c2d","#0d9e6e"])
                        fanio.update_traces(textfont=dict(color="#1a1a2e")); fanio.update_layout(paper_bgcolor="white",font=dict(color="#1a1a2e"))
                        buf=generar_pdf(df,dfa,prod_sel,df_prod,df_pred,dem,ss,pp,mae,fig_pred_plot,fpie,fanio)
                    nombre=f"PredictaStock_{prod_sel.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                    st.download_button("⬇️ Descargar PDF",data=buf,file_name=nombre,mime="application/pdf",use_container_width=True)

            with st.expander("🔍 Parámetros técnicos del modelo"):
                st.markdown(f"|Parámetro|Valor|\n|---|---|\n|Algoritmo|RandomForestRegressor|\n|Árboles|200|\n|Profundidad|6|\n|Features|mes, dia_semana, anio, lag_1|\n|Registros entren.|{len(df_f)-min(15,len(df_f)-1)}|\n|Registros valid.|{min(15,len(df_f)-1)}|\n|MAE|{mae:.4f} und.|\n|Nivel servicio|95% (Z=1.65)|\n|Lead time|3 días|")

# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
init_users()
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    pantalla_login()
else:
    app_principal()