#!/usr/bin/env python3
"""Genera docs/predicciones.json: Dixon-Coles para las 5 grandes ligas.
Solo usa la librería estándar. Fuente: football-data.co.uk"""
import csv, difflib, io, json, math, os, re, unicodedata, urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

LIGAS = {"E0": "Premier League", "SP1": "LaLiga", "I1": "Serie A",
         "D1": "Bundesliga", "F1": "Ligue 1"}
TEMPORADAS = 3        # temporadas de historial
XI = 0.0025           # decaimiento diario (partidos recientes pesan más)
MIN_PARTIDOS = 10     # mínimo de partidos por equipo para predecir
BASE = "https://www.football-data.co.uk"
PAIS = {"E0": "ENG", "SP1": "ESP", "I1": "ITA", "D1": "GER", "F1": "FRA"}
PRIOR = 3.0          # contracción hacia el promedio para equipos con poca muestra (Champions)
XIS = (0.0012, 0.0025, 0.0050)   # memoria larga / central / corta: mide la sensibilidad
CUOTAS = {}          # (fecha, local, visitante) -> prob. implícitas del mercado, para el backtest
MIN_CL = 6           # mínimo de partidos por equipo para predecir Champions
OF_URL = "https://raw.githubusercontent.com/openfootball/champions-league/master/{t}/cl.txt"
SALIDA = Path(__file__).parent / "docs" / "predicciones.json"


def bajar(url, headers=None):
    h = {"User-Agent": "Mozilla/5.0", **(headers or {})}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=30) as r:
            return r.read().decode("utf-8-sig", "replace")
    except Exception as e:
        print(f"  ! no se pudo descargar {url}: {e}")
        return ""


def descargar(url):
    return list(csv.DictReader(io.StringIO(bajar(url))))


def fecha(s):
    for f in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime((s or "").strip(), f).date()
        except ValueError:
            pass
    return None


def temporadas():
    hoy = date.today()
    y = hoy.year if hoy.month >= 7 else hoy.year - 1
    return [f"{(y - k) % 100:02d}{(y - k + 1) % 100:02d}" for k in range(TEMPORADAS)]


def historial(div):
    out = []
    for t in temporadas():
        for r in descargar(f"{BASE}/mmz4281/{t}/{div}.csv"):
            d = fecha(r.get("Date"))
            try:
                hg, ag = int(r["FTHG"]), int(r["FTAG"])
            except (KeyError, ValueError, TypeError):
                continue
            if d and r.get("HomeTeam") and r.get("AwayTeam"):
                out.append((d, r["HomeTeam"], r["AwayTeam"], hg, ag))
                CUOTAS[(d, r["HomeTeam"], r["AwayTeam"])] = mercado(r)
    return out


def pois(k, l):
    return math.exp(-l) * l ** k / math.factorial(k)


def tau(x, y, l, m, r):
    if x == 0 and y == 0: return 1 - l * m * r
    if x == 0 and y == 1: return 1 + l * r
    if x == 1 and y == 0: return 1 + m * r
    if x == 1 and y == 1: return 1 - r
    return 1.0


def ajustar(ms, prior=0.0, xi=XI):
    """Ataque/defensa/localía por punto fijo ponderado + rho de Dixon-Coles."""
    hoy = max(m[0] for m in ms)
    w = [math.exp(-xi * (hoy - m[0]).days) for m in ms]
    eq = sorted({m[1] for m in ms} | {m[2] for m in ms})
    n = dict.fromkeys(eq, 0)
    for m in ms:
        n[m[1]] += 1; n[m[2]] += 1
    at, df, g = dict.fromkeys(eq, 1.0), dict.fromkeys(eq, 1.0), 1.3
    for _ in range(100):
        # 1) ataque (con defensa y localía actuales), normalizado a media 1
        nA, dA = {t: 1e-9 for t in eq}, {t: 1e-9 for t in eq}
        for (d, h, a, hg, ag), wi in zip(ms, w):
            nA[h] += wi * hg; dA[h] += wi * df[a] * g
            nA[a] += wi * ag; dA[a] += wi * df[h]
        for t in eq:
            at[t] = (nA[t] + prior) / (dA[t] + prior)
        mean = sum(at.values()) / len(at)
        for t in eq:
            at[t] /= mean
        # 2) defensa con el ataque ya actualizado
        nD, dD = {t: 1e-9 for t in eq}, {t: 1e-9 for t in eq}
        for (d, h, a, hg, ag), wi in zip(ms, w):
            nD[a] += wi * hg; dD[a] += wi * at[h] * g
            nD[h] += wi * ag; dD[h] += wi * at[a]
        for t in eq:
            df[t] = (nD[t] + prior) / (dD[t] + prior)
        # 3) ventaja de localía
        sg = sl = 0.0
        for (d, h, a, hg, ag), wi in zip(ms, w):
            sg += wi * hg; sl += wi * at[h] * df[a]
        g = sg / sl
    mejor, bl = 0.0, -1e18
    for k in range(-30, 11):
        r, ll = k / 100, 0.0
        for (d, h, a, hg, ag), wi in zip(ms, w):
            if hg > 1 or ag > 1:
                continue
            t = tau(hg, ag, at[h] * df[a] * g, at[a] * df[h], r)
            ll += wi * math.log(t) if t > 0 else -1e6
        if ll > bl:
            bl, mejor = ll, r
    return {"at": at, "df": df, "g": g, "rho": mejor, "n": n}


def predecir(M, h, a, N=10):
    l = M["at"][h] * M["df"][a] * M["g"]
    m = M["at"][a] * M["df"][h]
    P, s = {}, 0.0
    for x in range(N):
        for y in range(N):
            q = pois(x, l) * pois(y, m) * max(tau(x, y, l, m, M["rho"]), 0.0)
            P[(x, y)] = q; s += q
    w = d = v = o15 = o25 = o35 = b = csl = csv = h2 = 0.0
    for (x, y), q in P.items():
        q /= s
        if x > y: w += q
        elif x == y: d += q
        else: v += q
        t = x + y
        if t > 1: o15 += q
        if t > 2: o25 += q
        if t > 3: o35 += q
        if x > 0 and y > 0: b += q
        if y == 0: csl += q
        if x == 0: csv += q
        if x - y >= 2: h2 += q
    top = sorted(P.items(), key=lambda kv: -kv[1])[:6]
    r4 = lambda z: round(z, 4)
    return {"xg_l": round(l, 2), "xg_v": round(m, 2), "p1": r4(w), "px": r4(d), "p2": r4(v),
            "o15": r4(o15), "o25": r4(o25), "o35": r4(o35), "btts": r4(b),
            "cs_l": r4(csl), "cs_v": r4(csv), "h2": r4(h2),
            "top": [[f"{x}-{y}", r4(q / s)] for (x, y), q in top]}


def indice(ms):
    idx = {}
    for d, h, a, hg, ag in sorted(ms, key=lambda m: m[0], reverse=True):
        idx.setdefault(h, []).append((a, hg, ag, "L"))
        idx.setdefault(a, []).append((h, ag, hg, "V"))
    return idx


def forma(idx, t):
    return [["G" if gf > gc else "E" if gf == gc else "P", f"{gf}-{gc}", rival, lv]
            for rival, gf, gc, lv in idx.get(t, [])[:5]]


def confianza(n1, n2, rango, cl):
    amp = max(r[1] - r[0] for r in rango.values())
    nmin = min(n1, n2)
    nivel = "Alta" if nmin >= 60 and amp <= 0.06 else "Media" if nmin >= 25 and amp <= 0.12 else "Baja"
    razones = [f"Historial usado: {n1} y {n2} partidos",
               f"Las probabilidades varían hasta {amp * 100:.0f} pts según cuánta memoria se le dé al pasado"]
    if cl:
        if nivel == "Alta":
            nivel = "Media"
        razones.append("Si los equipos son de ligas distintas, la comparación depende de pocos cruces de Champions")
        if nmin < 25:
            razones.append("Algún equipo tiene poco historial: su rating se acerca al promedio")
    return nivel, razones


def completar(modelos, idx, h, a, cl=False):
    ps = [predecir(M, h, a) for M in modelos]
    pr = ps[len(ps) // 2]
    rango = {k: [min(q[k] for q in ps), max(q[k] for q in ps)] for k in ("p1", "px", "p2")}
    M = modelos[len(modelos) // 2]
    nivel, razones = confianza(M["n"][h], M["n"][a], rango, cl)
    info = lambda t: {"ataque": round(M["at"][t], 2), "defensa": round(M["df"][t], 2),
                      "n": M["n"][t], "forma": forma(idx, t)}
    pr.update(rango=rango, confianza=nivel, razones=razones, info_l=info(h), info_v=info(a))
    return pr


def backtest(por_liga, dias=300, paso=30):
    """Predice cada partido solo con datos anteriores y mide qué tan bien acertó."""
    filas = []
    for ms in por_liga.values():
        fin = max(m[0] for m in ms)
        c = fin - timedelta(days=dias)
        while c <= fin:
            train = [m for m in ms if m[0] < c]
            test = [m for m in ms if c <= m[0] < c + timedelta(days=paso)]
            if len(train) >= 300 and test:
                M = ajustar(train)
                for d, h, a, hg, ag in test:
                    if M["n"].get(h, 0) < MIN_PARTIDOS or M["n"].get(a, 0) < MIN_PARTIDOS:
                        continue
                    q = predecir(M, h, a)
                    filas.append(([q["p1"], q["px"], q["p2"]],
                                  0 if hg > ag else 1 if hg == ag else 2, CUOTAS.get((d, h, a))))
            c += timedelta(days=paso)
    if not filas:
        return None
    n = len(filas)
    ll = lambda pr, k: -math.log(max(pr[k], 1e-9))
    con = [f for f in filas if f[2]]
    bins = [("menos de 40%", 0, .4), ("40–50%", .4, .5), ("50–60%", .5, .6), ("60–70%", .6, .7), ("70% o más", .7, 1.01)]
    tabla = []
    for nom, lo, hi in bins:
        sel = [f for f in filas if lo <= max(f[0]) < hi]
        if sel:
            tabla.append([nom, len(sel), round(sum(max(f[0]) for f in sel) / len(sel), 3),
                          round(sum(max(range(3), key=lambda i: f[0][i]) == f[1] for f in sel) / len(sel), 3)])
    return {"n": n,
            "acierto": round(sum(max(range(3), key=lambda i: f[0][i]) == f[1] for f in filas) / n, 3),
            "base_local": round(sum(f[1] == 0 for f in filas) / n, 3),
            "logloss": round(sum(ll(f[0], f[1]) for f in filas) / n, 4),
            "n_mercado": len(con),
            "logloss_modelo_vs_mercado": round(sum(ll(f[0], f[1]) for f in con) / len(con), 4) if con else None,
            "logloss_mercado": round(sum(ll(f[2], f[1]) for f in con) / len(con), 4) if con else None,
            "bins": tabla}


def mercado(r):
    """Probabilidades implícitas (sin margen) de la casa, si el CSV trae cuotas."""
    for pref in ("B365", "Avg", "PS"):
        try:
            inv = [1 / float(r[pref + k]) for k in "HDA"]
            s = sum(inv)
            return [round(i / s, 4) for i in inv]
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            continue
    return None


RUIDO = {"fc", "cf", "afc", "sc", "ac", "as", "ss", "ssc", "ud", "cd", "rcd", "rc", "bk", "kv",
         "fk", "sk", "cfc", "bc", "sv", "vfb", "vfl", "fsv", "tsg", "ogc", "losc", "us",
         "de", "del", "la", "le", "los", "club", "calcio", "balompie"}
# Casos donde el nombre largo no se parece al de football-data.co.uk (clave ya normalizada).
# Si un equipo no se empareja, agrégalo aquí.
ALIAS = {
    "manchester city": "Man City", "manchester united": "Man United",
    "tottenham hotspur": "Tottenham", "newcastle united": "Newcastle",
    "wolverhampton wanderers": "Wolves", "nottingham forest": "Nott'm Forest",
    "leeds united": "Leeds", "brighton hove albion": "Brighton",
    "athletic": "Ath Bilbao", "athletic bilbao": "Ath Bilbao", "atletico madrid": "Ath Madrid",
    "real betis": "Betis", "real sociedad": "Sociedad", "celta vigo": "Celta",
    "rayo vallecano": "Vallecano", "espanyol": "Espanol", "deportivo alaves": "Alaves",
    "internazionale milano": "Inter", "internazionale": "Inter", "hellas verona": "Verona",
    "bayern munchen": "Bayern Munich", "borussia monchengladbach": "M'gladbach",
    "eintracht frankfurt": "Ein Frankfurt", "borussia dortmund": "Dortmund",
    "bayer leverkusen": "Leverkusen", "paris saint germain": "Paris SG",
    "olympique lyonnais": "Lyon", "olympique marseille": "Marseille",
    "stade rennais": "Rennes", "stade brestois": "Brest", "saint etienne": "St Etienne",
}


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return " ".join(w for w in re.sub(r"[^a-z0-9]+", " ", s).split()
                    if w not in RUIDO and not w.isdigit())


class Nombres:
    """Unifica nombres de equipos entre football-data.co.uk, openfootball y football-data.org."""
    def __init__(self):
        self.liga, self.visto = {}, {}

    def registrar_liga(self, pais, equipos):
        self.liga[pais] = {norm(e): e for e in equipos}
        self.visto.update(self.liga[pais])

    def canon(self, nombre, pais=None):
        k = norm(nombre)
        if k in self.visto:
            return self.visto[k]
        res = ALIAS.get(k)
        cand = self.liga.get(pais) if pais else {c: n for d in self.liga.values() for c, n in d.items()}
        if not res and cand:
            tk = set(k.split())
            ms = [(len(set(c.split()) & tk), n) for c, n in cand.items()
                  if c and (set(c.split()) <= tk or tk <= set(c.split()))]
            if len(ms) == 1 or (len(ms) > 1 and sorted(ms)[-1][0] > sorted(ms)[-2][0]):
                res = max(ms)[1]
            else:
                cerca = difflib.get_close_matches(k, cand.keys(), 1, 0.85)
                res = cand[cerca[0]] if cerca else None
        res = res or nombre.strip()
        self.visto[k] = res
        return res


MESES = {m: i for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), 1)}
RE_DIA = re.compile(r"^\s+(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z][a-z]{2})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$")
RE_EQ = re.compile(r"^\s*(?:\d{1,2}:\d{2}\s+)?(.+?)\s+\(([A-Z]{3})\)\s+v\s+(.+?)\s+\(([A-Z]{3})\)\s")
RE_90 = re.compile(r"\((\d+)-(\d+),\s*\d+-\d+\)\s*$")          # prórroga/penales: 1.º = 90 min
RE_FT = re.compile(r"\s(\d+)-(\d+)(?:\s*\(\d+-\d+\))?\s*$")


def cl_texto(txt, nombres):
    """Resultados (90 min) del formato openfootball."""
    out, anio, mes_prev, dia = [], None, 0, None
    for ln in txt.splitlines():
        f = RE_DIA.match(ln)
        if f:
            mes = MESES[f.group(1)]
            if f.group(3):
                anio = int(f.group(3))
            elif anio and mes < mes_prev:
                anio += 1
            mes_prev = mes
            dia = date(anio, mes, int(f.group(2))) if anio else None
            continue
        e = RE_EQ.match(ln)
        sc = RE_90.search(ln) or RE_FT.search(ln)
        if e and sc and dia:
            out.append((dia, nombres.canon(e.group(1), e.group(2)),
                        nombres.canon(e.group(3), e.group(4)), int(sc.group(1)), int(sc.group(2))))
    return out


ETAPAS = {"PLAYOFFS": "Playoffs", "LAST_16": "Octavos", "QUARTER_FINALS": "Cuartos",
          "SEMI_FINALS": "Semifinales", "FINAL": "Final"}


def cl_api(token, y, nombres):
    """Resultados de la temporada en curso y próximos partidos (football-data.org)."""
    txt = bajar(f"https://api.football-data.org/v4/competitions/CL/matches?season={y}",
                {"X-Auth-Token": token})
    try:
        datos = json.loads(txt)["matches"]
    except (ValueError, KeyError):
        print("  ! respuesta inválida de football-data.org (¿token correcto?)")
        return [], []
    res, fut = [], []
    zona = ZoneInfo("Europe/London") if ZoneInfo else timezone.utc
    for m in datos:
        h, a = (m.get("homeTeam") or {}).get("name"), (m.get("awayTeam") or {}).get("name")
        if not h or not a:
            continue
        h, a = nombres.canon(h), nombres.canon(a)
        d = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00")).astimezone(zona)
        sc = m.get("score") or {}
        if m.get("status") == "FINISHED":
            ft = sc.get("fullTime") if sc.get("duration") == "REGULAR" else sc.get("regularTime")
            if ft and ft.get("home") is not None and ft.get("away") is not None:
                res.append((d.date(), h, a, int(ft["home"]), int(ft["away"])))
        elif m.get("status") in ("SCHEDULED", "TIMED"):
            st = m.get("stage", "")
            etapa = f"Fase de liga J{m.get('matchday')}" if st == "LEAGUE_STAGE" else ETAPAS.get(st, st.title())
            fut.append({"fecha": d.date().isoformat(), "hora": d.strftime("%H:%M"),
                        "local": h, "visitante": a, "etapa": etapa})
    return res, fut


def champions(token, nombres, base):
    hoy = date.today()
    y = hoy.year if hoy.month >= 7 else hoy.year - 1
    hist = []
    for k in range(1, TEMPORADAS):
        t = f"{y - k}-{(y - k + 1) % 100:02d}"
        hist += cl_texto(bajar(OF_URL.format(t=t)), nombres)
    res, fut = cl_api(token, y, nombres)
    cl = hist + res
    print(f"  {len(hist)} partidos históricos + {len(res)} de la temporada actual; {len(fut)} por jugar")
    if not cl or not fut:
        return [], None
    todo = base + cl
    modelos = [ajustar(todo, PRIOR, xi) for xi in XIS]
    idx, out = indice(todo), []
    for f in fut:
        n = modelos[1]["n"]
        if n.get(f["local"], 0) < MIN_CL or n.get(f["visitante"], 0) < MIN_CL:
            print(f"  sin historial suficiente: {f['local']} vs {f['visitante']}")
            continue
        pr = completar(modelos, idx, f["local"], f["visitante"], cl=True)
        pr.update(liga="CL", mercado=None, **f)
        out.append(pr)
    return out, {"nombre": "Champions League", "rho": round(modelos[1]["rho"], 2), "historial": len(cl)}


def main():
    partidos, ligas, todos, por_liga = [], {}, [], {}
    nombres = Nombres()
    fx = descargar(f"{BASE}/fixtures.csv")
    print(f"{len(fx)} partidos en fixtures.csv")
    for div, nombre in LIGAS.items():
        print(f"{nombre}…")
        ms = historial(div)
        if len(ms) < 100:
            print("  datos insuficientes, se omite")
            continue
        todos += ms
        por_liga[div] = ms
        nombres.registrar_liga(PAIS[div], {m[1] for m in ms} | {m[2] for m in ms})
        modelos, idx = [ajustar(ms, 0.0, xi) for xi in XIS], indice(ms)
        M = modelos[1]
        ligas[div] = {"nombre": nombre, "rho": round(M["rho"], 2), "historial": len(ms)}
        for r in fx:
            if r.get("Div") != div:
                continue
            h, a, d = r.get("HomeTeam"), r.get("AwayTeam"), fecha(r.get("Date"))
            if not d or M["n"].get(h, 0) < MIN_PARTIDOS or M["n"].get(a, 0) < MIN_PARTIDOS:
                continue
            pr = completar(modelos, idx, h, a)
            pr.update(liga=div, fecha=d.isoformat(), hora=r.get("Time") or "",
                      local=h, visitante=a, mercado=mercado(r))
            partidos.append(pr)
    token = os.environ.get("FOOTBALL_DATA_TOKEN", "").strip()
    if token and todos:
        print("Champions League…")
        cl, info = champions(token, nombres, todos)
        partidos += cl
        if info:
            ligas["CL"] = info
    else:
        print("Champions League omitida: falta el secreto FOOTBALL_DATA_TOKEN")
    print("Backtest…")
    cal = backtest(por_liga) if por_liga else None
    partidos.sort(key=lambda q: (q["fecha"], q["hora"], q["liga"]))
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(
        {"generado": datetime.now(timezone.utc).isoformat(timespec="minutes"),
         "ligas": ligas, "calibracion": cal, "partidos": partidos},
        ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Listo: {len(partidos)} partidos escritos en {SALIDA}")


if __name__ == "__main__":
    main()
