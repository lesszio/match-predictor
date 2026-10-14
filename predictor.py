#!/usr/bin/env python3
"""Genera docs/predicciones.json: Dixon-Coles para las 5 grandes ligas.
Solo usa la librería estándar. Fuente: football-data.co.uk"""
import csv, difflib, io, json, math, os, re, unicodedata, urllib.request
from datetime import date, datetime, timezone
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
    return out


def pois(k, l):
    return math.exp(-l) * l ** k / math.factorial(k)


def tau(x, y, l, m, r):
    if x == 0 and y == 0: return 1 - l * m * r
    if x == 0 and y == 1: return 1 + l * r
    if x == 1 and y == 0: return 1 + m * r
    if x == 1 and y == 1: return 1 - r
    return 1.0


def ajustar(ms, prior=0.0):
    """Ataque/defensa/localía por punto fijo ponderado + rho de Dixon-Coles."""
    hoy = max(m[0] for m in ms)
    w = [math.exp(-XI * (hoy - m[0]).days) for m in ms]
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
            p = pois(x, l) * pois(y, m) * max(tau(x, y, l, m, M["rho"]), 0.0)
            P[(x, y)] = p; s += p
    w = d = v = o = b = 0.0
    for (x, y), p in P.items():
        p /= s
        if x > y: w += p
        elif x == y: d += p
        else: v += p
        if x + y > 2: o += p
        if x > 0 and y > 0: b += p
    top = sorted(P.items(), key=lambda kv: -kv[1])[:5]
    return {"xg_l": round(l, 2), "xg_v": round(m, 2),
            "p1": round(w, 4), "px": round(d, 4), "p2": round(v, 4),
            "o25": round(o, 4), "btts": round(b, 4),
            "top": [[f"{x}-{y}", round(p / s, 4)] for (x, y), p in top]}


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
    M = ajustar(base + cl, prior=PRIOR)
    out = []
    for f in fut:
        if M["n"].get(f["local"], 0) < MIN_CL or M["n"].get(f["visitante"], 0) < MIN_CL:
            print(f"  sin historial suficiente: {f['local']} vs {f['visitante']}")
            continue
        p = predecir(M, f["local"], f["visitante"])
        p.update(liga="CL", mercado=None, **f)
        out.append(p)
    return out, {"nombre": "Champions League", "rho": round(M["rho"], 2), "historial": len(cl)}


def main():
    partidos, ligas, todos = [], {}, []
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
        nombres.registrar_liga(PAIS[div], {m[1] for m in ms} | {m[2] for m in ms})
        M = ajustar(ms)
        ligas[div] = {"nombre": nombre, "rho": round(M["rho"], 2), "historial": len(ms)}
        for r in fx:
            if r.get("Div") != div:
                continue
            h, a, d = r.get("HomeTeam"), r.get("AwayTeam"), fecha(r.get("Date"))
            if not d or M["n"].get(h, 0) < MIN_PARTIDOS or M["n"].get(a, 0) < MIN_PARTIDOS:
                continue
            p = predecir(M, h, a)
            p.update(liga=div, fecha=d.isoformat(), hora=r.get("Time") or "",
                     local=h, visitante=a, mercado=mercado(r))
            partidos.append(p)
    token = os.environ.get("FOOTBALL_DATA_TOKEN", "").strip()
    if token and todos:
        print("Champions League…")
        cl, info = champions(token, nombres, todos)
        partidos += cl
        if info:
            ligas["CL"] = info
    else:
        print("Champions League omitida: falta el secreto FOOTBALL_DATA_TOKEN")
    partidos.sort(key=lambda p: (p["fecha"], p["hora"], p["liga"]))
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(
        {"generado": datetime.now(timezone.utc).isoformat(timespec="minutes"),
         "ligas": ligas, "partidos": partidos},
        ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Listo: {len(partidos)} partidos escritos en {SALIDA}")


if __name__ == "__main__":
    main()
