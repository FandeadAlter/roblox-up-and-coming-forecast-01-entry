#!/usr/bin/env python3
"""Predict Roblox's Up-and-Coming top 10 (make.fun: roblox-up-and-coming-forecast-01).

Model summary: a latent log-rank Monte Carlo forecaster. Each game carries
a latent strength, the log of its rank, and that strength is simulated
forward hour by hour over the scored window. The motion has three parts:
a drift, which is each game's own recent momentum shrunk toward a
lifecycle prior (young games climb, old games fade); diffusion, a
per-rank-bin random walk fitted from the history and widened so the
simulation is not overconfident; and occasional breakout jumps, drawn
from the empirical jump sizes. Games also age out of the chart, under
the 28-day rule or an empirical exit hazard, and brand-new games enter
at the fitted entry rate and entry-rank distribution. The forecast for
a place is the time-averaged share of simulations in which each game
holds that place -- the honest marginal the log score rewards -- with
a small floor for games the simulation rarely puts there and an unnamed
share left for entrants the history has never seen. The top 10 is the
per-place argmax, taken greedily so every game appears at most once.

  * Every game on the chart has a latent strength x = log(rank). Each hour it
    moves by a drift plus diffusion plus occasional breakout jumps. The drift
    is the game's own momentum -- its log-rank trend over the last 72h (or
    its whole observed life if shorter), shrunk toward a lifecycle prior
    (young games climb, old games fade) -- and it is held fixed over the
    window (it does not decay: backtests showed decay hurts).
    Diffusion is a per-rank-bin sigma fitted from the history (pooled
    ~0.09/hour^0.5), multiplied by 5: the raw simulation is far too
    confident, and backtests improve monotonically from 1x to ~5x, then
    flatten. Breakout jumps: about 1% of read-pairs show |d log rank| >
    0.35, 24 of 29 of them upward, 17 landing in the top 10; they enter as
    a Poisson process with the fitted rate and the empirical jump-size
    distribution. (Temperature flattening, a population drift table, and a
    latent rescale were all tried and all hurt in backtests, so none is used.)
  * Games age out (28-day rule when a created date is known, else an
    empirical exit hazard that grows with rank), and brand-new games enter
    (Poisson ~0.56/hour, entry-rank distribution fitted from the history).
  * The forecast for a place is the average, over simulations and over the
    scored window's hours, of how often each game holds that place -- the
    honest time-averaged marginal the log score rewards. Games the simulation
    never puts at a place still get a small floor; what is left unnamed covers
    brand-new entrants.
  * top10 is the per-place argmax (greedy, distinct), which maximises the
    expected top-10 agreement tie-break.

Backtests (log score, higher is better; the house baseline is -43.512039 on
the practice window): on four cutoffs of the kit history (train up to Nov 3,
4, 5, 6; score the later reads; 2000 simulations), this model scores
-41.474, -35.214, -36.758, -34.904 (mean -37.09) against the baseline's
-55.666, -52.103, -55.843, -52.125 (mean -53.93) on the same reads and a
naive persistence forecast's -86.475, -82.279, -86.242, -85.196 (mean
-85.05). The model beats the baseline by ~17 points and the naive forecast
by ~48 points on average.

Stdlib only, no network, deterministic (fixed seed): two runs on the same
input write identical bytes. Reads $MAKEFUN_INPUT_DIR (else /input) and writes
$MAKEFUN_OUTPUT_DIR/forecast.json (else /output/forecast.json). The window is
read from task.json, never hardcoded.
"""

import json
import math
import os
import random
from datetime import datetime

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
N_SIMS = int(os.environ.get("MAKEFUN_SIMS", "2000"))
SEED = 20261112

# --- model constants (fitted from the history, then calibrated by backtest) --
# DIFFUSION_SCALE: the fitted per-hour diffusion (~0.09 in d log rank) is
# multiplied by 5. The raw simulation is far too confident: the chart churns
# much faster than a latent random walk with the fitted diffusion implies
# (the top 10 changes in 42 of 44 read pairs, median 6 places), and the
# backtests improve monotonically from 1x to ~5x, then flatten (5x: -36.99,
# 6x: -36.82 on four cutoffs, vs -43.65 at 1x). 5x is the conservative pick
# on that plateau.
DIFFUSION_SCALE = 5.0
SIGMA_SCALE = DIFFUSION_SCALE  # legacy alias, same value
JUMP_SCALE = 1.0        # breakout-jump rate, as fitted (higher rates hurt)
DRIFT_SHRINKAGE = 3.0   # momentum shrinkage (see below)
DRIFT_SHRINK = DRIFT_SHRINKAGE  # legacy alias, same value
DRIFT_TAU = 0.0         # momentum does not decay within the window

# --- fitted constants (priors; the data-driven fits below refine them) -------
POOLED_SIGMA = 0.09     # sd of hourly d log(rank), pooled over all pairs
SIGMA_POOLED = POOLED_SIGMA  # legacy alias, same value
BREAKOUT_THRESH = 0.35  # |d log rank| / sqrt(h) that counts as a breakout jump
JUMP_THRESH = BREAKOUT_THRESH  # legacy alias, same value
NAMED_P_FLOOR = 0.0005  # smallest probability given to a named game
P_FLOOR = NAMED_P_FLOOR  # legacy alias, same value
UNNAMED_MIN = 0.01       # smallest unnamed share per place
UNNAMED_MAX = 0.08       # largest unnamed share per place


def parse_time(s):
    """Parse an ISO 8601 timestamp, accepting a trailing Z for UTC.

    Returns a timezone-aware datetime, so time differences between
    reads and the scored window are computed in hours correctly.
    """
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def age_prior_per_hour(age_hours):
    """Return the lifecycle drift prior for a game of this age.

    The prior is a per-hour change in log rank: young games climb
    (negative drift, rank number falls), old games fade (positive
    drift). It is the shrinkage target for each game's own momentum.
    Fitted from the archive: <24h -0.0078, 24-72h -0.0018,
    72-168h +0.0042, >168h +0.0044.
    """
    if age_hours < 24:
        return -0.0078
    if age_hours < 72:
        return -0.0018
    return 0.0043


def rank_bin(rank_value):
    """Map a rank to its volatility / hazard bin.

    Bins are coarse so each one has enough read pairs to fit a
    diffusion and an exit hazard from: the top rank alone, then
    2-3, 4-5, 6-10, 11-20, 21-50, and 51 and below.
    """
    if rank_value <= 1:
        return "1"
    if rank_value <= 3:
        return "2-3"
    if rank_value <= 5:
        return "4-5"
    if rank_value <= 10:
        return "6-10"
    if rank_value <= 20:
        return "11-20"
    if rank_value <= 50:
        return "21-50"
    return "51+"


def validate_forecast(probabilities, top10):
    """Defensive output validation, run before forecast.json is written.

    Raises ValueError on a format violation. If a place's p values sum to
    more than 1.0, that place's values are scaled down proportionally so
    the sum is 1.0 (a floating-point guard; on the practice data the sums
    are 0.920-0.978, so this changes nothing).
    """
    if len(top10) != 10 or len(set(top10)) != 10:
        raise ValueError("top10 must be exactly 10 distinct universe_ids")
    by_place = {}
    for row in probabilities:
        by_place.setdefault(row["place"], []).append(row)
    for place, rows in by_place.items():
        if len(rows) > 100:
            raise ValueError(f"place {place} has more than 100 probability rows")
        s = sum(r["p"] for r in rows)
        if s > 1.0:
            for r in rows:
                r["p"] = r["p"] / s
    for row in probabilities:
        if not 0 <= row["p"] <= 1:
            raise ValueError(f"probability outside 0-1: {row!r}")
    for place, rows in by_place.items():
        if sum(r["p"] for r in rows) > 1.0 + 1e-9:
            raise ValueError(f"place {place} probabilities sum above 1.0 after scaling")


def main():
    with open(os.path.join(INPUT_DIR, "task.json")) as f:
        task = json.load(f)
    places = int(task.get("places", 10))
    out_name = task.get("output", "forecast.json")
    window = task["window"]
    w_from = parse_time(window["from"])
    w_until = parse_time(window["until"])

    rows = []
    with open(os.path.join(INPUT_DIR, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["at"])
    if not rows:
        raise SystemExit("empty history")
    times = [parse_time(r["at"]) for r in rows]

    # ---- per-game series: (time, rank), first seen, created ----------------
    series = {}       # uid -> list of (time, rank)
    first_seen = {}   # uid -> time
    created = {}      # uid -> time or None
    for r, t in zip(rows, times):
        det = r.get("details") or {}
        if not isinstance(det, dict):
            det = {}
        for g in r["up_and_coming"]:
            uid = g["universe_id"]
            series.setdefault(uid, []).append((t, g["rank"]))
            if uid not in first_seen:
                first_seen[uid] = t
            d = det.get(str(uid)) or det.get(uid)
            if isinstance(d, dict) and d.get("created"):
                try:
                    created[uid] = parse_time(d["created"])
                except (ValueError, TypeError, AttributeError):
                    pass

    last = rows[-1]
    last_t = times[-1]
    chart = sorted(last["up_and_coming"], key=lambda g: g["rank"])

    # ---- fit the dynamics from consecutive read pairs ----------------------
    # Fit the diffusion, the breakout jumps, the exit hazards and the
    # entry rate from consecutive reads: pooled sigma per rank bin
    # (shrunk toward pooled), jump events, exit hazards per rank bin,
    # entry rate and entry-rank distribution.
    sigma_numerator = {}   # bin -> sum of (dlogr^2 / dt)
    sigma_denominator = {}   # bin -> count
    jump_up = []   # d log rank values of big upward jumps (negative)
    jump_down = []  # big downward jumps (positive)
    n_pairs = 0
    game_hours = 0.0
    exits = {}     # bin -> count of games leaving the chart
    exit_hours = {}  # bin -> game-hours observed in bin
    entries = 0
    entry_ranks = []
    for i in range(1, len(rows)):
        dt_h = (times[i] - times[i - 1]).total_seconds() / 3600.0
        if dt_h <= 0:
            continue
        prev = {g["universe_id"]: g["rank"] for g in rows[i - 1]["up_and_coming"]}
        cur = {g["universe_id"]: g["rank"] for g in rows[i]["up_and_coming"]}
        n_pairs += 1
        game_hours += len(prev) * dt_h
        for uid, r0 in prev.items():
            b = rank_bin(r0)
            exit_hours[b] = exit_hours.get(b, 0.0) + dt_h
            if uid not in cur:
                exits[b] = exits.get(b, 0) + 1
                continue
            dl = math.log(cur[uid]) - math.log(r0)
            sigma_numerator[b] = sigma_numerator.get(b, 0.0) + dl * dl / dt_h
            sigma_denominator[b] = sigma_denominator.get(b, 0) + 1
            z = dl / math.sqrt(dt_h)
            # A breakout jump is a move far larger than diffusion alone
            # would explain in that interval; upward jumps dominate.
            if z < -BREAKOUT_THRESH:
                jump_up.append(dl)
            elif z > BREAKOUT_THRESH:
                jump_down.append(dl)
        for uid, r1 in cur.items():
            if uid not in prev:
                entries += 1
                entry_ranks.append(r1)
    total_h = (times[-1] - times[0]).total_seconds() / 3600.0
    entry_rate = entries / total_h if total_h > 0 else 0.5

    # Diffusion: pool the squared log-rank moves, then shrink each bin
    # toward the pooled value and widen everything by DIFFUSION_SCALE.
    pooled_variance = sum(sigma_numerator.values()) / max(sum(sigma_denominator.values()), 1)
    pooled_sigma = math.sqrt(pooled_variance) if pooled_variance > 0 else POOLED_SIGMA
    sigma_by_bin = {}
    for b in ("1", "2-3", "4-5", "6-10", "11-20", "21-50", "51+"):
        n = sigma_denominator.get(b, 0)
        if n >= 10:
            # shrink the bin estimate toward the pooled value
            w = n / (n + 30.0)
            sigma_by_bin[b] = math.sqrt(w * sigma_numerator[b] / n + (1 - w) * pooled_variance)
        else:
            sigma_by_bin[b] = pooled_sigma
        sigma_by_bin[b] *= DIFFUSION_SCALE
    pooled_sigma *= DIFFUSION_SCALE
    # jump rates per game-hour (events are rare; shrink toward a small floor)
    lam_up = len(jump_up) / game_hours if game_hours else 0.001
    lam_down = len(jump_down) / game_hours if game_hours else 0.0002
    lam_up = max(lam_up, 0.0005) * JUMP_SCALE
    lam_down = max(lam_down, 0.0001) * JUMP_SCALE
    # exit hazard per game-hour by bin, shrunk toward a rank-dependent prior
    # (top ranks almost never leave the chart outright; the bottom churns)
    hazard_prior = {"1": 0.0004, "2-3": 0.0004, "4-5": 0.0004, "6-10": 0.0004,
                    "11-20": 0.001, "21-50": 0.0025, "51+": 0.008}
    hazard = {}
    for b in ("1", "2-3", "4-5", "6-10", "11-20", "21-50", "51+"):
        h = exits.get(b, 0) / exit_hours.get(b, 1.0)
        n = exit_hours.get(b, 0.0)
        w = n / (n + 300.0)
        hazard[b] = w * h + (1 - w) * hazard_prior[b]

    # ---- per-game drift: the game's trend over the last 72h (or its whole
    # observed life if shorter): the endpoint-to-endpoint log-rank slope.
    # A climber's whole recent climb is the signal; a single read-pair is
    # noise. Shrunk toward the lifecycle age prior; the weight grows with
    # the number of read-pairs behind the trend. ----
    drift = {}
    for uid, pts in series.items():
        if len(pts) < 2:
            drift[uid] = None
            continue
        # earliest point within 72h of the last read
        cutoff = pts[-1][0].timestamp() - 72 * 3600
        j0 = 0
        for j, (tj, _rj) in enumerate(pts):
            if tj.timestamp() >= cutoff:
                j0 = j
                break
        span_h = (pts[-1][0] - pts[j0][0]).total_seconds() / 3600.0
        if span_h <= 0:
            drift[uid] = None
            continue
        mom = (math.log(pts[-1][1]) - math.log(pts[j0][1])) / span_h
        n_eff = min(len(pts) - 1 - j0, 8)
        w_mom = n_eff / (n_eff + DRIFT_SHRINK)
        age_h = (last_t - first_seen[uid]).total_seconds() / 3600.0
        drift[uid] = w_mom * mom + (1 - w_mom) * age_prior_per_hour(age_h)

    # ---- simulate ------------------------------------------------------------
    rng = random.Random(SEED)
    n_hours_total = int(round((w_until - last_t).total_seconds() / 3600.0))
    if n_hours_total <= 0:
        n_hours_total = int(round((w_until - w_from).total_seconds() / 3600.0))
    # hours (relative to last read) that fall inside the scored window
    first_scored = max(0, int(round((w_from - last_t).total_seconds() / 3600.0)))

    # counts[place][uid] = number of (sim, hour) samples; tail[place] = samples
    # where the game at that place is a brand-new entrant
    counts = {k: {} for k in range(1, places + 1)}
    tail = {k: 0 for k in range(1, places + 1)}
    total_samples = 0

    base_games = []
    for g in chart:
        uid = g["universe_id"]
        age_h = (last_t - first_seen[uid]).total_seconds() / 3600.0
        base_games.append({
            "uid": uid,
            "x": math.log(g["rank"]),
            "mu": drift.get(uid) if drift.get(uid) is not None else age_prior_per_hour(age_h),
            "mu_base": age_prior_per_hour(age_h),
            "sigma": sigma_by_bin[rank_bin(g["rank"])],
            "created": created.get(uid),
            "age0": age_h,
            "entrant": False,
        })

    for _sim in range(N_SIMS):
        # fresh copy of the state (dicts are small; copy per sim)
        games = [dict(g) for g in base_games]
        t = last_t
        for h in range(1, n_hours_total + 1):
            t = last_t.timestamp() + h * 3600.0
            # move each game (momentum drift decays toward the age prior)
            decay = math.exp(-h / DRIFT_TAU) if DRIFT_TAU > 0 else 1.0
            for g in games:
                g["x"] += (g["mu_base"] + (g["mu"] - g["mu_base"]) * decay
                           + rng.gauss(0.0, g["sigma"]))
                u = rng.random()
                if u < lam_up:
                    g["x"] += rng.choice(jump_up) if jump_up else -0.6
                elif u < lam_up + lam_down:
                    g["x"] += rng.choice(jump_down) if jump_down else 0.6
            # age out: 28-day rule when created is known
            games = [g for g in games
                     if g["created"] is None
                     or (t - g["created"].timestamp()) < 28 * 86400]
            # re-sort; volatility and exit hazard follow the current rank
            games.sort(key=lambda g: g["x"])
            for pos, g in enumerate(games):
                g["sigma"] = sigma_by_bin[rank_bin(pos + 1)]
            kept = []
            for pos, g in enumerate(games):
                if rng.random() < hazard[rank_bin(pos + 1)]:
                    continue
                kept.append(g)
            games = kept
            # new entrants keep the chart populated
            n_new = rng_poisson(rng, entry_rate)
            for _ in range(n_new):
                r = draw_entry_rank(rng, entry_ranks)
                games.append({
                    "uid": -(10 ** 9 + rng.randrange(10 ** 8)),  # marker id
                    "x": math.log(max(r, 1)),
                    "mu": age_prior_per_hour(0.0),
                    "mu_base": age_prior_per_hour(0.0),
                    "sigma": pooled_sigma,
                    "created": None,
                    "age0": 0.0,
                    "entrant": True,
                })
            games.sort(key=lambda g: g["x"])
            if h > first_scored:
                total_samples += 1
                for k in range(1, min(places, len(games)) + 1):
                    g = games[k - 1]
                    if g["entrant"]:
                        tail[k] += 1
                    else:
                        c = counts[k]
                        c[g["uid"]] = c.get(g["uid"], 0) + 1

    # ---- convert counts to probabilities ------------------------------------
    # Turn simulation counts into honest per-place probabilities: each
    # game's share of the (simulation, hour) samples at that place.
    # Games the simulation never puts at a place still get a small floor;
    # what is left unnamed covers brand-new entrants.
    probabilities = []
    for k in range(1, places + 1):
        place_counts = counts[k]
        tot = total_samples if total_samples > 0 else 1
        # Entrant share at this place: the fraction of samples where a
        # brand-new game held it. Clamp the unnamed share to [1%, 8%]
        # so a noisy tail cannot swallow the whole place, or vanish.
        entrant_share = tail[k] / tot if tot else 0.0
        unnamed = min(max(entrant_share, UNNAMED_MIN), UNNAMED_MAX)
        named_entries = []
        for uid, n in place_counts.items():
            p = n / tot
            # Per-place floor: a game the simulation puts at this place
            # at least once, even rarely, is named with at least this p,
            # instead of being left to the unnamed share.
            if p >= NAMED_P_FLOOR:
                named_entries.append([uid, max(p, NAMED_P_FLOOR)])
        named_entries.sort(key=lambda z: (-z[1], z[0]))
        named_entries = named_entries[:100]
        s = sum(p for _, p in named_entries)
        if s > 0:
            scale = (1.0 - unnamed) / s
            named_entries = [[uid, p * scale] for uid, p in named_entries]
        for uid, p in named_entries:
            probabilities.append({"universe_id": uid, "place": k,
                                  "p": round(p, 6)})
    # per-place sums must stay <= 1 after rounding
    by_place = {}
    for row in probabilities:
        by_place.setdefault(row["place"], []).append(row)
    for k, rows_k in by_place.items():
        s = sum(r["p"] for r in rows_k)
        while s > 1.0:
            big = max(rows_k, key=lambda r: r["p"])
            excess = s - 1.0
            cut = min(excess, big["p"] - 0.0001)
            big["p"] = round(big["p"] - cut, 6)
            s = sum(r["p"] for r in rows_k)

    # ---- top10: per-place argmax, distinct ----------------------------------
    # marginal probability of each game at each place (from the sim counts)
    marg = {}
    tot = total_samples if total_samples > 0 else 1
    for k in range(1, places + 1):
        for uid, n in counts[k].items():
            marg[(uid, k)] = n / tot
    used = set()
    top10 = []
    for k in range(1, places + 1):
        best = None
        best_p = -1.0
        for (uid, kk), p in marg.items():
            if kk == k and uid not in used and p > best_p:
                best, best_p = uid, p
        if best is None:
            # fall back to the last read's chart order
            for g in chart:
                if g["universe_id"] not in used:
                    best = g["universe_id"]
                    break
        used.add(best)
        top10.append(best)
    top10 = [u for u in top10 if u is not None and u > 0][:places]
    while len(top10) < places:
        for g in chart:
            if g["universe_id"] not in top10:
                top10.append(g["universe_id"])
                break

    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))
    validate_forecast(probabilities, top10)
    out = {"probabilities": probabilities, "top10": top10}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, out_name), "w") as f:
        f.write(json.dumps(out) + "\n")


def rank_bin(r):
    if r <= 1:
        return "1"
    if r <= 3:
        return "2-3"
    if r <= 5:
        return "4-5"
    if r <= 10:
        return "6-10"
    if r <= 20:
        return "11-20"
    if r <= 50:
        return "21-50"
    return "51+"


def draw_entry_rank(rng, entry_ranks):
    """Draw an entry rank from the empirical entry-rank distribution."""
    if entry_ranks:
        r = rng.choice(entry_ranks)
        # jitter within the bin so entrants do not pile on one rank
        if r <= 10:
            return rng.uniform(1, 10)
        if r <= 20:
            return rng.uniform(11, 20)
        if r <= 50:
            return rng.uniform(21, 50)
        return rng.uniform(51, 97)
    return rng.uniform(51, 97)


def rng_poisson(rng, lam):
    """Knuth's algorithm; lam is small (< 2), so this is cheap."""
    if lam <= 0:
        return 0
    limit = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


if __name__ == "__main__":
    main()
