"""Matched whole-year comparisons for daily and monthly reservoir records."""
import numpy as np
from .representative_years import YearSelection
from .strategy_comparison import comparison_selections as pilot_selections


def comparison_selections(frame, annual, seed=42):
    # Preserve the exact pilot design wherever its monthly assumptions hold.
    if annual.complete.all() and annual.rows.eq(12).all():
        try:
            return pilot_selections(frame, annual, seed)
        except ValueError:
            pass
    eligible = annual.index[annual.complete & annual.role.isin(['calibration', 'assessment'])].tolist()
    counts = ['rows', 'storage_pairs', 'release_pairs']
    totals = [len(frame), frame.storage_m3.notna().sum(), frame.release_m3s.notna().sum()]
    # Equal numbers of whole years, with enough rows/pairs even in the shortest
    # possible combination (daily leap-year differences are retained).
    count = next((n for n in range(1, len(eligible))
                  if all(annual.loc[eligible, c].nsmallest(n).sum() >= np.ceil(.65*t)
                         for c, t in zip(counts, totals))), None)
    if count is None:
        raise ValueError('No matched 65% whole-year split with independent assessment')
    chronology = eligible[:count]
    later = eligible[count:]
    # Leave a later training opportunity whenever more than one later year exists.
    maximum_held = min(4, max(1, len(later)-1))
    held = []
    for regime in ('dry', 'normal', 'wet'):
        group = [y for y in later if annual.loc[y, 'regime'] == regime]
        if not group or len(held) >= maximum_held:
            continue
        if regime == 'dry':
            year = min(group, key=lambda y: (annual.loc[y, 'inflow_mean_m3s'], y))
        elif regime == 'wet':
            year = max(group, key=lambda y: (annual.loc[y, 'inflow_mean_m3s'], -y))
        else:
            median = annual.loc[group, 'inflow_mean_m3s'].median()
            year = min(group, key=lambda y: (abs(annual.loc[y, 'inflow_mean_m3s']-median), y))
        held.append(year)
    for regime in ('dry', 'normal', 'wet'):
        remaining = [y for y in later if y not in held and annual.loc[y, 'regime'] == regime]
        if remaining and len(held) < maximum_held:
            held.append(max(remaining, key=lambda y: (annual.loc[y, 'inflow_mean_m3s'], -y)))
    held = sorted(held)
    available = [y for y in eligible if y not in held]
    regimes = ['dry', 'normal', 'wet']
    desired = {r: count*sum(annual.loc[y, 'regime']==r for y in eligible)/len(eligible) for r in regimes}
    quota = {r: min(int(np.floor(desired[r])), sum(annual.loc[y, 'regime']==r for y in available)) for r in regimes}
    while sum(quota.values()) < count:
        choices = [r for r in regimes if quota[r] < sum(annual.loc[y, 'regime']==r for y in available)]
        r = max(choices, key=lambda r: (desired[r]-quota[r], -regimes.index(r)))
        quota[r] += 1
    features = annual.loc[available, ['inflow_mean_m3s', 'inflow_p10_m3s', 'antecedent_inflow_m3s']].copy()
    features['time'] = np.array(available, dtype=float)
    features = (features-features.min()).divide((features.max()-features.min()).replace(0, 1))
    rng = np.random.default_rng(seed)
    mixed = []
    for r in regimes:
        if quota[r] == 0:
            continue
        group = [y for y in available if annual.loc[y, 'regime']==r]
        if r == 'dry': first = min(group, key=lambda y: (annual.loc[y, 'inflow_mean_m3s'], y))
        elif r == 'wet': first = max(group, key=lambda y: (annual.loc[y, 'inflow_mean_m3s'], -y))
        else: first = min(group, key=lambda y: (abs(annual.loc[y, 'inflow_mean_m3s']-annual.loc[available, 'inflow_mean_m3s'].median()), y))
        chosen = [first]
        while len(chosen) < quota[r]:
            remaining = sorted(set(group)-set(chosen))
            scores = [min(float(np.linalg.norm(features.loc[y]-features.loc[z])) for z in chosen) for y in remaining]
            candidates = np.flatnonzero(np.isclose(scores, max(scores), rtol=0, atol=1e-12))
            chosen.append(remaining[int(rng.choice(candidates))])
        mixed.extend(chosen)
    warm = set(annual.index[annual.role=='warmup'])
    plans = {}
    for arm, years in [('chronological_pooled', chronology), ('mixed_pooled', mixed), ('mixed_balanced', mixed)]:
        roles = tuple('warmup' if d.year in warm else 'calibration' if d.year in years else 'assessment' if d.year in held else 'context' for d in frame.index)
        plans[arm] = YearSelection(tuple(d.date().isoformat() for d in frame.index), roles, tuple(int(y) for y in frame.index.year), tuple(str(annual.loc[d.year, 'regime']) for d in frame.index), .65, seed, 1, 1)
        assert not set(years)&set(held) and len(set(years)) == count
    return plans, held
