"""Calibration routes for event-level miss-rate control.

STRUCTURAL FACT the routes must survive: the collar-based event miss rate is
NOT monotone in the threshold in two ways:
  (local)  the max(0.2s, 0.5*dur) offset collar lets a lowered threshold
           lengthen a prediction out of its match;
  (global) as lam -> 0 detections merge into clip-long blobs whose onsets
           miss the 0.2 s onset collar => risk climbs back to ~1.
So the risk-vs-lam curve is U-shaped-ish with an interior minimum: there is NO
a-priori safe end of the grid to anchor monotonization or fixed-sequence
testing. Valid routes therefore either (a) pay a Bonferroni price over a
candidate set, or (b) split the calibration data: one half chooses the
anchor/ordering, the other half tests (order independent of test data).

Loss encodings (target: pooled event miss share <= alpha, the task gate):
  pooled:   per-clip excess X_c(lam) = missed_c - alpha*M_c, H0: E[X]>0.
            Hoeffding p-value with per-clip ranges (widths M_c) - conservative.
  clipmean: L_c = missed_c/M_c in [0,1] on clips with M_c>0, H0: E[L]>alpha.
            Hoeffding-Bentkus p-value - tight, but controls the per-clip mean,
            not the pooled share (gap reported empirically via the gate).
"""
import numpy as np
from scipy.stats import binom


# ---------- p-values ----------

def p_pooled_hoeffding(miss_g, nref, alpha):
    X = miss_g - alpha * nref
    s = X.sum()
    if s >= 0:
        return 1.0
    denom = (nref.astype(np.float64) ** 2).sum()
    if denom == 0:
        return 1.0
    return float(np.exp(-2.0 * s * s / denom))


def p_clipmean_hb(miss_g, nref, alpha):
    m = nref > 0
    n = int(m.sum())
    if n == 0:
        return 1.0
    r = float(np.mean(miss_g[m] / nref[m]))
    if r >= alpha:
        return 1.0
    p_h = float(np.exp(-2.0 * n * (alpha - r) ** 2))
    p_b = float(np.e * binom.cdf(int(np.ceil(n * r)), n, alpha))
    return min(p_h, p_b, 1.0)


P_VALUES = {"pooled": p_pooled_hoeffding, "clipmean": p_clipmean_hb}


def pooled_risk_curve(miss_sub, nref_sub):
    tot = max(int(nref_sub.sum()), 1)
    return miss_sub.sum(0) / tot  # [G]


# ---------- routes; each returns a grid index ----------

def crc_naive(miss_sub, nref_sub, alpha, **kw):
    """CRC as if the loss were monotone: rightmost g with
    (sum X + B)/(n+1) <= 0, B=(1-alpha)*max M_c. Deliberately ignores both
    non-monotonicities (E1 baseline)."""
    B = (1.0 - alpha) * max(nref_sub.max(), 1)
    tot = miss_sub.sum(0) - alpha * nref_sub.sum()
    ok = (tot + B) <= 0.0
    idx = np.where(ok)[0]
    return int(idx[-1]) if len(idx) else int(np.argmin(tot))


def _split(rng, n):
    perm = rng.permutation(n)
    h = n // 2
    return perm[:h], perm[h:]


def crc_split_mono(miss_sub, nref_sub, alpha, rng=None, **kw):
    """Split: half A picks anchor g0 = argmin pooled risk; half B monotonizes
    the excess curve rightward from g0 (running max) and applies the CRC
    condition; select the rightmost satisfying g >= g0, else g0."""
    A, Bh = _split(rng, len(nref_sub))
    g0 = int(np.argmin(pooled_risk_curve(miss_sub[A], nref_sub[A])))
    mB, nB = miss_sub[Bh], nref_sub[Bh]
    Bbound = (1.0 - alpha) * max(nB.max(), 1)
    tot = mB.sum(0) - alpha * nB.sum()
    env = tot.copy()
    env[g0:] = np.maximum.accumulate(tot[g0:])
    ok = np.zeros(len(tot), bool)
    ok[g0:] = (env[g0:] + Bbound) <= 0.0
    idx = np.where(ok)[0]
    return int(idx[-1]) if len(idx) else g0


def ltt_split(miss_sub, nref_sub, alpha, rng=None, delta=0.05, loss="clipmean",
              n_candidates=200, **kw):
    """Split fixed-sequence LTT: half A orders a candidate subgrid by its
    empirical risk (safest first, ties -> larger lam); half B walks the order
    testing at level delta, stops at first non-rejection; among accepted
    candidates returns the largest lam. Fallback: A's argmin-risk candidate."""
    G = miss_sub.shape[1]
    cand = np.unique(np.linspace(0, G - 1, n_candidates).astype(int))
    A, Bh = _split(rng, len(nref_sub))
    riskA = pooled_risk_curve(miss_sub[A], nref_sub[A])[cand]
    order = cand[np.lexsort((-cand, riskA))]  # risk asc, then lam desc
    pfun = P_VALUES[loss]
    mB, nB = miss_sub[Bh], nref_sub[Bh]
    accepted = []
    for g in order:
        if pfun(mB[:, g], nB, alpha) <= delta:
            accepted.append(g)
        else:
            break
    if accepted:
        return int(max(accepted))
    return int(cand[np.argmin(riskA)])


def ltt_split_pooledcap(miss_sub, nref_sub, alpha, rng=None, delta=0.05,
                        n_candidates=200, delta_tail_frac=0.2, **kw):
    """Pooled-cap route: proven components certifying a capped pooled
    surrogate via a probabilistic cap. The full pooled excess X_c = m_c - alpha*M_c is
    unbounded above (no a-priori cap on events per clip exists in-corpus:
    validation max 15 < test max 31), and without any bound on M_c no
    distribution-free finite-sample test of E[X] <= 0 has power (mass can
    hide in the tail). This route therefore certifies the maximal honest
    statement, splitting delta = delta_tail + delta_test:

      half A  chooses the cap K = max_c M_c over A and the candidate
              ordering by empirical pooled risk (exactly as ltt_split);
      half B, component 1 (level delta_tail = delta_tail_frac * delta):
              Clopper-Pearson upper bound q_up on P(M_c > K);
      half B, component 2 (level delta_test = delta - delta_tail):
              fixed-sequence walk over A's order testing, per candidate,
              H0: E[m_c 1{M_c<=K}] > alpha * E[M_c 1{M_c<=K}] via a
              one-sided Hoeffding p-value on X_c restricted to B-clips
              with M_c <= K -- bounded in [-alpha*K, (1-alpha)*K], width
              K, deterministic given A's choice.

    Joint guarantee (prob >= 1-delta over the calibration draw): the pooled
    miss share among clips with at most K events is <= alpha, AND clips
    with more than K events occur with frequency <= q_up. The event mass
    carried by tail clips is reported, not certified. Fallback: A's
    argmin-risk candidate, uncertified."""
    G = miss_sub.shape[1]
    cand = np.unique(np.linspace(0, G - 1, n_candidates).astype(int))
    A, Bh = _split(rng, len(nref_sub))
    riskA = pooled_risk_curve(miss_sub[A], nref_sub[A])[cand]
    order = cand[np.lexsort((-cand, riskA))]  # risk asc, then lam desc
    K = max(int(nref_sub[A].max()), 1)
    delta_test = delta * (1.0 - delta_tail_frac)
    mB, nB = miss_sub[Bh], nref_sub[Bh]
    inK = nB <= K
    mK, nK_ref = mB[inK], nB[inK]
    n_K = int(inK.sum())
    accepted = []
    if n_K > 0:
        for g in order:
            X = mK[:, g] - alpha * nK_ref
            xbar = float(X.mean())
            p = np.exp(-2.0 * n_K * xbar * xbar / (K * K)) if xbar < 0 else 1.0
            if p <= delta_test:
                accepted.append(g)
            else:
                break
    if accepted:
        return int(max(accepted))
    return int(cand[np.argmin(riskA)])


def pooledcap_tail_bound(nref_B, K, delta_tail):
    """Clopper-Pearson upper confidence bound on P(M_c > K) at level
    delta_tail, from the B-half counts (the lambda-independent component of
    the pooled-cap certificate)."""
    from scipy.stats import beta
    n = len(nref_B)
    k = int((nref_B > K).sum())
    if k >= n:
        return 1.0
    return float(beta.ppf(1.0 - delta_tail, k + 1, n - k))


def ltt_bonferroni(miss_sub, nref_sub, alpha, delta=0.05, loss="clipmean",
                   n_candidates=50, **kw):
    """Bonferroni LTT over a coarse candidate set: test every candidate at
    delta/n_candidates on the FULL calibration set; among passing candidates
    return the largest lam. Fallback: pooled-risk argmin (no guarantee)."""
    G = miss_sub.shape[1]
    cand = np.unique(np.linspace(0, G - 1, n_candidates).astype(int))
    pfun = P_VALUES[loss]
    thr = delta / len(cand)
    passing = [g for g in cand if pfun(miss_sub[:, g], nref_sub, alpha) <= thr]
    if passing:
        return int(max(passing))
    risk = pooled_risk_curve(miss_sub, nref_sub)[cand]
    return int(cand[np.argmin(risk)])


def rcps_fixedseq(miss_sub, nref_sub, alpha, delta=0.05, loss="clipmean", **kw):
    """Canonical RCPS-style fixed-sequence calibration for a loss that is
    NON-DECREASING in lam (valid ONLY for fixed-extent box decodings, cf.
    Remark 1): walk lam ascending from 0, test H0: R(lam)>alpha at level
    delta at each step, stop at the first non-rejection, deploy the last
    accepted lam. The ordering is data-independent BECAUSE of monotonicity,
    so the full calibration set is used with no split and no Bonferroni."""
    pfun = P_VALUES[loss]
    G = miss_sub.shape[1]
    last_ok = None
    for g in range(G):
        if pfun(miss_sub[:, g], nref_sub, alpha) <= delta:
            last_ok = g
        else:
            break
    return int(last_ok) if last_ok is not None else 0


def crc_c_nonmono(miss_sub, nref_sub, alpha, rng=None, n_boot=200, **kw):
    """CRC-C of arXiv:2602.20151 (Angelopoulos, non-monotonic losses), adapted
    to our geometry: clip-balanced loss L_i(lam) in [0,1]; the base selector
    A(D) picks the RIGHTMOST grid lam with empirical risk <= level (our
    efficiency direction mirrors the paper's inf over its conservatism-ordered
    grid). The algorithmic-stability deficit beta is estimated by the paper's
    bootstrap (Eq. 58): B replicates of n+1 clips drawn with replacement; per
    replicate, mean over i of [loss of the leave-one-out refit at Z_i minus
    loss of the full refit at Z_i]; beta_hat = positive part of the average.
    Final selection at the deflated level alpha - beta_hat. Guarantee: in
    expectation, E[R(lam_hat)] <= alpha, conditional on beta_hat >= true beta
    (the bootstrap estimate has no finite-sample proof; paper's own caveat).
    NOTE: the paper's rigorous finite-grid result (Prop. 2) assumes a zero-loss
    safe parameter, which the U-shaped SED event loss does not possess."""
    m = nref_sub > 0
    L = (miss_sub[m] / nref_sub[m][:, None]).astype(np.float64)  # [n, G]
    n, G = L.shape
    if n == 0:
        return 0

    def select(risk_row, level):
        ok = risk_row <= level
        idx = np.where(ok)[0]
        return int(idx[-1]) if len(idx) else int(np.argmin(risk_row))

    rng = rng or np.random.default_rng(0)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        rows = rng.integers(0, n, n + 1)
        Lb = L[rows]                       # [n+1, G]
        S = Lb.sum(0)                      # [G]
        g_full = select(S / (n + 1), alpha)
        # leave-one-out risks for all i at once: (S - Lb[i]) / n
        loo = (S[None, :] - Lb) / n        # [n+1, G]
        ok = loo <= alpha
        # rightmost satisfying index per row (0 fallback -> argmin)
        rev = ok[:, ::-1]
        has = rev.any(1)
        g_loo = np.where(has, G - 1 - rev.argmax(1), loo.argmin(1))
        li = np.take_along_axis(Lb, g_loo[:, None], 1).ravel()
        deltas[b] = (li - Lb[np.arange(n + 1), g_full]).mean()
    beta_hat = max(float(deltas.mean()), 0.0)
    return select(L.mean(0), alpha - beta_hat)


def crc_nm(miss_sub, nref_sub, alpha, **kw):
    """CRC-NM of arXiv:2604.01502 (Aldirawi-Li-Guo, Theorem 1), Hoeffding
    variant at the deflated level alpha' = alpha - D(m,n),
    D = sqrt(log(2m)/(2n)) + 1/(2*sqrt(2n*log(2m))), clip-balanced loss
    (B=1). Selection walks a FIXED grid order and returns the first index
    satisfying (sum_i L_i + 1)/(n+1) <= alpha'. The paper orders its grid so
    the last element is a guaranteed-feasible conservative anchor; the
    U-shaped SED event loss has no such anchor, so we adapt by walking from
    lam=1 downward (first crossing = largest satisfying lam, the efficient
    direction) and note the violated anchor assumption. Guarantee (under the
    paper's assumptions): E[R(lam_hat)] <= alpha, in expectation only.
    Fallback when nothing satisfies: risk minimizer, uncertified."""
    m_ = nref_sub > 0
    L = (miss_sub[m_] / nref_sub[m_][:, None]).astype(np.float64)
    n, G = L.shape
    if n == 0:
        return 0
    D = np.sqrt(np.log(2 * G) / (2 * n)) + 1.0 / (2 * np.sqrt(2 * n * np.log(2 * G)))
    level = alpha - D
    crit = (L.sum(0) + 1.0) / (n + 1) <= level
    idx = np.where(crit)[0]
    if len(idx):
        return int(idx[-1])          # first crossing walking from lam=1 down
    return int(np.argmin(L.mean(0)))


ROUTES = {
    "crc_naive": crc_naive,
    "crc_c_nonmono": crc_c_nonmono,
    "crc_nm": crc_nm,
    "crc_split_mono": crc_split_mono,
    "ltt_split_clipmean": lambda ms, nr, a, **kw: ltt_split(ms, nr, a, loss="clipmean", **kw),
    "ltt_split_pooled": lambda ms, nr, a, **kw: ltt_split(ms, nr, a, loss="pooled", **kw),
    "ltt_split_pooledcap": ltt_split_pooledcap,
    "ltt_bonf_clipmean": lambda ms, nr, a, **kw: ltt_bonferroni(ms, nr, a, loss="clipmean", **kw),
    "ltt_bonf_pooled": lambda ms, nr, a, **kw: ltt_bonferroni(ms, nr, a, loss="pooled", **kw),
    "rcps_fixedseq": rcps_fixedseq,
    "rcps_fixedseq_pooled": lambda ms, nr, a, **kw: rcps_fixedseq(ms, nr, a, loss="pooled", **kw),
    # grid-equalized variants for the split-vs-Bonferroni deconfound
    "ltt_split_cm50": lambda ms, nr, a, **kw: ltt_split(ms, nr, a, loss="clipmean", n_candidates=50, **kw),
    "ltt_bonf_cm200": lambda ms, nr, a, **kw: ltt_bonferroni(ms, nr, a, loss="clipmean", n_candidates=200, **kw),
}
