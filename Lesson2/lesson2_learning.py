# -*- coding: utf-8 -*-
"""
Lesson 2 – Hebrew Learning Script
---------------------------------
סקריפט זה נועד ללמידה עצמית: לכל חלק יש הסבר תיאורטי קצר + קוד רץ.
תלויות: numpy, matplotlib
הרצה:
    python lesson2_learning.py            # מריץ הכול ומציג גרפים
    python lesson2_learning.py --save     # שומר גרפים לתיקיית out
"""
import numpy as np, math, argparse
import matplotlib.pyplot as plt
from math import comb
np.random.seed(42)

def save_or_show(fig, name, save):
    import pathlib
    out = pathlib.Path("out"); out.mkdir(exist_ok=True)
    if save:
        fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=140)
        print("[נשמר]", out / f"{name}.png")
    else:
        plt.show()

def section_two_dice(save=False):
    """שתי קוביות – PMF, תוחלת ושונות (הדגמה מהמצגת)."""
    sums = [i+j for i in range(1,7) for j in range(1,7)]
    s_vals, counts = np.unique(sums, return_counts=True)
    pmf = counts / counts.sum()
    E = (s_vals*pmf).sum()
    Var = ((s_vals**2)*pmf).sum() - E**2
    print("[שתי קוביות] E=%.4f Var=%.4f" % (E, Var))
    fig = plt.figure()
    plt.stem(s_vals, pmf, use_line_collection=True)
    plt.title("PMF — סכום שתי קוביות"); plt.xlabel("סכום"); plt.ylabel("הסתברות")
    save_or_show(fig, "two_dice_pmf", save)

def section_independence():
    """בדיקת עצמאות: P(G∈{3,6}, R=5) = P(G∈{3,6})P(R=5)."""
    outcomes = [(r,g) for r in range(1,7) for g in range(1,7)]
    p = 1/36
    P_joint = sum(1 for r,g in outcomes if g in {3,6} and r==5) * p
    P_G = sum(1 for r,g in outcomes if g in {3,6}) * p
    P_R = sum(1 for r,g in outcomes if r==5) * p
    print("[עצמאות] משותף=", P_joint, " מכפלה=", P_G*P_R)

def section_binomial():
    """בינומיאלי – נוסחאות מול סימולציה."""
    n,p = 20, 0.3
    theo_E, theo_V = n*p, n*p*(1-p)
    samp = np.random.binomial(n, p, size=100000)
    print("[בינומיאלי] תאורטי:", theo_E, theo_V, " סימולציה:", float(samp.mean()), float(samp.var(ddof=0)))

def section_geometric():
    """גיאומטרית – זמן עד הצלחה ראשונה."""
    p = 0.25
    samp = np.random.geometric(p, size=50000)
    print("[גיאומטרית] תאורטי:", 1/p, (1-p)/p**2, " סימולציה:", float(samp.mean()), float(samp.var(ddof=0)))

def negbin_trials(r, p, size):
    out=[]
    for _ in range(size):
        s=0; t=0
        while s<r:
            t+=1
            if np.random.rand()<p:
                s+=1
        out.append(t)
    return np.array(out)

def section_randomistan():
    """נג-בינומיאלי – השוואת E/Var ואמידת P(X1>X2)."""
    r,p,m = 5,0.4,2
    N=60000
    X1 = negbin_trials(r,p,N)
    X2 = negbin_trials(m*r, m*p, N)
    print("[Randomistan] E תאורטי:", r/p, (m*r)/(m*p))
    print("E סימולציה:", float(X1.mean()), float(X2.mean()))
    print("Var סימולציה:", float(X1.var(ddof=0)), float(X2.var(ddof=0)))
    print("P(X1>X2):", float((X1>X2).mean()))

def section_poisson_zero():
    """פואסון(λ=0.5) – P(0) בשנייה וב-10 שניות."""
    lam=0.5
    print("[Poisson] P(X=0) שנייה:", math.exp(-lam))
    print("P(0 ב-10 שניות) (Poisson(5)):", math.exp(-10*lam))
    print("דרך עצמאות 10 שניות:", (math.exp(-lam))**10)

def poisson_pmf(k, lam): return math.exp(-lam)*(lam**k)/math.factorial(k)

def section_binom_to_poisson(save=False):
    """גבול Binomial→Poisson – גרף התקרבות (L1)."""
    lam=3.0
    ks = np.arange(0,15)
    fig = plt.figure()
    for n in [10,50,100,200,500,1000]:
        p = lam/n
        pmf_bin = np.array([comb(n,k)*(p**k)*((1-p)**(n-k)) for k in ks])
        pmf_poi = np.array([poisson_pmf(k,lam) for k in ks])
        L1 = np.abs(pmf_bin-pmf_poi).sum()
        plt.plot(ks, pmf_bin, marker="o", label=f"Binom n={n}, L1={L1:.3f}")
    plt.plot(ks, [poisson_pmf(k,lam) for k in ks], marker="x", linestyle="--", label="Poisson(λ)")
    plt.title("Binom(np=λ) → Poisson(λ)"); plt.xlabel("k"); plt.ylabel("PMF"); plt.legend()
    save_or_show(fig, "binom_to_poisson", save)

def section_sum_poissons(save=False):
    """סכום שני פואסונים – אימות אמפירי."""
    lam1, lam2 = 1.2, 2.8
    N=50000
    X = np.random.poisson(lam1, size=N)
    Y = np.random.poisson(lam2, size=N)
    Z = X+Y
    theo = lam1+lam2
    print("[SumPois] תאורטי E=Var:", theo, " סימולציה:", float(Z.mean()), float(Z.var(ddof=0)))
    k = np.arange(0, Z.max()+1)
    pmf_theo = np.array([poisson_pmf(int(kk), theo) for kk in k])
    fig = plt.figure()
    plt.hist(Z, bins=range(0,int(Z.max())+2), density=True, alpha=0.6)
    plt.plot(k, pmf_theo, marker="o")
    plt.title("סכום שני פואסונים – התאמה לפואסון(λ1+λ2)"); plt.xlabel("k"); plt.ylabel("תדירות/PMF")
    save_or_show(fig, "sum_of_poissons", save)

def section_rate_vs_n():
    """השוואת P(X≥1) ו-E בין Binom(1,λ) לבין Binom(2,λ/2)."""
    lam=0.4; p1, p2 = lam, lam/2
    P1 = 1-(1-p1)**1; P2 = 1-(1-p2)**2
    E1, E2 = 1*p1, 2*p2
    print("[Rate vs n] P≥1:", P1, P2, " E:", E1, E2)

def section_third_moment():
    """רגע שלישי מרכזי לבינומיאלי – אימות נומרי."""
    def third_central_moment_binom(n,p):
        ks = np.arange(0, n+1)
        pmf = np.array([comb(n,k)*(p**k)*((1-p)**(n-k)) for k in ks])
        mu = n*p
        return np.sum(((ks-mu)**3)*pmf)
    for (n,p) in [(10,0.3),(20,0.4),(50,0.1)]:
        theo = n*p*(1-p)*(1-2*p)
        emp = third_central_moment_binom(n,p)
        print(f"[μ3] n={n}, p={p} תאורטי={theo:.6f} חישוב={emp:.6f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="שמור גרפים לתיקיית out")
    args = ap.parse_args()
    section_two_dice(save=args.save)
    section_independence()
    section_binomial()
    section_geometric()
    section_randomistan()
    section_poisson_zero()
    section_binom_to_poisson(save=args.save)
    section_sum_poissons(save=args.save)
    section_rate_vs_n()
    section_third_moment()

if __name__ == "__main__":
    main()
