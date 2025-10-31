# 📘 Distributions and Expectations – Summary  
### סיכום – התפלגויות ותוחלות

---

## 🇬🇧 **English Version**

### 🎲 Random Variables
- Represent outcomes of random experiments.  
- Can be **discrete** (countable values) or **continuous** (range of values).

### 📊 Probability Mass / Density Function
- **PMF (discrete):** \( P(X = x) \)  
- **PDF (continuous):** \( f(x) \), where \( P(a ≤ X ≤ b) = \int_a^b f(x)\,dx \)  
- Always: \( \sum P(X=x) = 1 \) or \( \int f(x)\,dx = 1 \)

### 📈 Cumulative Distribution Function (CDF)
\[
F(x) = P(X \le x)
\]
Gives the probability that the variable is smaller than or equal to a certain value.  
Always increases from **0 → 1**.

### 💡 Expectation & Variance
\[
E[X] = \sum x P(X=x)
\]
\[
Var(X) = E[(X - \mu)^2] = E[X^2] - (E[X])^2
\]
\[
\sigma = \sqrt{Var(X)}
\]

### 🔹 Chebyshev’s Inequality
\[
P(|X - \mu| > k\sigma) \le \frac{1}{k^2}
\]
At least 75% of values lie within 2σ of the mean (for any distribution).

### 🎯 Binomial Distribution
\[
P(Y=k) = \binom{n}{k} p^k (1-p)^{n-k}
\]
\[
E[Y] = np, \quad Var(Y) = np(1-p)
\]
Used when counting successes in **n independent trials** with success probability **p**.

### ⚖️ Uniform Distribution (discrete)
\[
P(X=x) = \frac{1}{b-a+1}, \quad E[X]=\frac{a+b}{2}, \quad Var(X)=\frac{(b-a+1)^2-1}{12}
\]

### 🎲 Example – Two Dice
\[
E[Y]=7, \quad Var(Y)=5.83, \quad \sigma\approx2.41
\]
Symmetric distribution peaking at 7.

✅ **Key Idea:**  
Distributions describe **how probability spreads** across possible outcomes.  
Expectation = average outcome, Variance = how spread out it is.

---

🔗 [Zoom Session](https://applications.zoom.us/lti/rich/open?x_zm_session_id_token=eyJ6bV9za20iOiJ6bV9vMm0iLCJ0eXAiOiJKV1QiLCJrIjoiV1QxUUQybTIiLCJhbGciOiJFUzI1NiJ9.eyJhdWQiOiJpbnRlZ3JhdGlvbiIsImlzcyI6ImludGVncmF0aW9uIiwiZXhwIjoxNzYxOTAyNTEyLCJpYXQiOjE3NjE5MDA3MTIsImp0aSI6ImRkMTczYmQzLTEyMDAtNGJkNC05NTJiLWYzM2JmNTczNWM1NyJ9.OjQ2MwEAn8zBFktEIc5tjEyH0_ibOMWQd3h4u0RJJEIM9m_oKvPfYoU0Y4AHK6ibqmoUjQSXxPB3v_3b8gwGfA&oauth_consumer_key=zLWWkB6vRTeTjgCDshrbQQ&lti_scid=75e2bec298a389ee6cbd2692a5d056e25965cffe50c182055e9b3364f58ea50b)

---

## 🇮🇱 **גרסה בעברית**

### 🎲 משתנים מקריים
מייצגים תוצאה של ניסוי אקראי.  
יכולים להיות **בדידים** (ערכים ספציפיים) או **רציפים** (טווח שלם).

### 📊 פונקציית הסתברות
- **PMF (בדיד):** \( P(X = x) \)  
- **PDF (רציף):** \( f(x) \), כך ש־\( P(a ≤ X ≤ b) = \int_a^b f(x)\,dx \)  
- תמיד: \( \sum P(X=x)=1 \) או \( \int f(x)\,dx=1 \)

### 📈 פונקציית התפלגות מצטברת (CDF)
\[
F(x) = P(X \le x)
\]
נותנת את ההסתברות שהמשתנה קטן או שווה לערך מסוים.  
תמיד עולה מ־0 עד 1.

### 💡 תוחלת ושונות
\[
E[X] = \sum x P(X=x)
\]
\[
Var(X) = E[(X - \mu)^2] = E[X^2] - (E[X])^2
\]
\[
\sigma = \sqrt{Var(X)}
\]

### 🔹 משפט צ’בישב
\[
P(|X - \mu| > k\sigma) \le \frac{1}{k^2}
\]
לפחות 75% מהערכים נמצאים עד 2 סטיות תקן מהממוצע.

### 🎯 התפלגות בינומית
\[
P(Y=k) = \binom{n}{k} p^k (1-p)^{n-k}
\]
\[
E[Y] = np, \quad Var(Y) = np(1-p)
\]
מתארת מספר הצלחות מתוך **n ניסויים עצמאיים** עם הסתברות הצלחה **p**.

### ⚖️ התפלגות אחידה (בדידה)
\[
P(X=x)=\frac{1}{b-a+1}, \quad E[X]=\frac{a+b}{2}, \quad Var(X)=\frac{(b-a+1)^2-1}{12}
\]

### 🎲 דוגמה – שתי קוביות
\[
E[Y]=7, \quad Var(Y)=5.83, \quad \sigma\approx2.41
\]
ההתפלגות סימטרית סביב הערך 7.

✅ **רעיון מרכזי:**  
התפלגות מתארת איך ההסתברות מתפזרת על פני הערכים האפשריים.  
התוחלת מייצגת ממוצע, והשונות את מידת הפיזור סביבו.

---

🔗 [כניסה לזום](https://applications.zoom.us/lti/rich/open?x_zm_session_id_token=eyJ6bV9za20iOiJ6bV9vMm0iLCJ0eXAiOiJKV1QiLCJrIjoiV1QxUUQybTIiLCJhbGciOiJFUzI1NiJ9.eyJhdWQiOiJpbnRlZ3JhdGlvbiIsImlzcyI6ImludGVncmF0aW9uIiwiZXhwIjoxNzYxOTAyNTEyLCJpYXQiOjE3NjE5MDA3MTIsImp0aSI6ImRkMTczYmQzLTEyMDAtNGJkNC05NTJiLWYzM2JmNTczNWM1NyJ9.OjQ2MwEAn8zBFktEIc5tjEyH0_ibOMWQd3h4u0RJJEIM9m_oKvPfYoU0Y4AHK6ibqmoUjQSXxPB3v_3b8gwGfA&oauth_consumer_key=zLWWkB6vRTeTjgCDshrbQQ&lti_scid=75e2bec298a389ee6cbd2692a5d056e25965cffe50c182055e9b3364f58ea50b)
