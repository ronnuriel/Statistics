# 📚 Lesson 2 – Statistics Learning Package | חבילת לימוד לשיעור 2

This project includes **a full learning environment** for understanding probability distributions,
expectation, variance, and independence — with explanations in **Hebrew and English**, and runnable code.

הפרויקט הזה כולל **סביבת לימוד מלאה בעברית ובאנגלית**, שמסבירה בצורה מדויקת וברורה את שיעור 2:
התפלגויות, תוחלת, שונות, פואסון, בינומיאלי, גיאומטרי, עצמאות ועוד — עם קוד רץ.

---

## 📂 Project Structure | מבנה הפרויקט

```
lesson2/
├── lesson2_learning_he.ipynb   ← מחברת Jupyter עם הסברים + קוד
├── lesson2_learning.py         ← סקריפט פייתון לימודי
├── README_lesson2_learning.md  ← קובץ ההסבר (אתה קורא אותו עכשיו)
└── out/                        ← נשמרים גרפים (אם משתמשים ב־--save)
```

---

## ⚙️ Installation | התקנה

```
pip install numpy matplotlib
```

---

## ▶️ How to Run | איך להריץ

### ✅ Option 1 – Jupyter Notebook

```
jupyter notebook lesson2_learning_he.ipynb
```

- Contains explanations + code + outputs.
- Includes formulas, Hebrew explanations and simulation results.
- הכי טוב ללמידה בקצב שלך.

---

### ✅ Option 2 – Python Script

```
python lesson2_learning.py
```

Want the graphs saved automatically?
```
python lesson2_learning.py --save
```

---

## 🎯 Topics Covered | נושאים שנלמדו

| Topic | תוכן |
|-------|------|
| Expected Value & Variance | תוחלת, שונות, סטיית תקן |
| Sum of Two Dice | סכום 2 קוביות, PMF, תוחלת, שונות |
| Independence | עצמאות סטטיסטית בין אירועים |
| Bernoulli & Binomial | ברנולי ובינומיאלי — נוסחאות + סימולציה |
| Geometric | ניסיונות עד הצלחה ראשונה |
| Negative Binomial (Randomistan) | מספר ניסיונות עד r הצלחות |
| Poisson Distribution | ביקורים באתר, P(X=0) |
| Binomial → Poisson Limit | גבול בינומיאלי לפואסון |
| Sum of Independent Poisson | סכום משתנים פואסוניים |
| Covariance & Var(X+Y) | קו-וריאנס, שונות סכום |
| Binomial rate (n=1 vs n=2) | השוואת P(X≥1), E[X] |
| Third Central Moment | רגע שלישי מרכזי (Skewness) |

---

## 💡 Study Tips | טיפים ללמידה

✅ רוץ על תא → נסה להסביר במילים → רק אז המשך הלאה  
✅ תשנה ערכים (p,n,λ) ותראה מה קורה – זה מקבע הבנה  
✅ תשווה תמיד בין נוסחה תיאורטית לתוצאה אמפירית  
✅ אם משהו לא ברור → תחזור לשקף לפי הכותרת שרשומה במחברת  
 
