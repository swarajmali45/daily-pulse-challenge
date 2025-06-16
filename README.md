# 🎬  “Mini Casting Pulse” Coding Exercise

Actors find work through an online casting platform—basically a job board dedicated to film, TV, commercials, and voice work. When an actor refreshes their casting feed, a live stream of brand-new breakdowns appears. A breakdown is a structured role notice: it lists the project type (film, series, commercial, etc.), shoot location, pay rate, union status, and the age/gender/ethnicity the casting director is looking for, along with a short role description.

Every time one of our users sees a role notice, we record that breakdown as a single row in a raw data table. Your challenge is to take those raw rows and roll them up into a clean, daily “Casting Pulse” summary. This file will let industry analysts quickly spot trends such as rising day-rates, regional production shifts, or surges in particular project types—without ever exposing details about individual actors or roles.

> **TL;DR** – From one raw CSV you’ll generate a single `daily_pulse.csv` that buckets, aggregates, and lightly enriches the data; we’ll inspect the output and your code.

---

## 📂  Repo layout

```

data/
breakdowns\_sample.csv   # \~1000 breakdown rows (same schema we use in prod)
pulse.py                # <-- you create this script
daily\_pulse.csv         # <-- put your final daily\_pulse.csv here
README.md               # (this file)

````

---

## 🎯  What you need to build

Run:

```bash
python build_pulse.py \
       --input  breakdowns_sample.csv \
       --output daily_pulse.csv
````

Your script should finish in **< 5 min** on a laptop (8 GB RAM).

### 🔑  Required columns & rules

| Column                | Build rule                                                                               |
| --------------------- | ---------------------------------------------------------------------------------------- |
| `date_utc`            | `posted_date` truncated to `YYYY-MM-DD` (UTC)                                            |
| `region_code`         | Map `work_city` → `LA`, `NY`, etc.  <br>*↳ Pick any sane mapping & document it*          |
| `proj_type_code`      | Map `project_type` → `F` (film), `T` (TV/streaming), `C` (commercial), `V` (voice/other) |
| `role_count_day`      | Number of rows in that bucket                                                            |
| `lead_share_pct_day`  | `(Lead + Principal rows) ÷ role_count_day`, **1 dp**                                     |
| `union_share_pct_day` | `(union rows) ÷ role_count_day`, **1 dp**                                                |
| `median_rate_day_usd` | Median `rate_value`, **rounded to the nearest \$25**                                    |
| `sentiment_avg_day`   | Score each `role_description` −1…+1 (any open model), store bucket mean **rounded 0.05** |

*Extra credit:* `theme_ai_share_pct_day` – % of rows whose text contains “AI” / “robot” / “android”.

Sort by `date_utc, region_code, proj_type_code`.

---

## 🤫  Intentional vagueness — and what we’re testing

| We left this vague…                      | Because we want to see…                                                                                                        |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Country→Region map**                   | Can you design & document a simple taxonomy that works well with our preferred hourly × city cuts?                             |
| **Project-type mapping**                 | Can you normalize messy categorical data?                                                                                      |
| **Sentiment tool choice**                | Pragmatic ML instinct (TextBlob? HuggingFace? OpenAI API? Your call)                                                           |
| **Privacy touches**<br>(k-anon, Laplace) | Can you reason about privacy? Bonus if you drop low-volume buckets (e.g. <5 rows) or add Laplace noise to prevent tracebacks. |
| **No tests, no CI**                      | Code clarity & runnable script matter more than boilerplate.                                                                   |

> **Note on Privacy & Noise**
> Our goal is to expose rich, meaningful production trends—*without* leaking any personally identifiable or traceable role-level data. Anonymize your outputs by:

* Dropping sparse groupings (e.g., `role_count_day` < 5)
* Adding light noise: `np.random.laplace(0, 1)` to any sensitive counts or rates
  This encourages a differential privacy mindset while maintaining analytic fidelity.

---

## 📝  Submission checklist

1. ✅ `build_pulse.py` – clear functions, docstrings, sensible libs.
2. ✅ `output/daily_pulse.csv` – ≤ 30 MB (gzip if larger).
3. ✅ **< 15-word** note in this README (bottom) explaining:

   * your country/region mapping
   * your project-type mapping
   * sentiment library you used
4. Push to your fork or send us a zipped repo link.

---

## 🚀  Hints

* **Pandas groupby** is plenty for 5K rows.
* Median rounding: `int(round(x / 25.0)) * 25`
* For a quick sentiment baseline: `pip install textblob`
* Want anonymization? Drop buckets `< 5` rows or add `np.random.laplace(0, 1)` to counts.

## My README

Region map: city → LA/NY/CHI/etc.  
Project-type map: film→F, series→T, commercial→C, voice/other→V  
Sentiment: TextBlob  
