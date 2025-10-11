"""
AI Agent - Analyst Workbench (No API Required)
Features: Overview • Cleaning • GroupBy • Sorting • Filtering • Correlations •
Time Series • Prophet/Statsmodels Forecast (fallback MA) • Outliers • Data Quality •
Sales Pipeline • ML (Regress/Classify) • Churn • Cohort • Funnel • RFM • Advisor
Copyright © 2025 Gardel Hiram
"""
import logging
import re
from typing import Optional, List, Tuple, Dict, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ------------------------
# Optional dependencies
# ------------------------
_HAS_SKLEARN = False
_HAS_PROPHET = False
_HAS_STATSMODELS = False

try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score
    _HAS_SKLEARN = True
except Exception:
    pass

# Prophet may be `prophet` (new) or `fbprophet` (legacy)
try:
    from prophet import Prophet  # type: ignore
    _HAS_PROPHET = True
except Exception:
    try:
        from fbprophet import Prophet  # type: ignore
        _HAS_PROPHET = True
    except Exception:
        pass

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _HAS_STATSMODELS = True
except Exception:
    pass


# =========================
# Utility helpers (no extra deps)
# =========================

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def _num_in_text(q: str, default: int, cap: int) -> int:
    m = re.search(r"\b(\d{1,5})\b", q)
    if not m: return default
    try:
        return max(1, min(int(m.group(1)), cap))
    except Exception:
        return default

def _format_num(x) -> str:
    try:
        if isinstance(x, (int, np.integer)): return f"{int(x):,}"
        if isinstance(x, (float, np.floating)):
            return f"{x:,.2f}" if abs(x - round(x)) > 1e-9 else f"{int(round(x)):,}"
        return str(x)
    except Exception:
        return str(x)

def _score_match(name: str, query: str) -> float:
    n = _norm(name); q = _norm(query)
    nt = set(re.split(r"[^a-z0-9]+", n)) - {""}
    qt = set(re.split(r"[^a-z0-9]+", q)) - {""}
    if not nt: return 0.0
    overlap = len(nt & qt) / len(nt)
    bonus = 0.25 if n in q or any(t and t in n for t in qt) else 0.0
    return min(1.0, overlap + bonus)

def _best_column(df: pd.DataFrame, query: str, want: Optional[str] = None) -> Optional[str]:
    cols = df.columns.tolist()
    if want == "numeric":
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    elif want == "text":
        cols = df.select_dtypes(include=["object","string","category"]).columns.tolist()
    elif want == "datetime":
        dcols = df.select_dtypes(include=["datetime64[ns]","datetime64[ns, UTC]"]).columns.tolist()
        if dcols: cols = dcols
        else:
            cols = [c for c in df.columns if "date" in _norm(c)]
            for c in cols:
                try:
                    pd.to_datetime(df[c], errors="raise"); return c
                except Exception: pass
            return None
    if not cols: return None
    for c in cols:
        if _norm(c) in _norm(query): return c
    best, score = None, 0.0
    for c in cols:
        s = _score_match(c, query)
        if s > score: best, score = c, s
    return best if score >= 0.25 else None

def _head_text(df: pd.DataFrame, n: int) -> str:
    n = max(1, min(n, 50))
    return df.head(n).to_string(index=False)

def _tail_text(df: pd.DataFrame, n: int) -> str:
    n = max(1, min(n, 50))
    return df.tail(n).to_string(index=False)

def _parse_simple_filter(query: str) -> List[Tuple[str, str, str]]:
    q = _norm(query)
    pairs = re.findall(r"(?:where|filter)\s+([^=<>!]+?)\s*(=|==|>=|<=|>|<|!=)\s*\"?([\w\-\.\s%:/]+)\"?", q)
    out = [(c.strip(), op, v.strip()) for c,op,v in pairs]
    contains = re.findall(r"(?:where|filter)\s+([^=<>!]+?)\s+contains\s+\"?([\w\-\.\s%:/]+)\"?", q)
    out += [(c.strip(), "contains", v.strip()) for c,v in contains]
    return out

def _apply_filters(df: pd.DataFrame, query: str) -> pd.DataFrame:
    filters = _parse_simple_filter(query)
    if not filters: return df
    work = df.copy()
    for col_raw, op, val_raw in filters:
        col = _best_column(work, col_raw)
        if not col or col not in work.columns: continue
        s = work[col]
        if op in {">", ">=", "<", "<=", "==", "=", "!="}:
            comp = "==" if op == "=" else op
            try:
                if s.dtype.kind in "biufc":
                    v = float(val_raw.replace(",", ""))
                else:
                    v = val_raw
            except Exception:
                v = val_raw
            try:
                if comp == "==": work = work[s == v]
                elif comp == "!=": work = work[s != v]
                elif comp == ">":  work = work[s.astype(float) >  float(v)]
                elif comp == ">=": work = work[s.astype(float) >= float(v)]
                elif comp == "<":  work = work[s.astype(float) <  float(v)]
                elif comp == "<=": work = work[s.astype(float) <= float(v)]
            except Exception:
                sv = s.astype(str).str.lower(); vv = str(v).lower()
                if comp == "==": work = work[sv == vv]
                elif comp == "!=": work = work[sv != vv]
        elif op == "contains":
            try: work = work[s.astype(str).str.contains(val_raw, case=False, na=False)]
            except Exception: pass
    return work

def _detect_sales_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = list(df.columns)
    lc = {c: _norm(c) for c in cols}
    stage = next((c for c in cols if "stage" in lc[c]), None)
    amount = next((c for c in cols if any(k in lc[c] for k in ["amount","value","revenue","arr","mrr"])), None)
    owner = next((c for c in cols if any(k in lc[c] for k in ["owner","rep","sales rep","account manager"])), None)
    date  = next((c for c in cols if any(k in lc[c] for k in ["close date","closed","date"])), None)
    return {"stage": stage, "amount": amount, "owner": owner, "date": date}

def _closed_mask(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.lower()
    return s.str.contains("won|closed", regex=True)

# =========================
# Agent
# =========================

class AIAgent:
    def __init__(self):
        logger.info("✅ Analyst Workbench (No API) initialized")

    def analyze_data(self, df: pd.DataFrame, query: str, data_summary: Optional[dict] = None) -> Dict[str, Any]:
        try:
            if not isinstance(df, pd.DataFrame) or df.empty:
                return {"text": "⚠️ I don't see a loaded dataset yet."}

            q = _norm(query)
            work = _apply_filters(df, q)

            # -------- Intents (ordered) --------
            if re.search(r"\b(advice|advisor|recommend|what next|help me decide)\b", q):
                return self._advisor(work)

            if re.search(r"\b(columns?|fields?|schema|variables?)\b", q):
                return self._columns(work)

            if re.search(r"\b(rows?|row count|records?)\b", q):
                return self._shape(work)

            if re.search(r"\b(missing|nulls?|nans?|empty|blanks?|na\b)\b", q):
                return self._missing(work)

            if re.search(r"\b(dtypes?|data types?|types?)\b", q):
                return self._types(work)

            if re.search(r"\b(summary|describe|statistics|stats|overview)\b", q):
                return self._summary(work)

            if re.search(r"\b(duplicate|dedup|duplicates)\b", q):
                return self._duplicates(work)

            if re.search(r"\b(correlat|relationship|related)\b", q):
                return self._correlations(work)

            if re.search(r"\b(value counts?|top categories?|top values?|frequency|mode)\b", q):
                return self._value_counts(work, q)

            if re.search(r"\b(head|preview|first|show|display|view)\b", q):
                n = _num_in_text(q, 5, 50)
                return self._preview(work, n, True)

            if re.search(r"\b(tail|last)\b", q):
                n = _num_in_text(q, 5, 50)
                return self._preview(work, n, False)

            if re.search(r"\b(sort|order by|top|largest|smallest)\b", q):
                return self._sort_top_bottom(work, q)

            if re.search(r"\b(group(ed)? by|aggregate|agg|sum|average|mean|median|count)\b", q) and "by" in q:
                return self._groupby(work, q)

            if re.search(r"\b(unique|distinct|different)\b", q):
                return self._unique(work, q)

            if re.search(r"\b(memory|size|storage)\b", q):
                return self._memory(work)

            if re.search(r"\b(outlier|iqr|z-?score|std dev)\b", q):
                return self._outliers(work, q)

            if re.search(r"\b(time|trend|over time|by month|by week|resample|timeseries|date)\b", q):
                return self._time_trend(work, q)

            if re.search(r"\b(forecast|predict|projection)\b", q):
                return self._forecast(work, q)

            if re.search(r"\b(quality|trust score|data quality)\b", q):
                return self._data_quality(work)

            if re.search(r"\b(sales|pipeline|deals?|opportunit|win rate|funnel)\b", q):
                return self._sales_pipeline(work)

            # ML
            if re.search(r"\b(regression|regress|predict .* (value|amount|price|score))\b", q):
                return self._ml_regression(work, q)
            if re.search(r"\b(classification|classify|predict .* (churn|category|status|label))\b", q):
                return self._ml_classification(work, q)

            # Domain plug-ins
            if re.search(r"\b(churn)\b", q):
                return self._churn_report(work, q)
            if re.search(r"\b(cohort)\b", q):
                return self._cohort_analysis(work, q)
            if re.search(r"\b(funnel)\b", q):
                return self._funnel_by_step(work, q)
            if re.search(r"\b(rfm|recency|frequency|monetary)\b", q):
                return self._rfm_scoring(work, q)

            # Min/Max/Average
            if re.search(r"\b(max(imum)?|highest|largest)\b", q):
                return self._extreme(work, q, "max")
            if re.search(r"\b(min(imum)?|lowest|smallest)\b", q):
                return self._extreme(work, q, "min")
            if re.search(r"\b(average|mean|avg)\b", q):
                return self._average(work, q)

            return self._help(work)

        except Exception as e:
            logger.exception("Analysis failed.")
            return {"text": f"❌ I hit an error: {e}"}

    # ------------------ Baseline handlers (unchanged ideas, condensed) ------------------

    def _columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        cols = df.columns.tolist()
        nums = df.select_dtypes(include=[np.number]).columns.tolist()
        cats = df.select_dtypes(include=["object","string","category"]).columns.tolist()
        dts  = df.select_dtypes(include=["datetime64[ns]","datetime64[ns, UTC]"]).columns.tolist()
        text = (f"📊 **Columns ({len(cols)})**\n\n"
                f"**All:** {', '.join(cols[:60])}{' ...' if len(cols)>60 else ''}\n\n"
                f"{'**Numeric:** ' + ', '.join(nums[:30]) + (' ...' if len(nums)>30 else '') + '\n\n' if nums else ''}"
                f"{'**Categorical/Text:** ' + ', '.join(cats[:30]) + (' ...' if len(cats)>30 else '') + '\n\n' if cats else ''}"
                f"{'**Datetime:** ' + ', '.join(dts[:30]) + (' ...' if len(dts)>30 else '') if dts else ''}")
        return {"text": text}

    def _shape(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {"text": f"📈 **Rows:** {len(df):,}   •   **Columns:** {len(df.columns)}"}

    def _missing(self, df: pd.DataFrame) -> Dict[str, Any]:
        m = df.isna().sum(); total = int(m.sum())
        if total == 0: return {"text": "✅ No missing values detected."}
        pct = (m / max(1, len(df)) * 100).round(2)
        lines = [f"⚠️ **Missing data** — total missing cells: **{total:,}**", ""]
        for col, n in m.sort_values(ascending=False).items():
            if n > 0: lines.append(f"- **{col}**: {n:,} ({pct[col]}%)")
        return {"text": "\n".join(lines[:120])}

    def _types(self, df: pd.DataFrame) -> Dict[str, Any]:
        lines = ["🔎 **Column dtypes**", ""]
        for c, t in df.dtypes.items():
            lines.append(f"- **{c}**: `{t}`")
        return {"text": "\n".join(lines)}

    def _summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        nums = df.select_dtypes(include=[np.number])
        if nums.empty: return {"text": "ℹ️ No numeric columns for summary."}
        desc = nums.describe().transpose().round(3)
        return {"text": f"📊 **Summary** (showing up to 25)", "payload": {"type": "preview", "table": desc.head(25)}}

    def _duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        dup = int(df.duplicated().sum())
        return {"text": f"🧹 **Duplicate rows:** {dup:,}" + ("" if dup==0 else "\n\nTip: consider dropping duplicates.")}

    def _correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        nums = df.select_dtypes(include=[np.number])
        if nums.shape[1] < 2: return {"text": "ℹ️ Need at least two numeric columns."}
        corr = nums.corr(numeric_only=True)
        pairs = []
        cs = corr.columns.tolist()
        for i in range(len(cs)):
            for j in range(i+1, len(cs)):
                r = corr.iloc[i,j]
                if np.isfinite(r) and abs(r) >= 0.3: pairs.append({"A": cs[i], "B": cs[j], "r": float(r)})
        text = "🔗 **Top correlations (|r| ≥ 0.3)**" if pairs else "ℹ️ No correlations above |0.3|."
        pairs_df = pd.DataFrame(sorted(pairs, key=lambda x: abs(x["r"]), reverse=True)).head(25) if pairs else pd.DataFrame()
        return {"text": text, "payload": {"type":"correlations", "pairs": pairs_df, "matrix_head": corr.head(12)}}

    def _value_counts(self, df: pd.DataFrame, q: str) -> Dict[str, Any]:
        col = _best_column(df, q, want="text") or _best_column(df, q)
        if not col: return {"text": "ℹ️ Tell me which column to count (e.g., 'value counts for Stage')."}
        vc = df[col].astype(str).value_counts(dropna=True)
        return {"text": f"📚 **Top values in {col}** (up to 25)", "payload": {"type": "value_counts", "table": vc.head(25).reset_index(names=[col, "count"])}}

    def _preview(self, df: pd.DataFrame, n: int, head: bool) -> Dict[str, Any]:
        table = df.head(n) if head else df.tail(n)
        label = "First" if head else "Last"
        return {"text": f"👁️ **{label} {n} rows** (after filters)", "payload": {"type":"preview", "table": table}}

    def _sort_top_bottom(self, df: pd.DataFrame, q: str) -> Dict[str, Any]:
        n = _num_in_text(q, 10, 100)
        desc = bool(re.search(r"\b(desc|descending|largest|top|highest)\b", q))
        asc  = bool(re.search(r"\b(asc|ascending|smallest|lowest|bottom)\b", q))
        m = re.search(r"\bby\s+([a-z0-9 _\-\.]+)$", q)
        col = _best_column(df, (m.group(1) if m else q), want="numeric") or _best_column(df, (m.group(1) if m else q))
        if not col: return {"text": "ℹ️ Please specify a column to sort by (e.g., 'top 10 by amount')."}
        order = True if asc and not desc else False
        try:
            s = df.sort_values(by=col, ascending=order)
        except Exception:
            s = df.sort_values(by=col, key=lambda x: x.astype(str), ascending=order)
        table = s.head(n) if asc else s.tail(n)
        label = "Smallest" if asc else "Top"
        return {"text": f"📑 **{label} {n} by {col}**", "payload": {"type":"sort", "table": table}}

    def _groupby(self, df: pd.DataFrame, q: str) -> Dict[str, Any]:
        agg_map = {"mean":["average","mean","avg"], "sum":["sum","total"], "median":["median"], "count":["count","freq","frequency","n"]}
        which = "mean"
        for k, syns in agg_map.items():
            if any(s in q for s in syns): which = k
        target = _best_column(df, q, want="numeric") or _best_column(df, q)
        m = re.search(r"\bby\s+([a-z0-9 _\-\.]+)", q)
        group = _best_column(df, m.group(1)) if m else None
        if not (target and group): return {"text": "ℹ️ Try 'average <numeric col> by <group col>'."}
        try:
            g = df.groupby(group, dropna=False)[target]
            stat = getattr(g, which)() if which != "count" else g.count()
            show = stat.sort_values(ascending=False).head(25).reset_index()
            return {"text": f"🧮 **{which.title()} of {target} by {group}** (top 25)", "payload": {"type": "groupby", "table": show}}
        except Exception as e:
            return {"text": f"❌ Grouping failed: {e}"}

    def _unique(self, df: pd.DataFrame, q: str) -> Dict[str, Any]:
        col = _best_column(df, q) or _best_column(df, q, want="text")
        if not col: return {"text": "ℹ️ Which column do you want unique values for?"}
        n = int(df[col].nunique(dropna=True))
        sample = df[col].dropna().astype(str).unique()[:12]
        return {"text": f"🔢 **{col}** has **{n}** unique values.\n\nExamples: {', '.join(map(str, sample))}{'' if n<=12 else ' …'}"}

    def _memory(self, df: pd.DataFrame) -> Dict[str, Any]:
        mb = df.memory_usage(deep=True).sum() / (1024**2)
        return {"text": f"💾 **Memory:** {mb:.2f} MB   •   Cells: {len(df)*len(df.columns):,}"}

    def _outliers(self, df: pd.DataFrame, q: str) -> Dict[str, Any]:
        col = _best_column(df, q, want="numeric")
        if not col: return {"text": "ℹ️ Specify a numeric column for outlier analysis."}
        s = df[col].dropna().astype(float)
        if s.empty: return {"text": f"ℹ️ No numeric data in **{col}**."}
        if "iqr" in q:
            q1,q3 = s.quantile(0.25), s.quantile(0.75); iqr = q3-q1
            low, high = q1-1.5*iqr, q3+1.5*iqr
            mask = (s<low)|(s>high)
            sample = s[mask].sort_values(ascending=False).head(10).tolist()
            text = (f"🚩 **IQR outliers** in **{col}** (<{_format_num(low)} or >{_format_num(high)}): "
                    f"{int(mask.sum())}\n\nExamples: {', '.join(_format_num(x) for x in sample)}")
        else:
            z = (s - s.mean()) / (s.std() or 1)
            mask = z.abs() > 3
            sample = s[mask].sort_values(ascending=False).head(10).tolist()
            text = (f"🚩 **Z-score outliers** in **{col}** (|z|>3): {int(mask.sum())}\n\n"
                    f"Examples: {', '.join(_format_num(x) for x in sample)}")
        return {"text": text}

    # ------------------ Time series + Forecast ------------------

    def _time_trend(self, df: pd.DataFrame, q: str) -> Dict[str, Any]:
        dt_col = _best_column(df, q, want="datetime")
        if not dt_col: return {"text": "ℹ️ I need a date/time column (e.g., 'Amount by month from Close Date')."}
        try:
            dates = pd.to_datetime(df[dt_col], errors="coerce")
        except Exception:
            return {"text": f"❌ Could not parse datetime from **{dt_col}**."}
        rule = "MS" if "month" in q else "W" if "week" in q else "D"
        num_col = _best_column(df, q, want="numeric")
        if num_col:
            series = pd.Series(df[num_col].astype(float), index=dates).resample(rule).sum().dropna()
            text = f"📈 **Trend of {num_col}** by {('month' if rule=='MS' else 'week' if rule=='W' else 'day')} (sum). Showing last 24."
        else:
            series = pd.Series(1, index=dates).resample(rule).sum().dropna()
            text = f"📈 **Trend (count)** by {('month' if rule=='MS' else 'week' if rule=='W' else 'day')}. Showing last 24."
        return {"text": text, "payload": {"type":"trend", "series": series.tail(24)}}

    def _forecast(self, df: pd.DataFrame, q: str) -> Dict[str, Any]:
        dt_col = _best_column(df, q, want="datetime")
        if not dt_col: return {"text": "ℹ️ For forecasting, mention a date column (e.g., 'forecast Amount by month from Close Date')."}
        target = _best_column(df, q, want="numeric")
        if not target: return {"text": "ℹ️ Tell me which numeric column to forecast (e.g., 'forecast Amount by month')."}
        rule = "MS" if "month" in q else "W" if "week" in q else "D"

        # Prepare series
        try:
            dates = pd.to_datetime(df[dt_col], errors="coerce")
            series = pd.Series(df[target].astype(float), index=dates).resample(rule).sum().dropna()
        except Exception as e:
            return {"text": f"❌ Could not prepare time series: {e}"}
        if len(series) < 6: return {"text": "ℹ️ Need at least 6 periods of data for forecasting."}
        history_tail = series.tail(24)

        # Prophet
        if _HAS_PROPHET:
            try:
                m = Prophet(seasonality_mode="additive", weekly_seasonality=True, yearly_seasonality=True)
                dfp = pd.DataFrame({"ds": series.index, "y": series.values})
                m.fit(dfp)
                future = m.make_future_dataframe(periods=6, freq=("MS" if rule == "MS" else "W" if rule == "W" else "D"))
                forecast = m.predict(future).tail(6)[["ds","yhat"]]
                fc = pd.Series(forecast["yhat"].values, index=pd.to_datetime(forecast["ds"]))
                return {"text": f"🔮 **Prophet forecast** for **{target}** (next 6)", "payload": {"type":"forecast", "history_tail": history_tail, "forecast": fc}}
            except Exception as e:
                logger.warning(f"Prophet failed, falling back. {e}")

        # Statsmodels Holt-Winters
        if _HAS_STATSMODELS:
            try:
                # auto seasonal period heuristic
                period = 12 if rule == "MS" else 52 if rule == "W" else 7
                model = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=period, initialization_method="estimated")
                fit = model.fit(optimized=True)
                fc = fit.forecast(6)
                return {"text": f"🔮 **Holt-Winters forecast** for **{target}** (next 6)", "payload": {"type":"forecast", "history_tail": history_tail, "forecast": fc}}
            except Exception as e:
                logger.warning(f"Holt-Winters failed, falling back. {e}")

        # MA(3) fallback
        window = 3
        fc_vals = []
        last = series.copy()
        for _ in range(6):
            fc = last.tail(window).mean()
            fc_vals.append(fc)
            next_idx = last.index[-1] + (pd.offsets.MonthBegin(1) if rule=="MS" else pd.offsets.Week(1) if rule=="W" else pd.Timedelta(days=1))
            last = pd.concat([last, pd.Series([fc], index=[next_idx])])
        fc_index = last.index[-6:]
        forecast = pd.Series(fc_vals, index=fc_index)
        lib_note = []
        if not _HAS_PROPHET: lib_note.append("Prophet not installed")
        if not _HAS_STATSMODELS: lib_note.append("statsmodels not installed")
        note = f" (fallback: {', '.join(lib_note)})" if lib_note else ""
        return {"text": f"🔮 **Moving-average forecast** for **{target}** (next 6){note}", "payload": {"type":"forecast", "history_tail": history_tail, "forecast": forecast}}

    # ------------------ Data Quality & Sales ------------------

    def _data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        rows = max(1, len(df))
        miss_cells = int(df.isna().sum().sum())
        miss_pct = miss_cells / (rows * max(1, len(df.columns)))
        miss_score = max(0, 1 - miss_pct*2)
        dup_score = max(0, 1 - (df.duplicated().sum()/rows))
        num_cols = df.select_dtypes(include=[np.number])
        bad_nums = int((num_cols <= 0).sum().sum()) if not num_cols.empty else 0
        num_score = max(0, 1 - (bad_nums / (rows * max(1, num_cols.shape[1] or 1))))
        dt_cols = [c for c in df.columns if "date" in _norm(c)]
        date_ok = 1.0
        for c in dt_cols[:3]:
            try: pd.to_datetime(df[c], errors="raise")
            except Exception: date_ok = 0.6; break
        type_score = date_ok
        overall = (0.4*miss_score + 0.2*dup_score + 0.2*num_score + 0.2*type_score)
        score = int(round(overall*100))
        grade = "Excellent" if score >= 85 else "Good" if score >= 70 else "Needs Improvement"
        components = {
            "missing_score": int(round(miss_score*100)),
            "duplicates_score": int(round(dup_score*100)),
            "numeric_sanity_score": int(round(num_score*100)),
            "type_consistency_score": int(round(type_score*100)),
        }
        details = [
            f"- Missing cells: {miss_cells:,} ({miss_pct*100:.2f}%)",
            f"- Duplicate rows: {int(df.duplicated().sum()):,}",
            f"- Non-positive numeric cells: {bad_nums:,}",
            f"- Date columns parse check: {'OK' if type_score==1.0 else 'Issues detected'}"
        ]
        return {"text": f"🛡️ **Data Trust Score:** {score}/100 ({grade})\n\n" + "\n".join(details),
                "payload": {"type":"data_quality", "overall_score": score, "components": components}}

    def _sales_pipeline(self, df: pd.DataFrame) -> Dict[str, Any]:
        cols = _detect_sales_columns(df)
        stage_col, amount_col, owner_col, date_col = cols["stage"], cols["amount"], cols["owner"], cols["date"]
        if not (stage_col and amount_col):
            return {"text": "ℹ️ I couldn’t find stage/amount columns. Try mentioning them (e.g., 'pipeline using Stage and Amount')."}
        work = df[[c for c in [stage_col, amount_col, owner_col, date_col] if c in df.columns]].copy()
        work = work.dropna(subset=[stage_col, amount_col])
        work[amount_col] = pd.to_numeric(work[amount_col], errors="coerce")
        work = work.dropna(subset=[amount_col])
        total_deals = len(work)
        total_value = float(work[amount_col].sum())
        closed_mask = _closed_mask(work[stage_col])
        closed_deals = int(closed_mask.sum())
        closed_value = float(work.loc[closed_mask, amount_col].sum())
        win_rate = (closed_deals / total_deals * 100) if total_deals else 0.0
        stage_summary = work.groupby(stage_col)[amount_col].agg(count="count", total_value="sum").sort_values("total_value", ascending=False).head(20).reset_index()
        payload = {"type":"sales_pipeline", "stage_summary": stage_summary}
        text = (f"💰 **Sales Pipeline**\n- Deals: **{total_deals:,}** • Pipeline: **${_format_num(total_value)}**\n"
                f"- Closed: **${_format_num(closed_value)}** • Win rate: **{win_rate:.1f}%**")
        if owner_col and owner_col in work.columns:
            owner_summary = work.groupby(owner_col)[amount_col].sum().sort_values(ascending=False).head(10).reset_index()
            payload["owner_summary"] = owner_summary
            text += "\n- Top owners table included."
        insights = []
        if win_rate < 15: insights.append(f"Win rate {win_rate:.1f}% < 15–20% norm → tighten qualification/late-stage conversions.")
        if total_value and (closed_value/total_value) < 0.25: insights.append("Most value still open → focus on late stages.")
        if not insights: insights.append("Healthy pipeline; monitor stage drop-offs.")
        text += "\n\n**Insights:**\n- " + "\n- ".join(insights)
        return {"text": text, "payload": payload}

    # ------------------ ML: Regression / Classification ------------------

    def _choose_features(self, df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
        y = df[target]
        # Candidate features: numeric + low-cardinality categoricals
        Xnum = df.select_dtypes(include=[np.number]).drop(columns=[c for c in [target] if c in df.columns], errors="ignore")
        cat = df.select_dtypes(include=["object","string","category"]).copy()
        # Encode low-cardinality cats
        for c in list(cat.columns):
            if c == target: 
                cat.drop(columns=[c], inplace=True)
                continue
            if cat[c].nunique(dropna=True) <= 50:
                dummies = pd.get_dummies(cat[c].astype(str).fillna("NA"), prefix=c, drop_first=True)
                Xnum = pd.concat([Xnum, dummies], axis=1)
        Xnum = Xnum.replace([np.inf,-np.inf], np.nan).fillna(0)
        return Xnum, y

    def _ml_regression(self, df: pd.DataFrame, q: str) -> Dict[str, Any]:
        if not _HAS_SKLEARN:
            return {"text": "ℹ️ scikit-learn not installed — cannot run regression. (pip install scikit-learn)"}
        target = _best_column(df, q, want="numeric")
        if not target: return {"text": "ℹ️ Tell me which numeric column to predict (e.g., 'regression to predict Amount')."}
        X, y = self._choose_features(df.dropna(subset=[target]), target)
        if X.empty or len(X) < 50: return {"text": "ℹ️ Not enough rows/features for a meaningful regression (need ~50+ rows)."}
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "R2": float(r2_score(y_test, preds)),
            "MAE": float(mean_absolute_error(y_test, preds)),
            "RMSE": float(mean_squared_error(y_test, preds, squared=False)),
            "Test_Size": int(len(y_test))
        }
        # Feature importances
        fi = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_}).sort_values("importance", ascending=False).head(30)
        text = (f"🤖 **Regression** predicting **{target}**\n"
                f"- R²: {metrics['R2']:.3f} • MAE: {metrics['MAE']:.2f} • RMSE: {metrics['RMSE']:.2f} (n={metrics['Test_Size']})\n"
                f"- Top features table included.")
        return {"text": text, "payload": {"type":"ml_regression", "metrics": metrics, "feature_importances": fi}}

    def _ml_classification(self, df: pd.DataFrame, q: str) -> Dict[str, Any]:
        if not _HAS_SKLEARN:
            return {"text": "ℹ️ scikit-learn not installed — cannot run classification. (pip install scikit-learn)"}
        # Pick a target categorical column (low-cardinality)
        target = None
        for c in df.columns:
            if df[c].dtype.kind in "OUS" or str(df[c].dtype).startswith("category"):
                if df[c].nunique(dropna=True) <= 20:
                    if _score_match(c, q) >= 0.25 or any(k in _norm(c) for k in ["churn","status","label","category","segment"]):
                        target = c; break
        if not target:
            return {"text": "ℹ️ Tell me which label to predict (e.g., 'classify churn' or mention a low-cardinality categorical column)."}
        data = df.dropna(subset=[target]).copy()
        y = data[target].astype(str)
        Xnum = data.select_dtypes(include=[np.number]).copy()
        cat = data.select_dtypes(include=["object","string","category"]).drop(columns=[target], errors="ignore")
        for c in list(cat.columns):
            if cat[c].nunique(dropna=True) <= 50:
                dummies = pd.get_dummies(cat[c].astype(str).fillna("NA"), prefix=c, drop_first=True)
                Xnum = pd.concat([Xnum, dummies], axis=1)
        if Xnum.empty or len(Xnum) < 50: return {"text": "ℹ️ Not enough rows/features for classification (need ~50+ rows)."}
        Xnum = Xnum.replace([np.inf,-np.inf], np.nan).fillna(0)
        X_train, X_test, y_train, y_test = train_test_split(Xnum, y, test_size=0.25, random_state=42, stratify=y)
        model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "Accuracy": float(accuracy_score(y_test, preds)),
            "F1_macro": float(f1_score(y_test, preds, average="macro")),
            "Test_Size": int(len(y_test))
        }
        # Importances
        fi = pd.DataFrame({"feature": Xnum.columns, "importance": getattr(model, "feature_importances_", np.zeros(len(Xnum.columns)))}).sort_values("importance", ascending=False).head(30)
        text = (f"🤖 **Classification** predicting **{target}**\n"
                f"- Accuracy: {metrics['Accuracy']:.3f} • F1 (macro): {metrics['F1_macro']:.3f} (n={metrics['Test_Size']})\n"
                f"- Top features table included.")
        return {"text": text, "payload": {"type":"ml_classification", "metrics": metrics, "feature_importances": fi}}

    # ------------------ Domain Plug-ins ------------------

    def _churn_report(self, df: pd.DataFrame, q: str) -> Dict[str, Any]:
        # Heuristics: need customer id, last activity/purchase date, optional status
        id_col = _best_column(df, "customer id") or _best_column(df, "account id") or _best_column(df, "user id") or _best_column(df, q)
        last_col = _best_column(df, "last purchase date") or _best_column(df, "last activity date") or _best_column(df, "last order date") or _best_column(df, "close date") or _best_column(df, "date")
        status_col = _best_column(df, "status") or _best_column(df, "churn") or _best_column(df, "active")
        amount_col = _best_column(df, "amount") or _best_column(df, "value") or _best_column(df, "revenue")
        if not (id_col and last_col):
            return {"text": "ℹ️ For churn, I need an ID and a last activity/purchase date column."}
        dates = pd.to_datetime(df[last_col], errors="coerce")
        cutoff = dates.max() - pd.Timedelta(days=90)  # churn if no activity in last 90d
        df2 = df.copy()
        df2["__churned"] = dates < cutoff
        churn_rate = float(df2["__churned"].mean()) if len(df2) else 0.0
        by_segment = None
        seg_col = _best_column(df, "segment") or _best_column(df, "plan") or _best_column(df, "tier")
        if seg_col:
            by_segment = df2.groupby(seg_col)["__churned"].mean().sort_values(ascending=False).reset_index().rename(columns={"__churned":"churn_rate"})
        summary = pd.DataFrame({
            "metric": ["customers", "churn_rate_90d", "active_customers"],
            "value": [len(df2), round(churn_rate, 4), int((~df2['__churned']).sum())]
        })
        if amount_col:
            revenue_loss = float(df2.loc[df2["__churned"], amount_col].sum())
            summary.loc[len(summary)] = ["estimated_revenue_at_risk", revenue_loss]
        text = f"📉 **Churn (90-day inactivity rule)**\n- Customers: {len(df2):,}\n- Churn rate: {churn_rate*100:.1f}%"
        return {"text": text, "payload": {"type":"churn", "summary": summary, "by_segment": by_segment} if by_segment is not None else {"type":"churn", "summary": summary}}

    def _cohort_analysis(self, df: pd.DataFrame, q: str) -> Dict[str, Any]:
        # Need user/customer id and a date (signup/order)
        id_col = _best_column(df, "customer id") or _best_column(df, "user id") or _best_column(df, "account id")
        date_col = _best_column(df, "signup date") or _best_column(df, "order date") or _best_column(df, "date")
        if not (id_col and date_col):
            return {"text": "ℹ️ For cohort analysis, I need an ID and a date column (signup/order)."}
        dates = pd.to_datetime(df[date_col], errors="coerce")
        df2 = df.copy()
        df2["cohort"] = dates.dt.to_period("M").astype(str)
        # Retention by cohort: users active each month
        df2["month"] = dates.dt.to_period("M").astype(str)
        cohort_pivot = (df2.groupby(["cohort","month"])[id_col]
                        .nunique().reset_index()
                        .pivot(index="cohort", columns="month", values=id_col).fillna(0).astype(int))
        # Normalize by cohort size (first column per row)
        base = cohort_pivot.apply(lambda r: r.iloc[0] if len(r) else 1, axis=1).replace(0,1)
        retention = (cohort_pivot.T / base).T.round(3)
        text = "👥 **Cohort retention** (rows=cohorts by first month, cols=subsequent months)"
        return {"text": text, "payload": {"type":"cohort", "cohort_table": retention.reset_index()}}

    def _funnel_by_step(self, df: pd.DataFrame, q: str) -> Dict[str, Any]:
        # Expect columns like step/stage and (optionally) user/session id
        step_col = _best_column(df, "step") or _best_column(df, "stage") or _best_column(df, "funnel step")
        id_col = _best_column(df, "session id") or _best_column(df, "user id") or _best_column(df, "deal id") or _best_column(df, "opportunity id")
        if not step_col:
            return {"text": "ℹ️ For a funnel, I need a step/stage column (and ideally a user/session id)."}
        if id_col:
            # Unique users per step
            counts = df.groupby(step_col)[id_col].nunique().reset_index().rename(columns={id_col:"unique_entities"})
        else:
            counts = df[step_col].value_counts().reset_index().rename(columns={"index": step_col, step_col: "count"}).sort_values(step_col, ascending=False)
            counts = counts.rename(columns={"count":"unique_entities"})
        counts["conversion_from_previous"] = (counts["unique_entities"] / counts["unique_entities"].shift(1)).round(3)
        text = "🪆 **Funnel by step** — unique entities per step and conversion vs previous."
        return {"text": text, "payload": {"type":"funnel", "funnel_table": counts}}

    def _rfm_scoring(self, df: pd.DataFrame, q: str) -> Dict[str, Any]:
        # Need customer id, date, amount
        id_col = _best_column(df, "customer id") or _best_column(df, "account id") or _best_column(df, "user id")
        date_col = _best_column(df, "order date") or _best_column(df, "purchase date") or _best_column(df, "date")
        amount_col = _best_column(df, "amount") or _best_column(df, "value") or _best_column(df, "revenue")
        if not (id_col and date_col and amount_col):
            return {"text": "ℹ️ For RFM, I need customer ID, date, and amount columns."}
        df2 = df[[id_col, date_col, amount_col]].copy()
        df2[date_col] = pd.to_datetime(df2[date_col], errors="coerce")
        df2[amount_col] = pd.to_numeric(df2[amount_col], errors="coerce")
        df2 = df2.dropna()
        snapshot_date = df2[date_col].max() + pd.Timedelta(days=1)
        r = (snapshot_date - df2.groupby(id_col)[date_col].max()).dt.days.rename("Recency")
        f = df2.groupby(id_col)[amount_col].count().rename("Frequency")
        m = df2.groupby(id_col)[amount_col].sum().rename("Monetary")
        rfm = pd.concat([r,f,m], axis=1)
        # quintile scores 1..5 (lower R is better)
        rfm["R_Score"] = pd.qcut(rfm["Recency"].rank(method="first"), 5, labels=[5,4,3,2,1]).astype(int)
        rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
        rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
        rfm["RFM_Total"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]
        rfm = rfm.sort_values("RFM_Total", ascending=False).reset_index().rename(columns={id_col: "Customer"})
        text = "💎 **RFM scoring** — 5=best for each R/F/M. Table includes top customers by RFM_Total."
        return {"text": text, "payload": {"type":"rfm", "rfm_table": rfm.head(200)}}

    # ------------------ Extremes & Average ------------------

    def _extreme(self, df: pd.DataFrame, q: str, kind: str) -> Dict[str, Any]:
        col = _best_column(df, q, want="numeric")
        if not col: return {"text": f"ℹ️ Which numeric column should I take the {kind} of?"}
        val = getattr(df[col], kind)()
        return {"text": f"{'📈' if kind=='max' else '📉'} **{kind.title()} {col}:** {_format_num(val)}"}

    def _average(self, df: pd.DataFrame, q: str) -> Dict[str, Any]:
        col = _best_column(df, q, want="numeric")
        if not col: return {"text": "ℹ️ Which numeric column should I average?"}
        val = float(df[col].mean())
        return {"text": f"📊 **Average {col}:** {_format_num(val)}"}

    # ------------------ Advisor (Reasoning) ------------------

    def _advisor(self, df: pd.DataFrame) -> Dict[str, Any]:
        rows, cols = len(df), len(df.columns)
        nums = df.select_dtypes(include=[np.number]).shape[1]
        cats = df.select_dtypes(include=["object","string","category"]).shape[1]
        dts  = df.select_dtypes(include=["datetime64[ns]","datetime64[ns, UTC]"]).shape[1]
        miss = int(df.isna().sum().sum())
        dup  = int(df.duplicated().sum())

        suggestions = []
        if miss > 0: suggestions.append("Run **data quality** to quantify missingness; consider imputations or drops.")
        if dup > 0: suggestions.append("Check **duplicates** and deduplicate if needed.")
        if nums >= 2: suggestions.append("Explore **correlations** to spot drivers and leakage.")
        if dts >= 1: suggestions.append("Plot **trend by month/week**; consider a **forecast** if seasonal.")
        if cats >= 1 and nums >= 1: suggestions.append("Use **groupby** (e.g., average Amount by Segment/Owner).")
        if nums >= 1: suggestions.append("Scan for **outliers** (z-score/IQR) before modeling.")
        # domain suggestions
        cols_map = _detect_sales_columns(df)
        if cols_map["stage"] and cols_map["amount"]:
            suggestions.append("Run **sales pipeline** to inspect funnel & win rate.")
        # ML suggestions
        if _HAS_SKLEARN and nums >= 1:
            suggestions.append("Try **regression** to predict a numeric KPI; **classification** for churn/status.")

        text = (
            "🧭 **Analyst Advisor**\n"
            f"- Shape: **{rows:,} × {cols}** • Numeric: {nums} • Categorical: {cats} • Datetime: {dts}\n"
            f"- Missing cells: **{miss:,}** • Duplicates: **{dup:,}**\n\n"
            "**Next best steps:**\n- " + "\n- ".join(suggestions[:8])
        )
        return {"text": text}

    def _help(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "text": (
                "🤖 **Analyst Workbench (no API)** — I can help with:\n\n"
                "• Advisor — *“what next”, “recommend”*\n"
                "• Columns/dtypes/missing/duplicates — *“show columns”, “data types”, “missing values”*\n"
                "• Summary/Correlations — *“summary”, “correlations”*\n"
                "• Value counts — *“value counts for Stage”*\n"
                "• Preview/sort/groupby — *“first 10 rows”, “top 10 by Amount”, “average Amount by Owner”*\n"
                "• Outliers — *“z-score outliers in Amount”, “IQR outliers in Price”*\n"
                "• Time trend — *“Amount by month from Close Date”*\n"
                "• Forecast — *“forecast Amount by month”* (Prophet → Holt-Winters → MA fallback)\n"
                "• Sales pipeline — *“sales pipeline overview”*\n"
                "• ML — *“regression to predict Amount”*, *“classification to predict churn”*\n"
                "• Churn/Cohort/Funnel/RFM — *“churn report”, “cohort analysis”, “funnel by step”, “RFM scoring”*\n\n"
                "Tip: prepend filters like *“where Stage = Proposal and Amount > 1000”*."
            )
        }
