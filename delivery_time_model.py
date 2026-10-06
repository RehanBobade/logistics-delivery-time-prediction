"""Last-mile delivery time prediction and optimization pipeline."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_validate, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from scipy.optimize import linear_sum_assignment
from scipy.stats import randint, uniform

RNG = np.random.default_rng(42)
import os
DATA, FIG, RES = "data/", "figures/", "results/"
for d in (DATA, FIG, RES):
    os.makedirs(d, exist_ok=True)
results = {}

# ---------------------------------------------------------------- 1. DATA
def simulate(n=6000):
    df = pd.DataFrame({
        "distance_km": RNG.gamma(3.0, 3.0, n).clip(0.5, 45),
        "num_stops": RNG.integers(1, 26, n),
        "package_weight_kg": RNG.lognormal(1.2, 0.7, n).clip(0.2, 60),
        "hour_of_day": RNG.integers(7, 21, n),
        "day_of_week": RNG.integers(0, 7, n),
        "weather": RNG.choice(["clear", "rain", "storm"], n, p=[0.70, 0.24, 0.06]),
        "vehicle_type": RNG.choice(["bike", "van", "truck"], n, p=[0.30, 0.50, 0.20]),
        "driver_experience_yrs": RNG.uniform(0, 12, n),
        "warehouse": RNG.choice(["WH_North", "WH_South", "WH_East"], n),
    })
    peak = (df.hour_of_day.between(8, 10) | df.hour_of_day.between(17, 19)).astype(int)
    df["traffic_index"] = (3 + 4 * peak + 0.8 * (df.day_of_week < 5)
                           + RNG.normal(0, 0.9, n)).clip(1, 10)

    speed = df.vehicle_type.map({"bike": 17, "van": 28, "truck": 24}).astype(float)
    speed *= (1 - 0.055 * (df.traffic_index - 1))
    speed *= df.weather.map({"clear": 1.0, "rain": 0.85, "storm": 0.62})
    speed *= (1 + 0.012 * df.driver_experience_yrs)
    speed = speed.clip(6, None)
    drive = df.distance_km / speed * 60
    service = df.num_stops * (2.2 + 0.03 * df.package_weight_kg) \
        * df.vehicle_type.map({"bike": 0.9, "van": 1.0, "truck": 1.25})
    wh_delay = df.warehouse.map({"WH_North": 3, "WH_South": 6, "WH_East": 4.5})
    noise = RNG.normal(0, 1, n) * (3 + 0.08 * (drive + service))
    df["delivery_time_min"] = (drive + service + wh_delay + noise).clip(5, None)
    # introduce a little missing data to demonstrate cleaning
    idx = RNG.choice(n, 120, replace=False)
    df.loc[idx, "traffic_index"] = np.nan
    return df

df = simulate()
df.to_csv(DATA + "delivery_data.csv", index=False)
results["n_rows"] = len(df)
results["target_mean"] = df.delivery_time_min.mean()
results["target_std"] = df.delivery_time_min.std()
results["target_min"] = df.delivery_time_min.min()
results["target_max"] = df.delivery_time_min.max()
results["missing_traffic"] = int(df.traffic_index.isna().sum())

# ---------------------------------------------------------------- 2. PREP
from sklearn.impute import SimpleImputer
df["is_peak"] = (df.hour_of_day.between(8, 10) | df.hour_of_day.between(17, 19)).astype(int)
df["is_weekend"] = (df.day_of_week >= 5).astype(int)
df["stops_per_km"] = df.num_stops / df.distance_km

target = "delivery_time_min"
num_cols = ["distance_km", "num_stops", "package_weight_kg", "hour_of_day",
            "traffic_index", "driver_experience_yrs", "is_peak", "is_weekend", "stops_per_km"]
cat_cols = ["weather", "vehicle_type", "warehouse"]
X, y = df[num_cols + cat_cols + ["day_of_week"]].drop(columns=["day_of_week"]), df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
results["n_train"], results["n_test"] = len(X_train), len(X_test)

def prep(scale):
    num = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num.append(("sc", StandardScaler()))
    return ColumnTransformer([("num", Pipeline(num), num_cols),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])

models = {
    "Linear Regression": Pipeline([("prep", prep(True)), ("m", LinearRegression())]),
    "Decision Tree": Pipeline([("prep", prep(False)),
                               ("m", DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=42))]),
    "Random Forest": Pipeline([("prep", prep(False)),
                               ("m", RandomForestRegressor(n_estimators=100, min_samples_leaf=3, n_jobs=2, random_state=42))]),
    "Gradient Boosting": Pipeline([("prep", prep(False)),
                                   ("m", GradientBoostingRegressor(random_state=42))]),
}

# ---------------------------------------------------------------- 3. CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
cv_rows = []
for name, pipe in models.items():
    cv = cross_validate(pipe, X_train, y_train, cv=kf, scoring=scoring, n_jobs=2)
    cv_rows.append({"Model": name,
                    "CV RMSE": -cv["test_rmse"].mean(), "CV RMSE std": cv["test_rmse"].std(),
                    "CV MAE": -cv["test_mae"].mean(), "CV R2": cv["test_r2"].mean()})
cv_df = pd.DataFrame(cv_rows)
print(cv_df.round(3))

# ---------------------------------------------------------------- 4. TUNING
param_dist = {
    "m__n_estimators": randint(100, 300),
    "m__learning_rate": uniform(0.02, 0.13),
    "m__max_depth": randint(2, 6),
    "m__min_samples_leaf": randint(5, 40),
    "m__subsample": uniform(0.6, 0.4),
}
search = RandomizedSearchCV(models["Gradient Boosting"], param_dist, n_iter=10, cv=kf,
                            scoring="neg_root_mean_squared_error", random_state=42, n_jobs=2)
search.fit(X_train, y_train)
best = search.best_estimator_
results["best_params"] = {k.replace("m__", ""): (round(v, 4) if isinstance(v, float) else int(v))
                          for k, v in search.best_params_.items()}
results["tuned_cv_rmse"] = -search.best_score_
print(results["best_params"], results["tuned_cv_rmse"])

# ---------------------------------------------------------------- 5. TEST
test_rows = []
fitted = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    fitted[name] = pipe
fitted["Tuned Gradient Boosting"] = best
for name, pipe in fitted.items():
    p = pipe.predict(X_test)
    test_rows.append({"Model": name,
                      "RMSE": np.sqrt(mean_squared_error(y_test, p)),
                      "MAE": mean_absolute_error(y_test, p),
                      "R2": r2_score(y_test, p)})
test_df = pd.DataFrame(test_rows)
print(test_df.round(3))

pred = best.predict(X_test)
resid = y_test - pred
results["resid_mean"], results["resid_std"] = resid.mean(), resid.std()
results["within_10min"] = float((resid.abs() <= 10).mean())
results["mape"] = float((resid.abs() / y_test).mean())
# 90% prediction interval via residual quantiles (from CV-free held-out train residual proxy)
q_lo, q_hi = np.quantile(resid, [0.05, 0.95])
results["pi_lo"], results["pi_hi"] = q_lo, q_hi

# ---------------------------------------------------------------- 6. IMPORTANCE
pi = permutation_importance(best, X_test, y_test, n_repeats=10, random_state=42,
                            scoring="neg_root_mean_squared_error", n_jobs=2)
imp = pd.Series(pi.importances_mean, index=X_test.columns).sort_values(ascending=False)
imp_pct = imp / imp.sum() * 100
results["importance"] = imp_pct.round(1).to_dict()
print(imp_pct.round(1))

# ---------------------------------------------------------------- 7. FIGURES
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREY = "#1F4E79", "#E07B39", "#8A8A8A"

fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.hist(df[target], bins=50, color=BLUE, edgecolor="white")
ax.set_xlabel("Delivery time (minutes)"); ax.set_ylabel("Number of deliveries")
ax.set_title("Distribution of the target variable")
fig.tight_layout(); fig.savefig(FIG + "fig_target.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.2, 3.6))
names = test_df.Model.tolist()
cols = [GREY, GREY, GREY, GREY, ORANGE]
ax.barh(names, test_df.RMSE, color=cols)
for i, v in enumerate(test_df.RMSE):
    ax.text(v + 0.3, i, f"{v:.1f}", va="center")
ax.invert_yaxis(); ax.set_xlabel("Test RMSE (minutes), lower is better")
ax.set_title("Model comparison on held-out test set")
fig.tight_layout(); fig.savefig(FIG + "fig_models.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.2, 4.6))
ax.scatter(y_test, pred, s=8, alpha=0.4, color=BLUE)
lim = [0, max(y_test.max(), pred.max()) * 1.02]
ax.plot(lim, lim, color=ORANGE, lw=1.5, label="Perfect prediction")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Actual (min)"); ax.set_ylabel("Predicted (min)")
ax.set_title("Actual vs predicted (test set)"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG + "fig_actual_pred.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.hist(resid, bins=45, color=BLUE, edgecolor="white")
ax.axvline(0, color=ORANGE, lw=1.5)
ax.set_xlabel("Residual = actual - predicted (min)"); ax.set_ylabel("Count")
ax.set_title("Residual distribution (test set)")
fig.tight_layout(); fig.savefig(FIG + "fig_resid.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.2, 4.0))
top = imp_pct.head(9)[::-1]
ax.barh(top.index, top.values, color=BLUE)
ax.set_xlabel("Share of total permutation importance (%)")
ax.set_title("What drives delivery time?")
fig.tight_layout(); fig.savefig(FIG + "fig_importance.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- 8. OPTIMIZATION A: assignment
# 30 routes (jobs) to be served by 30 drivers/vehicles. Model predicts time for every pair.
n_jobs = 30
jobs = X_test.sample(n_jobs, random_state=7).reset_index(drop=True)
fleet = pd.DataFrame({
    "vehicle_type": RNG.choice(["bike", "van", "truck"], n_jobs, p=[0.3, 0.5, 0.2]),
    "driver_experience_yrs": RNG.uniform(0, 12, n_jobs).round(1),
})
cost = np.zeros((n_jobs, n_jobs))
for i in range(n_jobs):
    batch = jobs.iloc[[i] * n_jobs].copy().reset_index(drop=True)
    batch["vehicle_type"] = fleet.vehicle_type.values
    batch["driver_experience_yrs"] = fleet.driver_experience_yrs.values
    cost[i] = best.predict(batch)
r, c = linear_sum_assignment(cost)
opt_total = cost[r, c].sum()
rand_totals = [cost[np.arange(n_jobs), RNG.permutation(n_jobs)].sum() for _ in range(5000)]
results["assign_opt"] = opt_total / 60
results["assign_rand_mean"] = float(np.mean(rand_totals)) / 60
results["assign_saving_pct"] = float((1 - opt_total / np.mean(rand_totals)) * 100)
results["assign_saving_hours"] = results["assign_rand_mean"] - results["assign_opt"]

# ---------------------------------------------------------------- 9. OPTIMIZATION B: departure window
flex = X_test.sample(400, random_state=11).reset_index(drop=True)
base_pred = best.predict(flex)
best_pred = base_pred.copy()
best_hour = flex.hour_of_day.values.copy()
for h in range(7, 21):
    t = flex.copy()
    t["hour_of_day"] = h
    t["is_peak"] = int(8 <= h <= 10 or 17 <= h <= 19)
    # traffic is lower off-peak: re-derive expected traffic index for the candidate hour
    t["traffic_index"] = 3 + 4 * t["is_peak"] + 0.8 * (1 - flex.is_weekend)
    p = best.predict(t)
    # only allow shifting within +/- 2 hours of the planned departure
    ok = np.abs(h - flex.hour_of_day.values) <= 2
    better = ok & (p < best_pred)
    best_pred[better] = p[better]
    best_hour[better] = h
results["shift_saving_pct"] = float((1 - best_pred.sum() / base_pred.sum()) * 100)
results["shift_saving_min_per_job"] = float((base_pred - best_pred).mean())
results["shift_share_moved"] = float((best_hour != flex.hour_of_day.values).mean())

# ---------------------------------------------------------------- 10. SLA risk flagging
promise = 1.15 * pred  # naive promise = point forecast + 15%
sla_minutes = 120
upper = pred + q_hi
flag = upper > sla_minutes
results["sla_flagged_share"] = float(flag.mean())
actual_breach = y_test.values > sla_minutes
results["sla_actual_breach_share"] = float(actual_breach.mean())
if actual_breach.sum() > 0:
    results["sla_recall"] = float((flag & actual_breach).sum() / actual_breach.sum())
    results["sla_precision"] = float((flag & actual_breach).sum() / max(flag.sum(), 1))

cv_df.to_csv(RES + "cv.csv", index=False)
test_df.to_csv(RES + "test.csv", index=False)
json.dump(results, open(RES + "results.json", "w"), indent=2, default=float)
print(json.dumps(results, indent=2, default=float))
