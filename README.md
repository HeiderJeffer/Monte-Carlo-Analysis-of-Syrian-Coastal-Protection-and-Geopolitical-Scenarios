### **Monte Carlo Analysis of Syrian Coastal Protection and Geopolitical Scenarios**

*Developed using Python by Heider Jeffer*
---

### **Introduction:**

This project uses a Monte Carlo simulation framework to explore potential geopolitical outcomes along the Syrian coast. By modeling key structural drivers—such as Russian and Iranian influence, Turkish expansion, Israeli involvement, jihadist pressure, and international appetite for safe zones—the analysis estimates the probability of five possible scenarios:

1. Regime Holds Coast
2. Iran Expands Influence
3. Israeli Indirect Role
4. Turkish Expansion
5. International Safe Zones

Each simulation samples these drivers randomly according to subjective Beta distributions, reflecting prior assumptions about their relative strength. Scenario outcomes are calculated as linear combinations of these drivers, with small random noise added to reflect uncertainty.

The resulting simulation provides probabilistic insights into which scenarios are most likely under the modeled assumptions and allows for **sensitivity analysis** to assess how changes in each driver affect the likelihood of different outcomes. This approach does **not rely on real-time data** and is intended as a **toy analytical model** for exploratory purposes.

---



## **Purpose of the Code**

This code is a **toy Monte Carlo simulation** to explore possible geopolitical outcomes for **Syria’s coastal region**. It:

1. Simulates uncertainty in **geopolitical drivers** (like Russian presence, Iranian influence, etc.).
2. Computes **scenario “scores”** based on these drivers.
3. Determines which scenario is most likely for each simulation.
4. Aggregates results to estimate **probabilities** for each scenario.
5. Performs **sensitivity analysis** to see how changes in each driver affect scenario probabilities.

> It is a **model based on assumptions**, not real-time data.

---

## **Libraries and Setup**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(42)
```

* `numpy`: for numerical operations and random sampling
* `pandas`: for data handling and tables
* `matplotlib`: for plotting
* `Path`: for file system paths
* `np.random.seed(42)`: ensures reproducible random results

---

## **Simulation Setup**

```python
N = 100_000
scenarios = [
    "Regime Holds Coast",
    "Iran Expands Influence",
    "Israeli Indirect Role",
    "Turkish Expansion",
    "International Safe Zones"
]
```

* **N = 100,000** → number of Monte Carlo simulations
* **scenarios** → 5 possible outcomes for Syria

---

## **Sampling Drivers (Uncertainties)**

```python
R = np.random.beta(4, 2, N)   # Russian commitment
I = np.random.beta(3, 2, N)   # Iranian capacity
IS = np.random.beta(2, 3, N)  # Israeli risk tolerance
T = np.random.beta(2, 3, N)   # Turkish intent
J = np.random.beta(2, 2, N)   # Jihadist pressure
U = np.random.beta(1, 4, N)   # International appetite
```

* Each driver is sampled from a **Beta distribution**, giving a **probability-like value** between 0 and 1.
* Beta distribution parameters encode prior beliefs:

  * Example: `Beta(4,2)` → skewed toward high Russian commitment
  * `Beta(1,4)` → skewed toward low international appetite

**Key idea:** each simulation generates a **randomized “world”** with different driver strengths.

---

## **Scenario Scoring**

```python
noise = np.random.normal(0, 0.05, (N, 5))
```

* Adds small Gaussian noise to prevent ties and simulate unpredictability.

**Scenario formulas:** (linear combinations of drivers)

```python
S_regime = 0.5*R + 0.2*I - 0.2*J - 0.1*T
S_iran    = 0.6*I + 0.3*(1 - R) + 0.1*J
S_israel  = 0.4*IS + 0.2*J + 0.2*I - 0.2*T - 0.2*R
S_turkey  = 0.5*T + 0.2*(1 - R) - 0.2*I - 0.1*J
S_safe    = 0.4*U + 0.2*(1 - R) + 0.1*J - 0.1*I
```

* Each scenario has **weights** for each driver reflecting its importance:

  * `S_regime` → benefits from Russian and Iranian presence, reduced by jihadists and Turkey
  * `S_iran` → depends on Iranian strength and Russian weakness
  * And so on…

```python
scores = np.vstack([S_regime, S_iran, S_israel, S_turkey, S_safe]).T + noise
```

* Stack all scenario scores for each simulation.
* Shape: `(N simulations x 5 scenarios)`

---

## **Determine the “Winning” Scenario**

```python
winners_idx = np.argmax(scores, axis=1)
winners = np.array(scenarios)[winners_idx]
```

* For each simulation, pick the scenario with **highest score**.

---

## **Aggregate Probabilities**

```python
probs = pd.Series(winners).value_counts(normalize=True).reindex(scenarios).fillna(0)
```

* Counts how many times each scenario won.
* `normalize=True` → converts counts to probabilities
* `reindex(scenarios).fillna(0)` → ensures all scenarios are included

**Output:** `probs` is the **estimated probability** for each scenario.

---

## **Save Results and Plot**

```python
probs_df.to_csv(csv_path, index=False)

plt.figure(figsize=(10, 6))
plt.barh(probs.index, probs.values)
plt.xlabel("Estimated Probability")
plt.title("Simulated Probabilities of Future Scenarios")
plt.gca().invert_yaxis()
plt.savefig(prob_png)
plt.show()
```

* Saves probabilities as CSV and creates a **horizontal bar chart**.

---

## **Sensitivity Analysis**

```python
def recompute_with_scaling(var_name):
    ...
```

* Tests effect of **+10% change in each driver**.
* Steps:

  1. Increase driver by 10% (clip to 0–1)
  2. Recompute scenario scores
  3. Recompute probabilities
  4. Return **Δ probability** relative to baseline

```python
drivers = ["R","I","IS","T","J","U"]
sens = {d: recompute_with_scaling(d) - probs for d in drivers}
sens_df = pd.DataFrame(sens).T
```

* Produces a **driver × scenario table** showing **how much each driver shifts scenario probability**.

---

## **Plot Sensitivity**

```python
for scen in scenarios:
    plt.bar(drivers, sens_df[scen].values)
```

* One bar chart per scenario
* Shows **absolute change in probability** if each driver increases by 10%

---

## **Save README / Assumptions**

```python
readme_text = """Toy Monte Carlo model (no internet access)
...
"""
with open(readme_path, "w") as f:
    f.write(readme_text)
```

* Documents:

  * Scenario definitions
  * Driver distributions
  * Sensitivity method

---

## **Display Tables in Notebook**

```python
display(probs_df)
display(sens_df.round(4))
```

* Shows **probabilities** and **sensitivity table** in the notebook.

---

## **Summary of the Workflow**

1. **Define scenarios and drivers**
2. **Randomly sample drivers** for each simulation
3. **Compute scenario scores** (linear combination + noise)
4. **Select winning scenario** per simulation
5. **Aggregate results** → probabilities
6. **Perform sensitivity analysis** → how +10% in each driver affects probabilities
7. **Save outputs** (CSV, PNG, README)
8. **Display tables** in the notebook

---

### **Key Concepts Illustrated**

* **Monte Carlo simulation**: estimate outcome probabilities with randomness
* **Scenario scoring**: encode assumptions as linear combinations of drivers
* **Probabilistic reasoning**: probabilities represent **likelihoods under assumptions**
* **Sensitivity analysis**: identifies which drivers most influence outcomes

---
