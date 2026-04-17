import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://bana290-assignment2.netlify.app/"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

print(soup.prettify()[:3000])


# STAGE 2: CLEAN

import re

# Grab the table and all its rows
table = soup.find("table", class_="operations-table")
rows = table.find_all("tr")

# First row is the header
headers = [td.get_text(strip=True) for td in rows[0].find_all("td")]

# Loop through rows
data = []
for row in rows[1:]:
    cols = row.find_all("td")
    if len(cols) == 0:
        continue
    row_data = []
    for i, col in enumerate(cols):
        if i == 0:  # CLERK - grab just the name from <strong>
            strong = col.find("strong")
            row_data.append(strong.get_text(strip=True) if strong else col.get_text(strip=True))
        elif i in [1, 2]:  # CLERK_ID, QUEUE - grab text from <span>
            span = col.find("span")
            row_data.append(span.get_text(strip=True) if span else col.get_text(strip=True))
        else:
            row_data.append(col.get_text(strip=True))
    data.append(row_data)

# Put into DataFrame
df = pd.DataFrame(data, columns=headers)

# Fix treatment column

treatment_keywords = ["AI Extract", "Group A", "AI", "Treatment", "AI Tool", "Assist-On", "Prefill Enabled"]
control_keywords = ["Manual Entry", "Control", "Group B", "Manual", "No AI", "Typing Only"]

def map_treatment(value):
    if value == "None" or value is None:
        return None
    for keyword in treatment_keywords:
        if keyword.lower() in value.lower():
            return 1
    for keyword in control_keywords:
        if keyword.lower() in value.lower():
            return 0
    return None

df["TREATMENT"] = df["TREATMENT"].apply(map_treatment)

# Fix columns
def extract_number(value):
    cleaned = re.sub(r"[^\d.]", "", str(value))
    try:
        return float(cleaned)
    except:
        return None

numeric_cols = [
    "YEARS_EXPERIENCE", "BASELINE_TASKS_PER_HOUR",
    "BASELINE_ERROR_RATE", "TRAINING_SCORE",
    "TASKS_COMPLETED", "ERROR_RATE"
]
for col in numeric_cols:
    df[col] = df[col].apply(extract_number)

# Fix timestamps
def parse_timestamp(val):
    formats = [
        "%b %d, %Y %H:%M",       # Feb 18, 2026 07:56
        "%d-%b-%Y %I:%M %p",     # 21-Feb-2026 08:19 AM
        "%Y-%m-%d %H:%M",        # 2026-02-21 08:11
        "%m/%d/%Y %I:%M %p",     # 02/21/2026 07:55 AM
        "%Y-%m-%dT%H:%M",        # 2026-02-21T08:11
        "%m/%d/%Y %H:%M",        # 02/21/2026 07:55
        "%d-%b-%Y %H:%M",        # 21-Feb-2026 08:19
    ]
    for fmt in formats:
        try:
            return pd.to_datetime(val, format=fmt)
        except:
            continue
    return pd.NaT

df["SHIFT_START"] = df["SHIFT_START"].apply(parse_timestamp)
df["SHIFT_END"] = df["SHIFT_END"].apply(parse_timestamp)
df["SHIFT_DURATION_HRS"] = (df["SHIFT_END"] - df["SHIFT_START"]).dt.total_seconds() / 3600

# Clean
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

for col in ["CLERK", "QUEUE", "SITE", "SHIFT"]:
    if col in df.columns:
        df[col] = df[col].str.title()

# Cap TRAINING_SCORE to realistic range (0-100)
df["TRAINING_SCORE"] = df["TRAINING_SCORE"].apply(lambda x: x if x <= 100 else None)

# Remove duplicates and bad rows
df = df.drop_duplicates()
df = df[df["SHIFT_DURATION_HRS"] > 0]
df = df.dropna()

# Print
print(f"\nShape: {df.shape}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nTreatment counts:\n{df['TREATMENT'].value_counts()}")


# STAGE 3: ANALYZE

# Split into treatment and control groups
treatment = df[df["TREATMENT"] == 1.0]
control = df[df["TREATMENT"] == 0.0]

# Balance Test
# Check if randomization created balanced groups by comparing baseline characteristics
balance_cols = ["YEARS_EXPERIENCE", "BASELINE_TASKS_PER_HOUR", "BASELINE_ERROR_RATE", "TRAINING_SCORE"]

print("=" * 55)
print("BALANCE TEST - Baseline Characteristics by Group")
print("=" * 55)
print(f"{'Characteristic':<28} {'Treatment':>10} {'Control':>10}")
print("-" * 55)
for col in balance_cols:
    t_mean = treatment[col].mean()
    c_mean = control[col].mean()
    print(f"{col:<28} {t_mean:>10.2f} {c_mean:>10.2f}")
print("=" * 55)
print(f"{'Group Size':<28} {len(treatment):>10} {len(control):>10}")

from scipy import stats

# T-tests
print("\n")
print("=" * 65)
print("IGNORABILITY TEST - T-Tests on Baseline Characteristics")
print("=" * 65)
print(f"{'Characteristic':<28} {'T-Stat':>10} {'P-Value':>10} {'Balanced?':>10}")
print("-" * 65)

for col in balance_cols:
    t_stat, p_value = stats.ttest_ind(treatment[col], control[col])
    balanced = "Yes" if p_value > 0.05 else "No"
    print(f"{col:<28} {t_stat:>10.3f} {p_value:>10.3f} {balanced:>10}")

print("=" * 65)
print("* p > 0.05 means no significant difference = groups are balanced")

#SUTVA Justification
print("\n")
print("=" * 65)
print("SUTVA - Stable Unit Treatment Value Assumption")
print("=" * 65)
print("""
Justification for no spillover effects:
- Each clerk works independently on their own computer and AI tool access, so one clerk's use of AI does not affect another's performance.
- Treatment and control clerks pull from different queues, minimizing any interaction or influence between them.
- Clerks do not share information about their tasks or performance during the audit week, reducing the chance of behavior changes based on others' conditions.
- AI assignment was random so there is no way a control clerk would have access to AI tools.
""")

#ATE Estimation
print("=" * 55)
print("ATE ESTIMATION - Average Treatment Effect")
print("=" * 55)

# ATE on Productivity (TASKS_COMPLETED)
avg_tasks_treatment = treatment["TASKS_COMPLETED"].mean()
avg_tasks_control = control["TASKS_COMPLETED"].mean()
ate_tasks = avg_tasks_treatment - avg_tasks_control

# ATE on Quality (ERROR_RATE)
avg_error_treatment = treatment["ERROR_RATE"].mean()
avg_error_control = control["ERROR_RATE"].mean()
ate_error = avg_error_treatment - avg_error_control

# Print 
print(f"\nPRODUCTIVITY (Tasks Completed)")
print(f"  Treatment group avg:  {avg_tasks_treatment:.2f} tasks")
print(f"  Control group avg:    {avg_tasks_control:.2f} tasks")
print(f"  ATE:                  {ate_tasks:+.2f} tasks")

print(f"\nQUALITY (Error Rate)")
print(f"  Treatment group avg:  {avg_error_treatment:.2f}%")
print(f"  Control group avg:    {avg_error_control:.2f}%")
print(f"  ATE:                  {ate_error:+.2f}%")
print("=" * 55)
print("* Positive ATE on tasks = AI increased productivity")
print("* Negative ATE on errors = AI improved quality")