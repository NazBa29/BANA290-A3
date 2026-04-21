# ============================================================
# BANA 290 Assignment 3 — DID Analysis of AI Training Subsidy
# Author: Ilnaz Bagheri (NazBa29)
# ============================================================

# ============================================================
# PHASE 1: SCRAPE
# ============================================================

# Copilot prompt: Import requests, BeautifulSoup, pandas, and re.
# We will scrape the Rust Belt Revival Labor Archive, which hosts
# four district labor briefs at bana290-assignment3.netlify.app.
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from urllib.parse import urljoin

BASE_URL = "https://bana290-assignment3.netlify.app"

# Copilot prompt: Fetch the index page, parse with BeautifulSoup,
# and find every <a> tag whose href points to an individual brief.
# Build absolute URLs for each brief.
print("Fetching index page...")
index_response = requests.get(BASE_URL)
index_soup = BeautifulSoup(index_response.text, "html.parser")

brief_links = []
for a_tag in index_soup.find_all("a", href=True):
    href = a_tag["href"]
    # Only keep links that look like internal brief pages
    if href.startswith("/") and href != "/" and "brief" in href.lower() or \
       (href.startswith("/") and any(x in href.lower() for x in ["ohio", "pennsylvania", "corridor", "benchmark"])):
        full_url = urljoin(BASE_URL, href)
        if full_url not in brief_links:
            brief_links.append(full_url)

print(f"Found {len(brief_links)} brief links:")
for link in brief_links:
    print(f"  - {link}")

# Copilot prompt: Loop through each brief URL, fetch the page,
# extract the employment table, and also pull out the brief's
# title and treatment status (TREATED vs CONTROL) from the page.
all_tables = []

for url in brief_links:
    print(f"\nScraping: {url}")
    page_response = requests.get(url)
    page_soup = BeautifulSoup(page_response.text, "html.parser")

    # Get the brief title (the main H1)
    title_tag = page_soup.find("h1")
    brief_title = title_tag.get_text(strip=True) if title_tag else "Unknown"

    # Get treatment status from the "TREATED REGION BRIEF" or "CONTROL REGION BRIEF" label
    page_text = page_soup.get_text()
    if "TREATED REGION BRIEF" in page_text:
        treatment_label = "TREATED"
    elif "CONTROL REGION BRIEF" in page_text:
        treatment_label = "CONTROL"
    else:
        treatment_label = "UNKNOWN"

    # Read all HTML tables on the page into pandas
    tables_on_page = pd.read_html(url)
    for tbl in tables_on_page:
        tbl["BRIEF_TITLE"] = brief_title
        tbl["TREATMENT_LABEL"] = treatment_label
        tbl["SOURCE_URL"] = url
        all_tables.append(tbl)

# Copilot prompt: Concatenate all per-brief tables into one DataFrame.
raw_df = pd.concat(all_tables, ignore_index=True)
print(f"\nRaw combined table shape: {raw_df.shape}")
print("Columns:", list(raw_df.columns))
print(raw_df.head())

# Save raw scraped data for inspection
raw_df.to_csv("raw_scraped.csv", index=False)
print("\nSaved raw_scraped.csv")



# ============================================================
# PHASE 2: CLEAN
# ============================================================

# Copilot prompt: Clean the raw scraped data. The scraped tables have
# messy headers, mixed row types, and the first row of each table is
# actually a header ("REGION", "STATE_GROUP", etc). We need to:
# (1) rebuild proper column names,
# (2) drop header rows that ended up in the data,
# (3) reshape from wide to long (one row per county-year),
# (4) convert messy employment strings like "30.9k", "~38.0k",
#     "31.4 thousand", "32,055 jobs" into clean integers,
# (5) standardize county names,
# (6) create TREATED and POST_POLICY dummy variables (policy year = 2022).

import re
import numpy as np

# Reload from CSV for a clean start of Phase 2
raw_df = pd.read_csv("raw_scraped.csv")

# The scraped tables use the first row of each sub-table as column labels.
# We need to rebuild proper column names. Based on inspection, the columns are:
# REGION, STATE_GROUP, PROGRAM_STATUS, ANCHOR_INDUSTRY, 2018, 2019, ..., 2025, PORTAL_NOTE
proper_columns = [
    "REGION", "STATE_GROUP", "PROGRAM_STATUS", "ANCHOR_INDUSTRY",
    "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025",
    "PORTAL_NOTE", "BRIEF_TITLE", "TREATMENT_LABEL", "SOURCE_URL"
]
raw_df.columns = proper_columns

# Drop rows where REGION is literally the word "REGION" (those are header rows)
clean_df = raw_df[raw_df["REGION"] != "REGION"].copy()
clean_df = clean_df.reset_index(drop=True)

print("After removing header rows:")
print(clean_df[["REGION", "TREATMENT_LABEL", "2018", "2022", "2025"]])
print(f"Shape: {clean_df.shape}")

# Copilot prompt: Standardize county names. The raw names have many
# inconsistent formats like "Lucas Cnty, Ohio", "Stark County / OH",
# "Mahoning County - Ohio", "Trumbull Cnty., OH", "Erie County (PA)",
# "Mercer Cnty, Pennsylvania", "Lawrence County - PA", "Beaver Cnty., PA".
# Convert each to a clean "<Countyname> County, <Statename>" format.

def standardize_county(name):
    # Extract just the county part (before any comma, slash, dash, or parenthesis)
    name = str(name).strip()
    # Remove state info at the end in various formats
    name = re.split(r"[,/\-()]", name)[0].strip()
    # Remove "Cnty", "Cnty.", "County" suffix
    name = re.sub(r"\b(Cnty\.?|County)\b", "", name, flags=re.IGNORECASE).strip()
    name = name.rstrip(".").strip()
    return name

def detect_state(name):
    name_lower = str(name).lower()
    if "ohio" in name_lower or " oh" in name_lower or "/oh" in name_lower:
        return "Ohio"
    if "pennsylvania" in name_lower or " pa" in name_lower or "(pa)" in name_lower:
        return "Pennsylvania"
    return "Unknown"

clean_df["COUNTY"] = clean_df["REGION"].apply(standardize_county)
clean_df["STATE"] = clean_df["REGION"].apply(detect_state)
clean_df["COUNTY_STATE"] = clean_df["COUNTY"] + " County, " + clean_df["STATE"]

print("\nStandardized county names:")
print(clean_df[["REGION", "COUNTY_STATE"]].to_string())

# Copilot prompt: Reshape the data from wide to long format. Each row
# should become one county-year observation.
year_columns = ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]
long_df = clean_df.melt(
    id_vars=["COUNTY_STATE", "STATE", "TREATMENT_LABEL", "BRIEF_TITLE"],
    value_vars=year_columns,
    var_name="YEAR",
    value_name="EMPLOYMENT_RAW"
)
long_df["YEAR"] = long_df["YEAR"].astype(int)

print(f"\nLong format shape: {long_df.shape}")  # Should be 8 counties * 8 years = 64 rows
print(long_df.head(10))

# Copilot prompt: Convert messy employment strings to integers. Handle
# formats: "32,055 jobs", "30.9k", "~38.0k", "31.4 thousand", "34.6 K",
# plain numbers like "32620". Strip approximate markers (~), remove unit
# words (jobs, thousand), handle k/K suffix (multiply by 1000).

def parse_employment(raw):
    s = str(raw).strip().lower()
    # Remove approximate marker
    s = s.replace("~", "")
    # Remove unit words
    s = s.replace("jobs", "").replace("thousand", "k")
    # Remove commas
    s = s.replace(",", "")
    # Remove extra whitespace
    s = s.strip()

    # Check for k/K suffix
    if s.endswith("k"):
        num = float(s[:-1].strip())
        return int(round(num * 1000))
    else:
        # Plain number
        try:
            return int(round(float(s)))
        except ValueError:
            return np.nan

long_df["EMPLOYMENT"] = long_df["EMPLOYMENT_RAW"].apply(parse_employment)

print("\nEmployment values after cleaning:")
print(long_df[["COUNTY_STATE", "YEAR", "EMPLOYMENT_RAW", "EMPLOYMENT"]].head(16))

# Copilot prompt: Create the TREATED and POST_POLICY dummy variables.
# TREATED = 1 for Ohio counties (got the subsidy), 0 for PA counties.
# POST_POLICY = 1 for years >= 2022, 0 for years before.
# The DID interaction term is TREATED * POST_POLICY.

long_df["TREATED"] = (long_df["STATE"] == "Ohio").astype(int)
long_df["POST_POLICY"] = (long_df["YEAR"] >= 2022).astype(int)
long_df["DID"] = long_df["TREATED"] * long_df["POST_POLICY"]

# Final clean panel
panel_df = long_df[[
    "COUNTY_STATE", "STATE", "YEAR", "EMPLOYMENT",
    "TREATED", "POST_POLICY", "DID", "BRIEF_TITLE"
]].sort_values(["COUNTY_STATE", "YEAR"]).reset_index(drop=True)

print("\nFinal clean panel:")
print(panel_df.head(20))
print(f"\nPanel shape: {panel_df.shape}")
print(f"Counties: {panel_df['COUNTY_STATE'].nunique()}")
print(f"Years: {panel_df['YEAR'].nunique()}")

# Save the cleaned panel
panel_df.to_csv("clean_panel.csv", index=False)
print("\nSaved clean_panel.csv")