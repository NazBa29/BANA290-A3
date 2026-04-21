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
