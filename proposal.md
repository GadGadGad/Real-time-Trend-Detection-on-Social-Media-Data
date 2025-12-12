# 📝 **PROJECT PROPOSAL**

# **Multi-Source Trend Detection System Using Search, Social, and News Signals**

## **1. Introduction**

In the digital age, public attention shifts rapidly across search platforms, social networks, and online news outlets. Traditional trend detection systems (e.g., Google Trends) rely mainly on search queries, which captures *intent-based interest* but misses trends that emerge purely on social media or are driven by news coverage without triggering search behavior.

This project proposes a **multi-source trend detection system** that integrates signals from **Google Trends**, **verified Facebook news pages**, and **online newspapers** to detect emerging trends more accurately and earlier than single-source systems. By combining **search behavior**, **social engagement**, and **news publishing patterns**, the system aims to produce a more holistic and reliable trend score.

---

## **2. Problem Statement**

Existing trend detection tools suffer from major limitations:

1. **Google Trends** only captures queries typed into Google.
   → Misses trends that people read passively on social media or news but do not search.

2. **Facebook** contains rapid viral dynamics but is noisy and lacks validation.
   → Many social spikes are not “real events”.

3. **News outlets** report verified events but do not guarantee public attention.
   → Some topics receive heavy coverage but little public interest.

Because each platform captures **different aspects of public behavior**, relying on one alone leads to incomplete or inaccurate trend detection.

---

## **3. Project Objective**

The project aims to build a system that:

1. **Collects** trending data from 3 independent sources:

   * Google Trends (search intent)
   * Facebook news pages (social buzz)
   * Online newspapers (verified events)

2. **Normalizes and merges** heterogeneous signals into a unified format.

3. **Clusters related keywords / headlines into topics** using NLP techniques.

4. **Computes a composite Trend Score** based on cross-platform signals.

5. **Classifies trends** into:

   * Social-only trends
   * News-only trends
   * Search-driven trends
   * Strong multi-source confirmed trends

6. **Visualizes** the detected trends over time.

---

## **4. Data Sources & Collection Method**

### **4.1 Google Trends**

* Use pytrends (unofficial API)
* Collect:

  * *Top rising queries* (realtime)
  * *Interest-over-time*
* Data fields:

  * query
  * relative popularity (0–100)
  * category

### **4.2 Facebook Verified News Pages**

Examples:

* VnExpress, Tuổi Trẻ, Thanh Niên, Vietnamnet, Zing, Báo Công An.

Methods:

* Facebook Graph API (if token available)
* Or headless crawler (Playwright/Selenium)

Extract:

* post text
* timestamp
* engagement: likes, shares, comments
* engagement/minute (normalized)

### **4.3 Online Newspapers**

* RSS feeds (very easy & reliable)
* Crawl headlines every 5 minutes

Extract:

* title
* publish time
* category
* article url

---

## **5. Processing & NLP Pipeline**

### **5.1 Text Normalization**

* lowercasing
* remove HTML/emoji
* Vietnamese normalization (Unicode NFC)
* Remove boilerplate (e.g., “[VIDEO]”, “[HOT]”)

### **5.2 Keyword → Topic Clustering**

Because the three sources express the same event differently, we use **topic clustering**:

Methods:

* Sentence-BERT for embeddings
* HDBSCAN or Agglomerative Clustering
* Cosine similarity threshold-based merging

Example:

* “bão yagi”, “bão số 5”, “áp thấp Yagi” → same topic cluster.

---

## **6. Trend Scoring**

Each topic has 3 components:

### **6.1 Google Search Score (G)**

[
G = \frac{\Delta popularity}{baseline}
]

### **6.2 Facebook Engagement Score (F)**

[
F = \frac{likes + shares + comments}{minutes}
]

### **6.3 News Coverage Score (N)**

[
N = \text{number of articles in topic per hour}
]

### **6.4 Unified Trend Score**

[
T = w_G \cdot G + w_F \cdot F + w_N \cdot N
]

Default weights:

* ( w_G = 0.4 )
* ( w_F = 0.35 )
* ( w_N = 0.25 )

Weights are tunable depending on objective.

---

## **7. Trend Classification**

Using threshold-based rules or a simple classifier:

| Class                         | Conditions             |
| ----------------------------- | ---------------------- |
| **Search-only trend**         | G high, F low, N low   |
| **Social-only trend**         | F high, G low, N low   |
| **News-only trend**           | N high, G low, F low   |
| **Strong multi-source trend** | G, F, N all high       |
| **Emerging trend (early)**    | F↑ before G↑ & N↑      |
| **Fading trend**              | All signals decreasing |

---

## **8. System Architecture**

```
            ┌───────────┐       ┌────────────┐
            │ Google     │       │ Facebook    │
            │ Trends API │       │ News Pages  │
            └──────┬────┘       └──────┬──────┘
                   │                   │
                   └─────┬─────────────┘
                         ↓
                ┌───────────────────┐
                │   Data Collector  │
                └─────┬────────────┘
                      ↓
           ┌─────────────────────────┐
           │ NLP Preprocessing       │
           └──────────┬─────────────┘
                      ↓
           ┌─────────────────────────┐
           │ Topic Clustering (NLP)  │
           └──────────┬─────────────┘
                      ↓
           ┌─────────────────────────┐
           │ Multi-source Scoring    │
           └──────────┬─────────────┘
                      ↓
           ┌─────────────────────────┐
           │ Trend Classification    │
           └──────────┬─────────────┘
                      ↓
           ┌─────────────────────────┐
           │ Dashboard / Report      │
           └─────────────────────────┘
```

---

## **9. Expected Outcomes**

* A working system capable of detecting trends faster and more accurately than Google Trends alone.
* Identification of:

  * viral social events
  * news-driven events
  * search-driven public interest
  * strong multi-platform trends
* Visual dashboards showing:

  * top daily/weekly trends
  * signal contributions (search/social/news)
  * trend evolution over time

---

## **10. Tools & Technologies**

* **Python**
* **Libraries:**

  * pytrends
  * requests / playwright
  * BeautifulSoup4
  * pandas
  * sentence-transformers
  * scikit-learn / HDBSCAN
  * matplotlib / seaborn / streamlit
* **Database:** SQLite or MongoDB
* **Deployment:** Streamlit dashboard or local notebook

---

## **11. Project Timeline**

| Week | Task                                   |
| ---- | -------------------------------------- |
| 1    | Data pipeline setup, collect 3 sources |
| 2    | Text normalization + preprocessing     |
| 3    | Topic clustering experiments           |
| 4    | Build scoring system                   |
| 5    | Build trend classification             |
| 6    | Visualization + dashboard              |
| 7    | Testing + report writing               |
| 8    | Final presentation                     |

---

## **12. Contribution to Research / Novelty**

* Multi-source approach (search + social + news)
* Vietnamese-language trend detection
* Topic-level fusion rather than keyword-level
* Time-aligned cross-platform trend scoring
* Detect social-only vs news-only vs real public attention trends

No existing system in Vietnam currently combines these three sources in this way.

---
