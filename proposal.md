# **PROJECT PROPOSAL**

# **Real-time Trend Detection and Sentiment Analysis System**

## **1. Introduction**
The project aims to build a system for **real-time detection of trends and events** from social media and news data, bridging the gap between *raw data* and *actionable information*.

### **Target Audience**
1.  **Government / Public Safety:** Early detection of social risks, disaster damages, protests, fake news, etc. (*Social Risk*).
2.  **Business / Marketing:** Rapidly capturing consumer trends, viral trends, etc. (*Market Opportunity*).

**Approach:** combining **multi-source signals** (Search, Social, News) and NLP to **cluster topics**, **score trends**, and capture the status and public reaction to the latest trends.

---

## **2. Taxonomy of Events**
To ensure commercial viability and clear actionable insights, events are categorized into **3 Action-Oriented Groups** based on urgency and target audience:

### **Group A: Critical Alerts (High Urgency)**
*Target: Emergency Services, Local Government.*
1.  **Public Safety & Security:** Accidents, fires, explosions, crimes.
2.  **Disaster & Environment:** Floods, storms, pollution, health outbreaks.
3.  **Civil Unrest:** Protests, strikes, riots (physical gatherings).

### **Group B: Social Signals (Monitoring & Sentiment)**
*Target: Policy Makers, PR Agencies.*
4.  **Socio-Political:** New policies, elections, statements by officials (neutral until analyzed for sentiment).
5.  **Controversy & Scandal:** "Drama", boycotts, public accusations, moral debates (often negative sentiment).

### **Group C: Market Trends (Opportunity)**
*Target: Marketing, Brands, Businesses.*
6.  **Consumer & Lifestyle:** Food trends (e.g., "mãng cầu tea"), fashion, travel, tech.
7.  **Pop Culture & Entertainment:** Movies, music, celebrities, memes.

*Why multi-source?* Search reflects *intent*, Social reflects *virality* (but noisy), News reflects *verification* (but delayed).

---

## **3. Data Sources & Collection**

### **3.1 Social Media (Facebook)**
*   **Source:** Large Fanpages / News pages (Theanh28, Dan tri, ...).
*   **Format:** JSON (content + post time + interactions).
*   **Trend Signals:** likes / comments / shares.
*   **Status:** Crawled and unified schema.
*   **Improvement (Apify):**
    *   **Goal:** Use **Apify** to crawl Facebook with **accurate timestamps** (hour/minute/second) even for old posts.
    *   **Fields:** `pageName`, `postId`, `time` (ISO), `timestamp` (epoch), `text`, `likes`, `comments`, `shares`.
    *   **Benefit:** More accurate engagement calculation over time $\rightarrow$ earlier trend detection.

### **3.2 News**
*   **Source:** Thanh Nien, Tuoi Tre, VnExpress, Vietnamnet, ...
*   **Format:** CSV (title, content, publish time, url).
*   **Role:** Event verification + signal of **news density** over time.
*   **Status:** **Crawled** and merged into the common pipeline.

---

## **4. System Architecture**
The system is divided into 2 main phases:

### **Phase 1: Offline Discovery**
*   Collect multi-source historical data.
*   Standardization & Vietnamese preprocessing.
*   Vectorization (embedding) and **Clustering**.
*   Create rules/semi-supervised labels to detect and classify trends.

### **Phase 2: Online Real-time**
*   Periodic collection / streaming.
*   Match posts to topics + detect new topics.
*   Score trends over time.
*   **Analyze sentiment / community psychology for each trend**.
*   Alert & store for dashboard.

**Current Focus:** **Trend detection is the foundational step**; sentiment analysis is performed after the trend is clearly identified.

---

## **5. NLP Trend Detection Pipeline**
**Technical Goal:** Group different expressions (Facebook/News/Search) of the same event into a **single unified topic**.

*   **Preprocessing:** Unicode normalization, boilerplate removal, cleaning character/emoji.
*   **Semantic Representation:** using **Vietnamese embeddings** (PhoBERT / Vietnamese bi-encoder).
*   **Context Enrichment (2 approaches):**
    *   **NER (Underthesea):** Extract entities to improve matching accuracy.
    *   **Aliases from Google Trends:** Expand keyword variants to increase recall for different phrasings.
*   **Matching & Clustering:** Cosine similarity (matching) + UMAP (dimensionality reduction) + HDBSCAN (clustering).

### **Issues & Adjustments**
*   **Problem:** Daily recurring data (weather, gold price, lottery) creates dense "fake topics" and noise.
*   **Solutions Applied:**
    *   Prioritize PhoBERT/Bi-encoder.
    *   Add NER for better grouping.
    *   Use Google Trends aliases.

---

## **6. Trend Scoring (Concept)**
For each **topic**, synthesize 3 signals:
*   **Google Search Score (G):** Interest/growth in search.
*   **Facebook Engagement Score (F):** Interaction normalized over time.
*   **News Coverage Score (N):** Density of articles over time.

**Unified Trend Score:**
$$
T = w_G \cdot G + w_F \cdot F + w_N \cdot N
$$

**Trend Classification:** Search-only / Social-only / News-only / Multi-source confirmed / Emerging / Fading.

---

## **7. Challenges & Future Plan**

### **Current Challenges**
1.  **Clustering Issues:**
    *   Cluster quality is not yet high; topics overlap.
    *   Some clusters are too large due to absorbing noise (daily/spam).
2.  **Noise Filtering:**
    *   No thorough mechanism to remove recurring news (gold price, weather).
    *   Need rule-based (cycle/keyword) + temporal statistics.
3.  **Time Synchronization:**
    *   Social is fast, News has delay, Search reflects intent.
    *   Need **time-binning** normalization for fair scoring.

### **Project Timeline (Status)**

| Task | Description | Status |
| :--- | :--- | :--- |
| **Task 1** | Schema normalization + Basic cleaning (Unicode/boilerplate) | In Progress |
| **Task 2** | Data Collection (Crawler) | **Completed** |
| **Task 3** | Semantic matching (embedding) + alias/NER enrichment | Testing |
| **Task 4** | Optimize Clustering (UMAP + HDBSCAN) + Cluster quality eval | Researching solutions |
| **Task 5** | Trend Naming and Core Information Summarization | TBD |
| **Task 6** | Sentiment/Psychology Analysis from detected trends | TBD |
| **Task 7** | Trend Scoring + Multi-source classification | TBD |
| **Task 8** | Dashboard/Report + Integrated periodic/streaming pipeline | TBD |

---

## **8. References**
*   (References to be added as per `references.bib`)
