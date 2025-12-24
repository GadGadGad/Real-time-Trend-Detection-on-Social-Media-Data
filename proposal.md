# **PROJECT PROPOSAL**

# **Real-time Trend Detection and Sentiment Analysis System**

## **1. Introduction**

The project aims to build a system for **real-time detection of trends and events** from social media and news data, bridging the gap between *raw data* and *actionable information*.

### **Target Audience**

1. **Government / Public Safety:** Early detection of social risks, disaster damages, protests, fake news, etc. (*Social Risk*).
2. **Business / Marketing:** Rapidly capturing consumer trends, viral trends, etc. (*Market Opportunity*).

**Approach:** combining **multi-source signals** (Search, Social, News) and NLP to **cluster topics**, **score trends**, and capture the status and public reaction to the latest trends.

---

## **2. Taxonomy of Events**

To ensure commercial viability and clear actionable insights, events are categorized into **3 Action-Oriented Groups** based on urgency and target audience:

### **Group A: Critical Alerts (High Urgency)**

*Target: Emergency Services, Local Government.*

1. **Public Safety & Security:** Accidents, fires, explosions, crimes.
2. **Disaster & Environment:** Floods, storms, pollution, health outbreaks.
3. **Civil Unrest:** Protests, strikes, riots (physical gatherings).

### **Group B: Social Signals (Monitoring & Sentiment)**

*Target: Policy Makers, PR Agencies.*

4. **Socio-Political:** New policies, elections, statements by officials (neutral until analyzed for sentiment).
5. **Controversy & Scandal:** "Drama", boycotts, public accusations, moral debates (often negative sentiment).

### **Group C: Market Trends (Opportunity)**

*Target: Marketing, Brands, Businesses.*

6. **Consumer & Lifestyle:** Food trends (e.g., "mãng cầu tea"), fashion, travel, tech.
7. **Pop Culture & Entertainment:** Movies, music, celebrities, memes.

*Why multi-source?* Search reflects *intent*, Social reflects *virality* (but noisy), News reflects *verification* (but delayed).

---

## **3. Data Sources & Collection**

### **3.1 Social Media (Facebook)**

* **Source:** Large Fanpages / News pages (Theanh28, Dan tri, ...).
* **Format:** JSON (content + post time + interactions).
* **Trend Signals:** likes / comments / shares.
* **Status:** Crawled and unified schema.
* **Improvement (Apify):**
  * **Goal:** Use **Apify** to crawl Facebook with **accurate timestamps** (hour/minute/second) even for old posts.
  * **Fields:** `pageName`, `postId`, `time` (ISO), `timestamp` (epoch), `text`, `likes`, `comments`, `shares`.
  * **Benefit:** More accurate engagement calculation over time $\rightarrow$ earlier trend detection.

### **3.2 News**

* **Source:** Thanh Nien, Tuoi Tre, VnExpress, Vietnamnet, ...
* **Format:** CSV (title, content, publish time, url).
* **Role:** Event verification + signal of **news density** over time.
* **Status:** **Crawled** and merged into the common pipeline.

---

## **4. System Architecture**

The system logic is divided into two operational modes, utilizing a specialized clustering strategy to handle multi-source variety:

### **4.1 Social-Aware Hierarchical Clustering (SAHC)**

Instead of standard flat clustering, we implement a **3-phase SAHC strategy** to balance factual grounding with viral discovery:

1. **Phase 1: News-First Anchoring:** Perform high-confidence clustering on verified News data to establish a factual "ground truth" for major current events.
2. **Phase 2: Social Attachment:** Map social media posts to the established News clusters using centroid-similarity. This validates viral chatter against reported news.
3. **Phase 3: Social Discovery:** Cluster all remaining unattached social posts to detect "Social-Only" trends (memes, drama, or early-breaking news not yet in professional media).

### **4.2 Processing Phases**

* **Offline Discovery:** Historical data analysis to establish baseline trends and category rules.
* **Online Real-time:** Periodic streaming through the SAHC pipeline to track trend evolution, score growth, and analyze sentiment.

---

## **5. NLP Trend Detection & LLM Refinement**

The pipeline moves beyond pure embedding matching by utilizing **LLM-in-the-Loop** refinement for high-quality output:

* **Stage 0: Context Summarization:** LLM/ViT5 summarizes long posts to distill core information before vectorization, reducing noise in high-dimensional space.
* **Stage 1: Semantic Representation:** Vietnamese-tuned embeddings (PhoBERT/mpnet) with NER and Keyword enrichment.
* **Stage 2: SAHC Clustering:** (As described in Section 4.1).
* **Stage 3: LLM Refinement & Naming:**
  * **Headline Generation:** LLM transforms raw cluster keywords into concise, factual Vietnamese headlines.
  * **Intelligent Taxonomy:** LLM assigns Group (A/B/C) based on event context and urgency.
  * **Specificity Filtering:** Automatic downgrading of "Generic" topics (weather, daily sports results) to prevent dashboard clutter.
* **Stage 4: Semantic Deduplication:** A final LLM pass identifies and merges clusters that refer to the same real-world event but were split due to linguistic variation.

### **Issues & Adjustments**

* **Problem:** Data has many small topics with no density peaks, often causing 80%+ noise in standard HDBSCAN.
* **Solution:** Transitioned to **SAHC** and **LLM Refinement** to proactively define and merge topics rather than relying purely on unsupervised density.

---

## **6. Trend Scoring (Concept)**

For each **topic**, synthesize 3 signals:

* **Google Search Score (G):** Interest/growth in search.
* **Facebook Engagement Score (F):** Interaction normalized over time.
* **News Coverage Score (N):** Density of articles over time.

**Unified Trend Score:**
$$
T = w_G \cdot G + w_F \cdot F + w_N \cdot N
$$

**Trend Classification:** Search-only / Social-only / News-only / Multi-source confirmed / Emerging / Fading.

---

## **7. Challenges & Future Plan**

### **Current Challenges**

1. **Clustering Issues:**
  * Cluster quality is not yet high; topics overlap.
  * Some clusters are too large due to absorbing noise (daily/spam).
2. **Noise Filtering:**
  * No thorough mechanism to remove recurring news (gold price, weather).
  * Need rule-based (cycle/keyword) + temporal statistics.
3. **Time Synchronization:**
  * Social is fast, News has delay, Search reflects intent.
  * Need **time-binning** normalization for fair scoring.

### **Project Timeline (Status)**

| Task | Description | Status |
| :--- | :--- | :--- |
| **Task 1** | Schema normalization + Basic cleaning (Unicode/boilerplate) | **Completed** |
| **Task 2** | Data Collection (Crawler) | **Completed** |
| **Task 3** | Semantic matching (embedding) + alias/NER enrichment | **Completed** |
| **Task 4** | Optimize Clustering (SAHC/HDBSCAN) + Cluster quality eval | **Completed** |
| **Task 5** | Trend Naming and Core Information Summarization | **Completed** |
| **Task 6** | Sentiment/Psychology Analysis from detected trends | **Completed** |
| **Task 7** | Trend Scoring + Multi-source classification | **Completed** |
| **Task 8** | Dashboard/Report + Integrated periodic/streaming pipeline | **Completed** |

---

## **8. References**

* (References to be added as per `references.bib`)
