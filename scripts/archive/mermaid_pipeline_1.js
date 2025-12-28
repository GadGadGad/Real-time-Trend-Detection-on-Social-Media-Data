flowchart TB
    subgraph Airflow["🎯 Airflow Orchestration"]
        direction TB
        A[seed_initial_trends] --> B{check_mode}
        B -->|Demo| C[trigger_demo_ingest]
        B -->|Live| D[trigger_live_ingest]
        C --> E[run_consumer_processing]
        D --> E
        E --> F[run_ai_analysis]
        F --> G[verify_pipeline_health]
    end
    
    subgraph DataSources["📥 Data Sources"]
        JSON[(trends.json<br/>Google Trends)]
        DEMO[(demo_data/<br/>Pre-computed)]
        LIVE[(VNExpress<br/>TuoiTre Crawler)]
    end
    
    subgraph Streaming["⚡ Kafka Streaming"]
        KAFKA[[Kafka Topic<br/>batch-stream]]
    end
    
    subgraph Processing["🔄 Consumer Processing"]
        direction TB
        EMB[Embedding Model<br/>vietnamese-document-embedding]
        P1["Phase 1: Use pre-assigned<br/>final_topic (similarity=0.95)"]
        P2["Phase 2: Cosine Similarity<br/>Matching (threshold=0.65)"]
        P3["Phase 3: HDBSCAN Clustering<br/>Discovery (new trends)"]
    end
    
    subgraph AI["🧠 Intelligence Worker"]
        LLM[LLM Refiner<br/>Gemma-3-27b-it]
        CAT["Category Classification<br/>T1-T7 Taxonomy"]
        SUM[Summary + 5W1H<br/>Advice Generation]
    end
    
    subgraph Storage["💾 PostgreSQL"]
        DB[(detected_trends<br/>+ embeddings)]
    end
    
    subgraph UI["📊 Streamlit Dashboard"]
        DASH[Real-time Visualization<br/>localhost:8501]
    end
    
    %% Data Flow
    JSON --> A
    DEMO --> C
    LIVE --> D
    C --> KAFKA
    D --> KAFKA
    KAFKA --> E
    E --> EMB
    EMB --> P1 --> P2 --> P3
    P1 --> DB
    P2 --> DB
    P3 --> DB
    DB --> F
    F --> LLM --> CAT --> SUM --> DB
    DB --> DASH
    
    %% Styling
    style Airflow fill:#e1f5fe,stroke:#01579b
    style Streaming fill:#fff3e0,stroke:#e65100
    style Processing fill:#f3e5f5,stroke:#7b1fa2
    style AI fill:#e8f5e9,stroke:#2e7d32
    style Storage fill:#fce4ec,stroke:#c2185b
    style UI fill:#fff8e1,stroke:#ff8f00