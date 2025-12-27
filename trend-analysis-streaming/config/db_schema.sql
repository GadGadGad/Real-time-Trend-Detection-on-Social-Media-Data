-- [CONFIG] db_schema.sql

-- Bảng lưu log thô (Raw Data) - Dùng để kiểm tra input
CREATE TABLE IF NOT EXISTS raw_logs (
    id SERIAL PRIMARY KEY,
    source VARCHAR(50),      -- 'Face: Beatvn', 'VNEXPRESS', ...
    content TEXT,
    published_at TIMESTAMP,  -- Thời gian gốc của bài viết
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- Thời gian bài vào hệ thống
);

-- Bảng lưu kết quả Clustering & Trend Analysis (Output cuối cùng)
CREATE TABLE IF NOT EXISTS detected_trends (
    batch_id VARCHAR(50),            -- ID của Spark Micro-batch
    cluster_label INT,               -- Nhãn cụm (từ HDBSCAN)
    
    -- Thông tin định danh Trend
    trend_name TEXT,                 -- Tên xu hướng
    topic_type VARCHAR(50),          -- 'Trending', 'Discovery', 'Noise'
    category VARCHAR(20),            -- T1, T2... T7
    
    -- Điểm số
    trend_score FLOAT,               -- Điểm tổng hợp
    score_g FLOAT,                   -- Google Score
    score_f FLOAT,                   -- Facebook Score
    score_n FLOAT,                   -- News Score
    
    -- Thông tin AI Analysis (Cập nhật mới)
    summary TEXT,
    sentiment VARCHAR(50),           -- Positive, Negative, Neutral
    top_entities JSONB,              -- Lưu danh sách thực thể bóc tách được
    intelligence JSONB,              -- {who, what, where, when, why}
    
    -- Metadata thống kê
    post_count INT,                  -- Số lượng bài trong cụm
    representative_posts JSONB,      -- Lưu 3 bài tiêu biểu
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index để Streamlit query cho nhanh
CREATE INDEX IF NOT EXISTS idx_trends_created_at ON detected_trends(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_trends_score ON detected_trends(trend_score DESC);