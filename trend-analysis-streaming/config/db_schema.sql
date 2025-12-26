-- Tạo bảng lưu log thô (Raw Data) - Dùng để kiểm tra input
CREATE TABLE IF NOT EXISTS raw_logs (
    id SERIAL PRIMARY KEY,
    source VARCHAR(50),      -- 'Face: Beatvn', 'VNEXPRESS', ...
    content TEXT,
    published_at TIMESTAMP,  -- Thời gian gốc của bài viết
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- Thời gian bài vào hệ thống
);

-- Tạo bảng lưu kết quả Clustering & Trend Analysis (Output cuối cùng)
CREATE TABLE IF NOT EXISTS detected_trends (
    batch_id VARCHAR(50),            -- ID của Spark Micro-batch
    cluster_label INT,               -- Nhãn cụm (từ HDBSCAN)
    
    -- Thông tin định danh Trend (từ LLM Refiner)
    trend_name TEXT,                 -- Tên xu hướng (Refined Title)
    topic_type VARCHAR(50),          -- 'Trending', 'Discovery', 'Noise'
    category VARCHAR(20),            -- T1, T2... T7
    
    -- Điểm số (từ ScoreCalculator)
    trend_score FLOAT,               -- Điểm tổng hợp
    score_g FLOAT,                   -- Google Score
    score_f FLOAT,                   -- Facebook Score
    score_n FLOAT,                   -- News Score
    
    -- Thông tin tóm tắt & Sentiment
    summary TEXT,
    sentiment VARCHAR(20),           -- Positive, Negative, Neutral
    
    -- 5W1H (Lưu dạng JSONB để linh hoạt)
    intelligence JSONB,              -- {who, what, where, when, why}
    
    -- Metadata thống kê
    post_count INT,                  -- Số lượng bài trong cụm
    representative_posts JSONB,      -- Lưu 3 bài tiêu biểu (title/url) để hiển thị
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index để Streamlit query cho nhanh
CREATE INDEX idx_trends_created_at ON detected_trends(created_at DESC);
CREATE INDEX idx_trends_score ON detected_trends(trend_score DESC);