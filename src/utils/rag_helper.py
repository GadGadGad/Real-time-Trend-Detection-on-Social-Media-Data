
import os
import json
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

class RAGHelper:
    def __init__(self, db_url=None, model_name="dangvantuan/vietnamese-document-embedding"):
        load_dotenv()
        self.db_url = db_url or os.getenv("POSTGRES_URL", "postgresql://user:password@localhost:5432/trend_db")
        self.engine = create_engine(self.db_url)
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
        return self._model

    def embed_query(self, query):
        emb = self.model.encode(query)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.tolist()

    def get_relevant_trends(self, query, top_k=5):
        query_emb = self.embed_query(query)
        
        # Use SQL similarity comparison if possible, or fetch all and compare in memory
        # For simplicity in demo, we'll fetch recently updated trends and compare
        with self.engine.connect() as conn:
            # Fetch trends that have embeddings
            query = text("""
                SELECT id, trend_name, summary, embedding, representative_posts 
                FROM detected_trends 
                WHERE embedding IS NOT NULL
                ORDER BY last_updated DESC 
                LIMIT 50
            """)
            results = conn.execute(query).fetchall()
            
            scored_trends = []
            for row in results:
                t_emb = json.loads(row.embedding)
                sim = np.dot(query_emb, t_emb)
                scored_trends.append({
                    "id": row.id,
                    "name": row.trend_name,
                    "summary": row.summary,
                    "reps": json.loads(row.representative_posts) if row.representative_posts else [],
                    "score": float(sim)
                })
            
            scored_trends.sort(key=lambda x: x['score'], reverse=True)
            return scored_trends[:top_k]

    def generate_answer(self, query, context_trends):
        """Prepare prompt and call Gemini for RAG answer"""
        from src.core.llm.llm_refiner import LLMRefiner
        
        api_key = os.getenv("GEMINI_API_KEY")
        refiner = LLMRefiner(provider="gemini", api_key=api_key)
        
        if not refiner.enabled:
            return "❌ AI Refiner is disabled. Please check your Gemini API key."

        # Construct context
        context_text = ""
        for i, t in enumerate(context_trends):
            context_text += f"\nTrend {i+1}: {t['name']}\nSummary: {t['summary']}\nSource Examples: "
            for p in t['reps'][:3]:
                context_text += f"[{p.get('source', '?')}] {p.get('content', '')[:100]}... "
            context_text += "\n"

        prompt = f"""Bạn là một chuyên gia phân tích dữ liệu truyền thông thông minh. 
        Dựa trên các sự kiện (trends) đang diễn ra dưới đây, hãy trả lời câu hỏi của người dùng một cách chuyên sâu, khách quan và có dẫn chứng từ dữ liệu.

        NGỮ CẢNH DỮ LIỆU:
        {context_text}

        CÂU HỎI: {query}

        HƯỚNG DẪN TRẢ LỜI:
        1. Nếu dữ liệu không cung cấp đủ thông tin, hãy nói rõ.
        2. Trình bày dưới dạng markdown rõ ràng.
        3. Không bịa đặt thông tin ngoài ngữ cảnh.
        4. Trả lời bằng tiếng Việt.
"""
        
        # We can use refiner.client.generate_content directly or add a new method to LLMRefiner
        # Use the internal _generate method which handles retries and safety settings
        try:
            answer = refiner._generate(prompt)
            return answer if answer else "❌ AI không thể tạo câu trả lời (có thể do bộ lọc an toàn hoặc lỗi kết nối)."
        except Exception as e:
            return f"❌ Lỗi khi gọi AI: {str(e)}"

    def get_semantic_map_data(self, limit=100):
        """Fetch trends and reduce dimensions for 2D plotting"""
        with self.engine.connect() as conn:
            query = text("""
                SELECT trend_name, trend_score, category, embedding 
                FROM detected_trends 
                WHERE embedding IS NOT NULL
                ORDER BY last_updated DESC 
                LIMIT :limit
            """)
            results = conn.execute(query, {"limit": limit}).fetchall()
            
            if not results:
                return None
                
            embeddings = []
            metadata = []
            for row in results:
                try:
                    emb = json.loads(row.embedding)
                    embeddings.append(emb)
                    metadata.append({
                        "name": row.trend_name,
                        "score": row.trend_score,
                        "category": row.category or "Other"
                    })
                except:
                    continue
            
            if len(embeddings) < 3:
                return None
                
            # Dimension reduction using PCA + t-SNE for stability
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
            import pandas as pd
            
            embs_np = np.array(embeddings)
            
            # Reduce to 10D with PCA first if many dimensions
            if embs_np.shape[1] > 10:
                pca = PCA(n_components=min(10, len(embeddings)))
                embs_np = pca.fit_transform(embs_np)
                
            # Reduce to 2D with t-SNE
            tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings)-1), random_state=42, init='pca', learning_rate='auto')
            coords = tsne.fit_transform(embs_np)
            
            df = pd.DataFrame(metadata)
            df['x'] = coords[:, 0]
            df['y'] = coords[:, 1]
            return df

    def generate_daily_report(self):
        """Generate a comprehensive intelligence summary of current trends"""
        with self.engine.connect() as conn:
            query = text("""
                SELECT trend_name, summary, category, trend_score
                FROM detected_trends 
                WHERE summary IS NOT NULL 
                AND summary != 'Waiting for analysis...'
                ORDER BY trend_score DESC 
                LIMIT 15
            """)
            results = conn.execute(query).fetchall()
            
            if not results:
                return "Chưa có đủ dữ liệu đã phân tích để lập báo cáo."
                
            trends_text = ""
            for r in results:
                trends_text += f"- [{r.category}] {r.trend_name}: {r.summary[:200]}...\n"
                
            prompt = f"""
            VAI TRÒ: Chuyên gia phân tích tin tức chiến lược.
            NHIỆM VỤ: Tổng hợp tình hình tin tức dựa trên danh sách các sự kiện đang hot nhất.
            
            DỮ LIỆU ĐẦU VÀO:
            {trends_text}
            
            YÊU CẦU BÁO CÁO (Markdown):
            1. TIÊU ĐỀ: Đặt tiêu đề thu hút và khái quát tình hình hiện tại.
            2. TỔNG QUAN: Nhận định chung về các nhóm chủ đề đang thống trị (ví dụ: Chính trị, Thiên tai, hay Giải trí).
            3. TIÊU ĐIỂM NÓNG: Phân tích 3 sự kiện quan trọng nhất, tại sao chúng quan trọng.
            4. DỰ BÁO & LỜI KHUYÊN: Đưa ra các rủi ro tiềm ẩn và kiến nghị hành động cho cơ quan chức năng/doanh nghiệp.
            
            Yêu cầu: Viết bằng tiếng Việt, văn phong chuyên nghiệp, súc tích nhưng đầy đủ ý.
            """
            
            from src.core.llm.llm_refiner import LLMRefiner
            api_key = os.getenv("GEMINI_API_KEY")
            refiner = LLMRefiner(provider="gemini", api_key=api_key)
            
            return refiner._generate(prompt)

