import os

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

# قاعدة بيانات Chafi (نفس قاعدة Laravel)
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USERNAME", "root")
DB_PASS = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_DATABASE", "chafi")

# نموذج البحث بالمعنى (مدرّب من قبل، يدعم العربية والفرنسية والإنجليزية)
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")

# نموذج إعادة الترتيب: يقرأ السؤال والجواب المرشح معاً ويحكم أيهما يجيب فعلاً (أدق بكثير)
# - BAAI/bge-reranker-v2-m3: الأدق، مناسب لـ GPU (Colab)
# - cross-encoder/mmarco-mMiniLMv2-L12-H384-v1: أخف وأسرع، مناسب لـ VPS بدون GPU
# - فارغ: بدون إعادة ترتيب
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# المحتوى غير المواد (مؤسسات، أسئلة، أنظمة...) يُعطى للنموذج كاملاً حتى هذا الطول
FULL_TEXT_MAX = int(os.getenv("FULL_TEXT_MAX", "2500"))

# النموذج اللغوي المدرّب من قبل، يشتغل عبر Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")

# مكان تشغيل النموذج:
# - ollama: محلياً على هذا السيرفر (LLM_MODEL أعلاه)
# - api: خدمة متوافقة مع OpenAI (OpenRouter، DeepInfra، Together، Groq...) تشغّله على GPU
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://openrouter.ai/api/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_API_MODEL = os.getenv("LLM_API_MODEL", "qwen/qwen-2.5-7b-instruct")

# ---------------- السرعة
# أقصى عدد كلمات (tokens) في الجواب، وحجم سياق النموذج
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "350"))
NUM_CTX = int(os.getenv("NUM_CTX", "4096"))
# عدد أنوية المعالج (0 = يختار Ollama وحده)
NUM_THREAD = int(os.getenv("NUM_THREAD", "0"))
# النموذج يبقى محمّلاً في الذاكرة (-1 = دائماً) حتى لا يُعاد تحميله مع كل سؤال
KEEP_ALIVE = os.getenv("KEEP_ALIVE", "-1")
# Qwen3: إطفاء وضع "التفكير" يقسم وقت الجواب تقريباً على 2
DISABLE_THINKING = os.getenv("DISABLE_THINKING", "1") == "1"

# حجم المقاطع عند بناء الفهرس (أصغر = نص أقل يقرأه النموذج = أسرع)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# عدد المقاطع المسترجعة لكل سؤال (6 مقاطع × 700 حرف ≈ نفس نص 4 مقاطع كبيرة)
TOP_K = int(os.getenv("TOP_K", "6"))

# ---------------- الدقة
# أقل درجة تشابه لاعتبار المصادر مناسبة، وإلا يعاد صياغة السؤال والبحث
# (درجات e5 تكون عادة بين 0.70 و 0.90؛ اضبطها بالتجربة)
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.82"))
MAX_REWRITES = int(os.getenv("MAX_REWRITES", "1"))
# عند عدم فهم السؤال: اقتراح مواضيع قريبة بدل "لم يتم العثور" (درجة تشابه أدنى للاقتراح)
CLARIFY_MIN_SCORE = float(os.getenv("CLARIFY_MIN_SCORE", "0.75"))
CLARIFY_OPTIONS = int(os.getenv("CLARIFY_OPTIONS", "3"))
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "2"))

# مرحلة فهم السؤال (استدعاء إضافي للنموذج):
# auto = فقط للسؤال الناقص أو بالدارجة أو فيه أخطاء (أسرع)، always = لكل سؤال
UNDERSTAND_MODE = os.getenv("UNDERSTAND_MODE", "auto")

# عدد الرسائل السابقة التي تُستعمل لتكملة سؤال ناقص (مثل "وفي G50؟")
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "6"))

# مفتاح سري مشترك مع Laravel (AI_SERVICE_KEY في .env الخاص بـ Laravel)
AI_SERVICE_KEY = os.getenv("AI_SERVICE_KEY", "")

# مجلد الفهرس. deploy/rebuild-index.sh يبني في مجلد مؤقت (INDEX_DIR_OVERRIDE)
# ثم يستبدل القديم، حتى لا تتوقف الخدمة إذا فشل البناء
INDEX_DIR = os.getenv("INDEX_DIR_OVERRIDE") or os.path.join(os.path.dirname(__file__), "index")
