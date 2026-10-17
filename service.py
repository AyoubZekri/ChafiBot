"""خدمة المحادثة: يستدعيها Laravel فقط (المحادثات تُحفظ في قاعدة Laravel).
التشغيل:  uvicorn service:app --host 127.0.0.1 --port 8001
"""
import hmac
import re
import time

import httpx
from fastapi import FastAPI, Header, HTTPException
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

import config
from graph import NOT_FOUND, graph, is_not_found, to_messages
from retrieval import unique_sources

app = FastAPI(title="ChafiBot")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    # [{"role": "user"|"assistant", "content": "..."}] من الأقدم إلى الأحدث
    history: list[dict] = []


# [1] أو [1، 3] في جواب النموذج = أرقام المصادر التي أخذ منها
_CITATION = re.compile(r"\s*\[(\d+(?:\s*[،,]\s*\d+)*)\]")


# "(المصدر: ...)" / "المصادر: ..." / "Source: ..." يكتبها النموذج أحياناً رغم التعليمات
_SOURCE_MENTION = re.compile(
    r"\(\s*(?:ال)?(?:مصدر|مصادر|source|sources)\s*:[^)]*\)"  # بين قوسين
    r"|(?:^|\n)[ \t]*[-*]?[ \t]*(?:ال)?(?:مصدر|مصادر|source|sources)\s*:[^\n]*",  # سطر مستقل
    re.IGNORECASE,
)


def used_sources(answer, docs):
    """المصادر التي استشهد بها النموذج فعلاً (بالترتيب)، والجواب بدون أرقام الاستشهاد
    وبدون ذكر المصدر (التطبيق يعرض المصادر أسفل الجواب).
    المصادر المعروضة مواد القانون فقط؛ إذا جاء الجواب من قسم آخر لا يُعرض مصدر.
    إذا لم يستشهد النموذج بشيء: أول 3 مصادر (الأنسب بعد إعادة الترتيب)."""
    numbers = []
    for group in _CITATION.findall(answer):
        for n in map(int, re.findall(r"\d+", group)):
            if 1 <= n <= len(docs) and n not in numbers:
                numbers.append(n)
    used = [docs[n - 1] for n in numbers] if numbers else docs[:3]
    used = [(m, s) for m, s in used if m.get("type") == "article"]

    text = _SOURCE_MENTION.sub("", answer)
    text = _CITATION.sub("", text)
    text = re.sub(r"[ \t]+([.,،:؛])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), unique_sources(used)


@app.get("/health")
def health():
    if config.LLM_PROVIDER == "api":
        return {"status": "ok" if config.LLM_API_KEY else "degraded",
                "provider": "api", "model": config.LLM_API_MODEL,
                "api_key": "ok" if config.LLM_API_KEY else "ناقص"}
    # هل Ollama يعمل وفيه النموذج؟ (حتى لا يبدو كل شيء سليماً وOllama متوقف)
    try:
        tags = httpx.get(f"{config.OLLAMA_URL}/api/tags", timeout=3).json()
        names = [m.get("name", "") for m in tags.get("models", [])]
        ollama = "ok" if any(n == config.LLM_MODEL or n.startswith(config.LLM_MODEL + ":")
                             for n in names) else f"النموذج {config.LLM_MODEL} غير محمّل"
    except Exception:  # noqa: BLE001
        ollama = "متوقف"
    return {"status": "ok" if ollama == "ok" else "degraded", "model": config.LLM_MODEL,
            "ollama": ollama}


@app.post("/chat")
def chat(req: ChatRequest, x_ai_key: str | None = Header(default=None)):
    # مفتاح مشترك بين Laravel والخدمة حتى لا يستعملها أي أحد
    if config.AI_SERVICE_KEY and not hmac.compare_digest(x_ai_key or "", config.AI_SERVICE_KEY):
        raise HTTPException(status_code=401, detail="unauthorized")

    started = time.time()
    history = to_messages(req.history[-config.HISTORY_LIMIT:])
    try:
        state = graph.invoke(
            {"messages": [*history, HumanMessage(req.message)]},
            config={"recursion_limit": 25},
        )
    except httpx.ConnectError:
        # Ollama متوقف (مثلاً بعد إعادة تشغيل Colab): خطأ واضح بدل 500
        raise HTTPException(status_code=503,
                            detail=f"Ollama غير متاح على {config.OLLAMA_URL}: شغّل ollama serve")
    except Exception as e:  # noqa: BLE001
        # أخطاء خدمة الـ API (مفتاح خاطئ، حد الاستعمال، انقطاع): 503 برسالة واضحة
        if type(e).__module__.startswith("openai"):
            raise HTTPException(status_code=503, detail=f"خدمة النموذج: {type(e).__name__}: {e}")
        raise

    answer = state["messages"][-1]
    # العمليات الحسابية التي نفذتها الأداة في هذا السؤال فقط
    new_messages = state["messages"][len(history) + 1:]
    calculations = [m.content for m in new_messages if isinstance(m, ToolMessage)]

    text = answer.content if isinstance(answer, AIMessage) else str(answer)
    found = not is_not_found(text)
    if not found:
        # نص موحد بلغة السؤال، مهما كتب النموذج
        text = NOT_FOUND[state.get("lang") or "ar"]
    # المصادر تظهر فقط إذا مر السؤال بالبحث، وفقط التي أخذ منها الجواب
    text, sources = used_sources(text, state.get("docs") or [])

    return {
        "answer": text,
        # false = لم يجد الجواب في قاعدة البيانات (Laravel يعتبرها عملية فاشلة)
        "found": found,
        "sources": sources if found else [],
        "calculations": calculations,
        # السؤال كما فهمه المساعد (للتشخيص: إذا كان خاطئاً فالمشكل في الفهم لا في البحث)
        "understood": state.get("standalone"),
        "search_query": state.get("search_query"),
        "language": state.get("lang"),
        "route": state.get("route"),
        "seconds": round(time.time() - started, 2),
    }
