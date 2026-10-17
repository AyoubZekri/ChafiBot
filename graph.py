"""مسار المساعد (LangGraph):

START → classify ─(تحية/شكر)→ small_talk → END                    (رد جاهز، بدون النموذج)
            │      └(كلام ملخبط)→ meaningless → END                 ("لم يتم العثور على إجابة")
         (سؤال)
            ↓
         understand   ← يصحح الأخطاء ويحوّل الدارجة للفصحى، ويكمّل السؤال من المحادثة
            │            فقط إذا كان تكملة لها (مثل "وفي G50؟")
            ├─(خارج الجباية: الطقس، الرياضة...)→ off_topic → END     ("أنا ChafiBot، مساعد جبائي فقط")
            ↓
         retrieve (بالسؤال الأصلي + المفهوم)
            │
            ├─(مصادر مناسبة)→ generate ─(طلب حساب)→ tools → generate ...
            │                     └──────────────→ END
     (غير مناسبة، محاولة < MAX)
            ↓
         rewrite → retrieve        (بعد MAX محاولات: generate بأحسن ما وُجد)

كل جواب يكتبه النموذج من المصادر لهذا السؤال (لا يُنسخ جواب محفوظ ولا جواب سابق).
النموذج لا يرى أجوبة المحادثة السابقة، حتى لا يكررها في سؤال لا علاقة له بها.
"""
import random
import re
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

import config
from calculator import calculate
from retrieval import format_sources, is_meaningless, retrieve, unknown_words


def _keep_alive():
    try:
        return int(config.KEEP_ALIVE)
    except ValueError:
        return config.KEEP_ALIVE  # مثل "24h"


def _ollama(num_predict):
    """النموذج اللغوي حسب LLM_PROVIDER:
    - ollama: على هذا الـ VPS (مجاني، بطيء بدون GPU)
    - api: خدمة متوافقة مع OpenAI تشغّل النموذج على GPU (سريع: ثوانٍ)"""
    if config.LLM_PROVIDER == "api":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.LLM_API_MODEL,
            base_url=config.LLM_API_BASE,
            api_key=config.LLM_API_KEY,
            temperature=0.1,
            max_tokens=num_predict,
            timeout=90,
            max_retries=1,
        )
    return ChatOllama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_URL,
        temperature=0.1,  # أجوبة أثبت وأدق
        num_predict=num_predict,
        num_ctx=config.NUM_CTX,
        num_thread=config.NUM_THREAD or None,
        keep_alive=_keep_alive(),
        # وضع التفكير موجود في Qwen3 (غير نسخة instruct) فقط؛ لا نرسل الخيار للنماذج الأخرى
        reasoning=False if config.DISABLE_THINKING and "qwen3" in config.LLM_MODEL
        and "instruct" not in config.LLM_MODEL else None,
    )


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text):
    """يحذف "تفكير" النموذج من الجواب: <think>...</think>، أو تفكير لم يُغلق
    (نسخ Qwen3 "thinking" تفكر دائماً حتى لو طُلب منها ألا تفعل)"""
    text = _THINK_BLOCK.sub("", str(text or ""))
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    if "<think>" in text.lower():  # تفكير لم يكتمل: لا جواب فيه
        return ""
    return text.strip()


llm = _ollama(config.NUM_PREDICT)
llm_with_tools = llm.bind_tools([calculate])
rewriter = _ollama(80)  # إعادة الصياغة سطر واحد
TOOLS = {"calculate": calculate}

SYSTEM_PROMPT = """أنت ChafiBot، المساعد الجبائي الذكي داخل تطبيق شافي الجزائري.
- إذا سُئلت عن اسمك فاسمك ChafiBot.
- جاوب باختصار ووضوح: من 2 إلى 5 جمل.
{language_rule}"""

# لغة الجواب = لغة سؤال المستخدم (المصادر قد تكون بالعربية فيُترجم منها)
LANGUAGE_RULES = {
    "ar": "- جاوب بالعربية (أو بالدارجة الجزائرية إذا كتب المستخدم بالدارجة).",
    "en": "- The user wrote in English: answer ONLY in English, translating from the sources "
          "if they are in Arabic or French. Keep article numbers as they are.",
    "fr": "- L'utilisateur a écrit en français : réponds UNIQUEMENT en français, en traduisant "
          "les sources si elles sont en arabe. Garde les numéros d'articles tels quels.",
}

NOT_FOUND = {
    "ar": "لم يتم العثور على إجابة",
    "en": "No answer found",
    "fr": "Aucune réponse trouvée",
}

# صيغ "لم أجد" التي قد يكتبها النموذج (القديمة أو بكلماته)، لاكتشافها في جوابه
_NOT_FOUND_PATTERNS = re.compile(
    r"لم يتم العثور على (أي )?إجابة|لم أجد (أي )?إجابة|لا توجد إجابة|"
    r"no answer (was )?found|could not find an answer|couldn't find an answer|"
    r"aucune réponse (n'a été )?trouvée|je n'ai pas trouvé de réponse",
    re.IGNORECASE,
)

ANSWER_RULES = """
قواعد الإجابة على الأسئلة الجبائية:
- جاوب عن آخر سؤال للمستخدم فقط، ومن "المصادر" أدناه، ولا تخترع أرقاماً أو نسباً أو مواد أو تواريخ غير موجودة فيها.
- اكتب الجواب بكلماتك من المصادر، ولا تنسخ نص مصدر كما هو.
- المصادر مرتبة من الأنسب للسؤال، وهي من كل أقسام التطبيق (مواد القانون، المؤسسات، الأنظمة الجبائية،
  الأسئلة الشائعة، المواعيد، المقالات). اعتمد على المصدر الذي يجيب عن السؤال بالضبط، حتى لو لم يكن مادة.
- اقرأ كل المصادر جيداً: الجواب قد يكون في أي واحد منها، وقد يُكتب بكلمات مختلفة عن السؤال
  (مثلاً "الدخل السنوي" و"رقم الأعمال السنوي").
- إذا أجابت المصادر عن جزء من السؤال فقط، جاوب بهذا الجزء ووضّح ما لم تجده.
- بعد كل معلومة ضع رقم المصدر الذي أخذتها منه بين قوسين مربعين فقط، مثل [1] أو [2].
- لا تكتب كلمة "المصدر" ولا اسم المصدر ولا عنوانه في الجواب: التطبيق يعرض المصادر وحده أسفل الجواب.
- قل "{not_found}" فقط إذا كانت المصادر كلها لا علاقة لها بالسؤال.
{calc_rule}
شكل الجواب المطلوب (الأرقام والمعلومات تؤخذ من المصادر فقط):
<جواب مباشر في جملة أو جملتين> [رقم المصدر] <تفصيل قصير إذا لزم> [رقم المصدر]

المصادر:
{sources}"""

CALC_RULE = "- لأي عملية حسابية استعمل الأداة calculate ولا تحسب بنفسك، ثم اذكر النتيجة.\n"

# فهم السؤال قبل البحث: المستخدم يكتب بالدارجة، بأخطاء، أو بسؤال ناقص يعتمد على ما قبله
UNDERSTAND_PROMPT = """مهمتك: أعد كتابة آخر رسالة للمستخدم كسؤال جبائي كامل وواضح ومستقل بالعربية الفصحى،
بحيث يُفهم وحده بدون المحادثة.
- استعمل المحادثة السابقة فقط إذا كانت الرسالة تكملة لها. مثال: بعد سؤال عن آخر أجل التصريح G12،
  الرسالة "وفي G50؟" تصبح "ما هو آخر أجل لإيداع التصريح G50؟".
- إذا كانت الرسالة سؤالاً جديداً عن موضوع آخر، فتجاهل المحادثة تماماً ولا تُدخل فيه أي شيء منها.
- إذا كانت الرسالة لا علاقة لها بالجباية والضرائب والتصريحات والمؤسسات والأنشطة التجارية
  والمحاسبة والقانون الجبائي (مثل الطقس، الرياضة، الطبخ، الصحة، النكت)، فاكتب فقط: OFF_TOPIC
- صحّح كل الأخطاء الإملائية، خاصة في المصطلحات الجبائية: حروف ناقصة أو زائدة أو مقلوبة،
  ة/ه، ى/ي، الهمزات (مثلاً: "المقاول الذتي" ← "المقاول الذاتي"، "ضريبه الدخل" ← "ضريبة الدخل"،
  "التصريج" ← "التصريح"، "رقم الاعمل" ← "رقم الأعمال").
- حوّل الدارجة إلى الفصحى (واش/شنو = ما، قداش/شحال = كم، نخلص = أدفع، شيفر دافار = رقم الأعمال).
- احتفظ بالرموز والأرقام كما هي (G12، G50، IFU، TVA، أرقام المواد، المبالغ).
- إذا كانت الرسالة واضحة أصلاً فأعد كتابتها بالفصحى فقط.
- لا تجب عن السؤال. اكتب السؤال وحده في سطر واحد.

المحادثة السابقة:
{history}

آخر رسالة: {question}
السؤال الكامل:"""

# المحاولة الثانية بعد الفشل: صياغة مختلفة بكلمات أخرى
REWRITE_PROMPT = """البحث بهذا السؤال لم يجد جواباً في نصوص القانون الجبائي الجزائري:
{question}

اكتب صياغة أخرى للسؤال بكلمات مختلفة: استعمل مرادفات ومصطلحات قانونية وجبائية رسمية
(مثل: غرامة/عقوبة/زيادة، أجل/موعد، تصريح/إقرار، مكلف/خاضع للضريبة).
اكتب السؤال الجديد فقط في سطر واحد."""

# ---------------- الردود الجاهزة (فورية)

GREETING = re.compile(
    r"^\s*(ال?سلام|سلام|مرحبا|اهلا|أهلا|صباح|مساء|hello|hi|hey|good (morning|evening)|"
    r"bonjour|bonsoir|salut)",
    re.IGNORECASE,
)
THANKS = re.compile(
    r"^\s*(صحيت|صحا|يعطيك|شكرا|شكراً|بارك|merci|thanks|thank you|thx)",
    re.IGNORECASE,
)

SMALL_TALK_REPLIES = {
    "greeting": {
        "ar": "وعليكم السلام ورحمة الله، أنا ChafiBot 😊 كيف يمكنني مساعدتك في الجباية اليوم؟",
        "en": "Hello, I'm ChafiBot 😊 How can I help you with taxes today?",
        "fr": "Bonjour, je suis ChafiBot 😊 Comment puis-je vous aider en fiscalité ?",
    },
    "thanks": {
        "ar": "العفو، يسعدني ذلك! إذا كان لديك سؤال آخر فأنا في الخدمة. 🙏",
        "en": "You're welcome! I'm here if you have another question. 🙏",
        "fr": "Avec plaisir ! Je suis là si vous avez une autre question. 🙏",
    },
}

# أسئلة خارج الجباية: عدة صياغات حتى لا يتكرر نفس الرد
OFF_TOPIC_REPLIES = {
    "ar": [
        "أنا ChafiBot، مساعد جبائي فقط 😊 لا أستطيع الإجابة عن هذا، لكن يسعدني مساعدتك في الضرائب والتصريحات والأنظمة الجبائية.",
        "عذراً، هذا خارج تخصصي؛ أنا ChafiBot، مساعد الجباية في تطبيق شافي. اسألني مثلاً عن آخر أجل تصريح أو عن نظامك الجبائي.",
        "هذا السؤال خارج مجال الجباية 🙂 أنا هنا لمساعدتك في الضرائب والغرامات والمواعيد الجبائية والمؤسسات.",
        "لا أملك معلومات عن هذا الموضوع، فأنا ChafiBot، مساعد جبائي فقط. هل لديك سؤال عن الضرائب أو التصريحات؟",
    ],
    "en": [
        "I'm ChafiBot, a tax-only assistant 😊 I can't help with that, but I'm happy to help with taxes, declarations and tax regimes.",
        "Sorry, that's outside my scope; I'm ChafiBot, Chafi's tax assistant. Ask me, for example, about a filing deadline or your tax regime.",
        "That question isn't about taxes 🙂 I'm here to help with taxes, penalties, tax deadlines and businesses.",
        "I don't have information on that topic, as I only cover taxation. Do you have a question about taxes or declarations?",
    ],
    "fr": [
        "Je suis ChafiBot, uniquement un assistant fiscal 😊 Je ne peux pas répondre à cela, mais je peux vous aider pour les impôts et les déclarations.",
        "Désolé, cela sort de mon domaine : je suis ChafiBot, l'assistant fiscal de Chafi. Demandez-moi par exemple une date limite de déclaration.",
        "Cette question ne concerne pas la fiscalité 🙂 Je suis là pour les impôts, les pénalités et les échéances fiscales.",
    ],
}

# كلمات تدل أن السؤال جبائي: لا يُعتبر خارج الموضوع أبداً حتى لو قال النموذج ذلك
_TAX_WORDS = re.compile(
    r"ضريب|ضرائب|جباي|جبائ|تصريح|تصاريح|رسم|رسوم|غرام|عقوب|جزاء|مكلف|مؤسس|مقاول|نشاط|"
    r"دخل|أعمال|اعمال|فاتور|محاسب|ميزاني|أجل|موعد|إعفاء|اعفاء|اقتطاع|اشتراك|ضمان|"
    r"\bg\s*\d+|ج\s*\d+|ifu|tva|tap|irg|ibs|"
    r"\btax|impôt|impot|taxe|fiscal|déclar|declar|pénalit|penalt|revenue|income|vat\b",
    re.IGNORECASE,
)

_FRENCH_WORDS = re.compile(
    r"\b(le|la|les|des|une|est|sont|quel|quelle|quels|combien|impôt|impôts|taxe|déclaration|"
    r"je|pour|avec|dans|bonjour|bonsoir|salut|merci|pénalité|retard)\b",
    re.IGNORECASE,
)

# أرقام تدل على حساب: 3 أرقام فأكثر، نسبة، أو عدد عشري (G12 أو G50 لا تُحسب)
_NEEDS_CALC = re.compile(r"\d{3,}|\d+\s*%|\d+[.,]\d+")


def detect_language(text):
    """ar إذا غلبت الحروف العربية، وإلا fr أو en"""
    arabic = len(re.findall(r"[؀-ۿ]", text))
    latin = len(re.findall(r"[A-Za-zÀ-ÿ]", text))
    if arabic >= latin:
        return "ar"
    return "fr" if len(_FRENCH_WORDS.findall(text)) >= 1 and not re.search(
        r"\b(the|is|what|how|my|for)\b", text, re.IGNORECASE) else "en"


def small_talk_kind(text):
    """تحية أو شكر قصير (بدون سؤال) ← رد جاهز"""
    if len(text.split()) > 5 or "?" in text or "؟" in text:
        return None
    if THANKS.match(text):
        return "thanks"
    if GREETING.match(text):
        return "greeting"
    return None


def system_prompt(lang):
    return SYSTEM_PROMPT.format(language_rule=LANGUAGE_RULES[lang])


class State(TypedDict):
    messages: Annotated[list, add_messages]  # المحادثة: السابقة + الحالية + ردود الأدوات
    question: str  # سؤال المستخدم الأصلي
    standalone: str  # السؤال كما فُهم: كامل، مصحح، بالفصحى
    off_topic: bool  # سؤال خارج الجباية (الطقس، الرياضة...)
    lang: str  # لغة الجواب: ar / en / fr
    search_query: str  # نص البحث (قد يُعاد صياغته)
    docs: list  # [(مقطع, درجة)]
    attempts: int  # عدد إعادات الصياغة
    tool_rounds: int  # عدد مرات استدعاء الأدوات
    route: str  # الطريق الذي أُخذ: small_talk / not_found / llm


# ---------------- العُقد

def classify(state: State):
    question = state["messages"][-1].content
    return {"question": question, "standalone": question, "search_query": question,
            "lang": detect_language(question), "off_topic": False,
            "docs": [], "attempts": 0, "tool_rounds": 0, "route": "llm"}


def route_after_classify(state: State):
    if small_talk_kind(state["question"]):
        return "small_talk"
    # كلام ملخبط: "لم يتم العثور على إجابة" فوراً بدون بحث ولا نموذج
    if is_meaningless(state["question"]):
        return "meaningless"
    return "understand"


def meaningless(state: State):
    return {"messages": [AIMessage(NOT_FOUND[state["lang"]])], "route": "not_found"}


# رسالة تكمّل ما قبلها: "وفي G50؟"، "وهذا؟"، "and for companies?"، "et pour G50 ?"
_FOLLOW_UP = re.compile(
    r"^\s*(و(في|على|عن|ل|ال|هل|كم|ش|ق|ما|اذا|إذا|كيف|بالنسبة)|ف(ي|ماذا)|ثم|بالنسبة|"
    r"and\b|what about|how about|et\b|alors|pour\b)"
    r"|(هذا|هذه|ذلك|تلك|هاذا|هاذي|هذاك|هذيك|نفس)\b|\b(this|that|it|same|ça|cela|même)\b",
    re.IGNORECASE,
)


def is_follow_up(question):
    """هل الرسالة تكملة للمحادثة (تحتاجها لتُفهم)، أم سؤال جديد مستقل؟"""
    return len(question.split()) <= 3 or bool(_FOLLOW_UP.search(question))


def _history_text(messages, limit=4, max_chars=300):
    """آخر الرسائل السابقة (بدون الرسالة الحالية) كنص قصير"""
    lines = []
    for m in messages[:-1]:
        if isinstance(m, HumanMessage):
            lines.append(f"المستخدم: {str(m.content)[:max_chars]}")
        elif isinstance(m, AIMessage) and m.content and not m.tool_calls:
            lines.append(f"المساعد: {str(m.content)[:max_chars]}")
    return "\n".join(lines[-limit:]) or "(لا توجد)"


# كلمات دارجة: السؤال يحتاج تحويلاً للفصحى قبل البحث
_DIALECT = re.compile(
    r"(^|\s)(واش|وش|شنو|شني|شنوة|قداش|شحال|كيفاش|علاش|وين|نخلص|نخلّص|نقدر|راني|راهي|راه|"
    r"بزاف|ماشي|نتاع|تاع|ديال|كاش|باش|نديرو|ندير|يخلص|مازال|برك|بصح|شيفر|دافار)(\s|$|[؟?،,.])"
)


def needs_understanding(state: State):
    """مرحلة الفهم (استدعاء إضافي للنموذج، بطيء على المعالج) فقط عند الحاجة:
    سؤال تكملة لمحادثة، أو بالدارجة، أو فيه كلمات غير معروفة (أخطاء إملائية أو لغة أخرى)"""
    if config.UNDERSTAND_MODE == "always":
        return True
    question = state["question"]
    has_history = any(isinstance(m, AIMessage) for m in state["messages"][:-1])
    return bool(
        (has_history and is_follow_up(question))
        or _DIALECT.search(question)
        or unknown_words(question)
    )


def understand(state: State):
    """السؤال الكامل والمصحح: يُستعمل للبحث ويُعطى للنموذج ليعرف ماذا يقصد المستخدم.
    المحادثة السابقة تُعطى فقط إذا كانت الرسالة تكملة لها؛ وإلا قد "يكمّل" النموذج
    سؤالاً جديداً بالسؤال السابق فيرجع جواباً قديماً لا علاقة له."""
    if not needs_understanding(state):
        # سؤال واضح بالفصحى: يُبحث به كما هو، بدون استدعاء النموذج
        return {"standalone": state["question"], "search_query": state["question"]}
    history = _history_text(state["messages"]) if is_follow_up(state["question"]) else "(لا توجد)"
    text = rewriter.invoke(UNDERSTAND_PROMPT.format(
        history=history, question=state["question"])).content.strip()
    # خارج الجباية، إلا إذا كان في السؤال كلمة جبائية (حماية من خطأ النموذج)
    if "OFF_TOPIC" in text.upper() and not _TAX_WORDS.search(state["question"]):
        return {"off_topic": True}
    text = strip_thinking(text)
    line = text.splitlines()[0].strip(" \"'«»") if text else ""
    line = line.replace("OFF_TOPIC", "").strip()
    line = re.sub(r"^(السؤال( الكامل)?|سؤال)\s*:\s*", "", line)
    # حماية: إذا خرج النموذج عن المطلوب (فارغ، طويل، أو كتب تفكيره بلغة أخرى
    # بدل سؤال بالعربية) نرجع للسؤال الأصلي
    if not line or len(line) > 300 or (
            state["lang"] == "ar" and detect_language(line) != "ar"):
        line = state["question"]
    return {"standalone": line, "search_query": line}


def route_after_understand(state: State):
    return "off_topic" if state["off_topic"] else "retrieve"


def off_topic(state: State):
    reply = random.choice(OFF_TOPIC_REPLIES[state["lang"]])
    return {"messages": [AIMessage(reply)], "route": "off_topic"}


def small_talk(state: State):
    kind = small_talk_kind(state["question"])
    reply = SMALL_TALK_REPLIES[kind][state["lang"]]
    return {"messages": [AIMessage(reply)], "route": "small_talk"}


def retrieve_node(state: State):
    # البحث بكل الصياغات معاً: ما يفوت واحدة تلتقطه الأخرى
    queries = [state["search_query"], state["standalone"], state["question"]]
    # المرتّب يحكم بالسؤال المفهوم (كامل ومصحح) لا بالأصلي
    return {"docs": retrieve(list(dict.fromkeys(q for q in queries if q)),
                             rerank_query=state["standalone"])}


def route_after_retrieve(state: State):
    # الترتيب هجين (معنى + كلمات)، فنأخذ أعلى درجة تشابه بالمعنى
    best = max((score for _, score in state["docs"]), default=0)
    if best >= config.RELEVANCE_THRESHOLD:
        return "generate"
    if state["attempts"] < config.MAX_REWRITES:
        return "rewrite"
    # بعد المحاولات: نعطي النموذج أحسن ما وجدنا، وهو يقرر إن كان فيه جواب
    return "generate"


def rewrite(state: State):
    text = strip_thinking(rewriter.invoke(
        REWRITE_PROMPT.format(question=state["standalone"])).content)
    line = text.splitlines()[0].strip(" \"'«»") if text else ""
    if not line or detect_language(line) != "ar":
        line = state["standalone"]
    return {"search_query": line, "attempts": state["attempts"] + 1}


def generate(state: State):
    lang = state["lang"]
    # أداة الحساب فقط إذا كان في السؤال أرقام، وبعد آخر جولة مسموحة نجبره على الجواب
    use_tools = bool(_NEEDS_CALC.search(state["question"])) and \
        state["tool_rounds"] < config.MAX_TOOL_ROUNDS
    system = system_prompt(lang) + ANSWER_RULES.format(
        sources=format_sources(state["docs"]), not_found=NOT_FOUND[lang],
        calc_rule=CALC_RULE if use_tools else "")
    # المستخدم قد يكتب بالدارجة أو بسؤال ناقص: نعطي النموذج ما فهمناه
    if state["standalone"] != state["question"]:
        system += f"\n\nالمقصود من آخر رسالة للمستخدم: {state['standalone']}"
    model = llm_with_tools if use_tools else llm
    # السؤال الحالي فقط (مع ردود أداة الحساب): لا أجوبة سابقة يكررها النموذج.
    # السؤال الناقص كُمّل في "understand"، فالمحادثة السابقة لم تعد لازمة هنا
    reply = model.invoke([SystemMessage(system), *_current_turn(state["messages"])])
    if isinstance(reply, AIMessage) and not reply.tool_calls:
        # بدون تفكير النموذج؛ وإذا لم يبقَ جواب (تفكير لم يكتمل) فهو "لم يتم العثور"
        text = strip_thinking(reply.content)
        # سؤال بالعربية وجواب بلا عربية = تفكير النموذج بالإنجليزية لا جواب
        if lang == "ar" and detect_language(text) != "ar":
            text = ""
        reply.content = text or NOT_FOUND[lang]
    return {"messages": [reply], "route": "llm"}


def _current_turn(messages):
    """الرسائل من آخر رسالة للمستخدم (السؤال الحالي + طلبات الأدوات ونتائجها)"""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i:]
    return messages[-1:]


def is_not_found(text):
    """هل الجواب هو "لم يتم العثور على إجابة" (بأي لغة أو صياغة)"""
    return bool(_NOT_FOUND_PATTERNS.search(str(text or "")))


def route_after_generate(state: State):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    # لم يجد الجواب: نعيد صياغة السؤال ونبحث مرة أخرى قبل الاستسلام
    if is_not_found(last.content) and state["attempts"] < config.MAX_REWRITES:
        return "retry"
    return END


def retry(state: State):
    # نحذف جواب "لم أجد" حتى لا يراه النموذج في المحاولة الثانية
    return {"messages": [RemoveMessage(id=state["messages"][-1].id)]}


def tools(state: State):
    results = []
    for call in state["messages"][-1].tool_calls:
        tool = TOOLS.get(call["name"])
        output = tool.invoke(call["args"]) if tool else f"أداة غير معروفة: {call['name']}"
        results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
    return {"messages": results, "tool_rounds": state["tool_rounds"] + 1}


# ---------------- بناء المسار

def build_graph():
    g = StateGraph(State)
    g.add_node("classify", classify)
    g.add_node("small_talk", small_talk)
    g.add_node("meaningless", meaningless)
    g.add_node("understand", understand)
    g.add_node("off_topic", off_topic)
    g.add_node("retrieve", retrieve_node)
    g.add_node("rewrite", rewrite)
    g.add_node("generate", generate)
    g.add_node("tools", tools)
    g.add_node("retry", retry)

    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", route_after_classify,
                            ["small_talk", "meaningless", "understand"])
    g.add_edge("meaningless", END)
    g.add_conditional_edges("understand", route_after_understand, ["off_topic", "retrieve"])
    g.add_edge("off_topic", END)
    g.add_conditional_edges("retrieve", route_after_retrieve, ["generate", "rewrite"])
    g.add_edge("rewrite", "retrieve")  # حلقة إعادة البحث
    g.add_conditional_edges("generate", route_after_generate, ["tools", "retry", END])
    g.add_edge("tools", "generate")  # حلقة الحساب
    g.add_edge("retry", "rewrite")  # لم يجد الجواب: بحث جديد بصياغة أخرى
    g.add_edge("small_talk", END)
    return g.compile()


graph = build_graph()


def to_messages(history):
    """تحويل المحادثة المحفوظة في Laravel إلى رسائل LangChain"""
    out = []
    for m in history:
        if m.get("role") == "user":
            out.append(HumanMessage(m.get("content", "")))
        elif m.get("role") == "assistant":
            out.append(AIMessage(m.get("content", "")))
    return out
