"""البحث في فهرس Chafi (يُبنى بـ build_index.py)

البحث الذكي على 3 مراحل:
1. مرشحون (هجين): بالمعنى (embeddings) + بالكلمات (BM25 يتسامح مع الأخطاء الإملائية)،
   مع حصة مضمونة لكل قسم من التطبيق حتى لا تطغى المواد (وهي الأكثر عدداً) على الباقي.
2. إعادة ترتيب (reranker): نموذج يقرأ السؤال وكل جواب مرشح معاً ويحكم أيهما يجيب فعلاً.
3. الأفضل: أعلى المرشحين، بحد أقصى مقطعين من نفس العنصر حتى تتنوع المصادر.
"""
import difflib
import json
import math
import os
import re
from collections import Counter, defaultdict

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

import config

embedder = SentenceTransformer(config.EMBED_MODEL)
reranker = CrossEncoder(config.RERANK_MODEL, max_length=512) if config.RERANK_MODEL else None
vectors = np.load(os.path.join(config.INDEX_DIR, "vectors.npy"))
with open(os.path.join(config.INDEX_DIR, "meta.json"), encoding="utf-8") as f:
    meta = json.load(f)

TYPE_LABELS = {
    "article": "مادة",
    "tax_system": "نظام جبائي",
    "penalty": "جزاءات",
    "institution": "مؤسسة",
    "different": "مواضيع جبائية أخرى",
    "appointment": "موعد والتزام",
    "activity": "نشاط",
    "post": "مقال",
}

# ---------------- البحث بالكلمات (BM25)

_STOP = set("""في من على الى إلى عن ما ماذا هل هي هو هذا هذه ذلك التي الذي و او أو مع كل عند
واش شنو شحال قداش كيفاش علاش وين نتاع تاع ديال حسب the a an of to in is are what how for my i
le la les de des du un une et est en pour""".split())
_PREFIXES = ("وال", "بال", "فال", "كال", "لل", "ال")


def _normalize(text):
    text = str(text or "").lower()
    text = re.sub(r"[ً-ْـ]", "", text)  # التشكيل والتطويل
    text = re.sub(r"[إأآا]", "ا", text)
    text = text.replace("ة", "ه").replace("ى", "ي")
    # "G 12" / "g-12" / "لG 12" / "ج12" ← "g12"
    text = re.sub(r"[؀-ۿ]+(?=[a-z])", " ", text)  # حرف عربي ملتصق بالرمز
    text = re.sub(r"\b([a-z])\s*[-]?\s*(\d+)\b", r"\1\2", text)
    text = re.sub(r"(?<![\w])ج\s*(\d+)", r"g\1", text)
    return text


def tokenize(text):
    tokens = []
    for t in re.findall(r"[\w%]+", _normalize(text)):
        if t in _STOP or (len(t) < 2 and not t.isdigit()):
            continue
        for p in _PREFIXES:  # تجذيع خفيف: "الضريبة" و"ضريبة" نفس الكلمة
            if t.startswith(p) and len(t) - len(p) >= 3:
                t = t[len(p):]
                break
        tokens.append(t)
    return tokens


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.postings = defaultdict(list)  # كلمة ← [(رقم المقطع, التكرار)]
        self.lengths = []
        for i, doc in enumerate(docs):
            counts = Counter(tokenize(doc))
            self.lengths.append(sum(counts.values()))
            for token, tf in counts.items():
                self.postings[token].append((i, tf))
        n = len(docs)
        self.avg = (sum(self.lengths) / n) if n else 0
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}
        # الكلمات حسب طولها، لإيجاد أقرب كلمة صحيحة لكلمة فيها خطأ إملائي
        self.by_length = defaultdict(list)
        for t in self.postings:
            self.by_length[len(t)].append(t)

    def _closest(self, token):
        """أقرب كلمات موجودة في المحتوى لكلمة غير موجودة (خطأ إملائي)"""
        if token.isdigit() or len(token) < 3:
            return []
        pool = [t for n in range(len(token) - 1, len(token) + 2) for t in self.by_length.get(n, ())]
        return difflib.get_close_matches(token, pool, n=2, cutoff=0.8)

    def scores(self, query):
        out = defaultdict(float)
        for token in set(tokenize(query)):
            # الكلمة كما هي، أو أقرب كلمة صحيحة لها بوزن أقل
            variants = [(token, 1.0)] if token in self.postings else \
                [(t, 0.8) for t in self._closest(token)]
            for word, weight in variants:
                for i, tf in self.postings[word]:
                    norm = 1 - self.b + self.b * self.lengths[i] / (self.avg or 1)
                    out[i] += weight * self.idf[word] * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return out


def _search_text(m):
    number = f" المادة {m['number']} article {m['number']}" if m.get("number") else ""
    return f"{m.get('label') or ''} {m['title']}{number} {m.get('document') or ''} {m['text']}"


bm25 = BM25([_search_text(m) for m in meta])

# ---------------- كلام ملخبط

_VOWELS = set("aeiouyàâäéèêëîïôöûüù")


def _looks_like_latin_word(token):
    """كلمة لاتينية معقولة (فيها حروف علة، وبدون 5 حروف ساكنة متتالية)"""
    if not re.fullmatch(r"[a-zà-ÿ]+", token):
        return False
    vowels = sum(ch in _VOWELS for ch in token)
    return vowels / len(token) >= 0.2 and not re.search(r"[^aeiouyàâäéèêëîïôöûüù]{5,}", token)


def unknown_words(text):
    """كلمات السؤال غير الموجودة في محتوى التطبيق (أخطاء إملائية، دارجة، أو لغة أخرى).
    الرموز والأرقام (G50، 192...) لا تُحسب"""
    return [t for t in tokenize(text)
            if t not in bm25.postings and not any(ch.isdigit() for ch in t)]


def is_vague(text):
    """سؤال عام جداً: كلمة واحدة شائعة في مواضيع كثيرة ("الغرامة؟"، "الضريبة؟")،
    فلا يُعرف أي موضوع يقصد المستخدم"""
    tokens = tokenize(text)
    if len(tokens) != 1 or any(ch.isdigit() for ch in tokens[0]):
        return False
    df = len(bm25.postings.get(tokens[0], ()))
    return df >= max(20, int(0.02 * len(meta)))


def is_meaningless(text):
    """سؤال ملخبط مثل "ثهتبثتبخه": لا كلمة فيه معروفة في محتوى التطبيق ولا قريبة من كلمة معروفة"""
    tokens = tokenize(text)
    if not tokens:
        # حروف فقط بدون كلمات (مثل "؟؟" أو "x")
        return len(re.findall(r"[^\W\d_]", text)) < 2 and not re.search(r"\d", text)
    for t in tokens:
        if any(ch.isdigit() for ch in t):  # رمز أو رقم (G50، 192...)
            return False
        if t in bm25.postings or bm25._closest(t) or _looks_like_latin_word(t):
            return False
    return True

# ---------------- البحث

_RRF_K = 60  # ثابت الدمج المعتاد
_CANDIDATES = 40  # مرشحون من كل طريقة بحث
_PER_TYPE = 3  # حصة مضمونة لكل قسم من التطبيق بين المرشحين


def _content(m):
    """النص الذي يقرأه المرتّب والنموذج: الجواب كاملاً إن وُجد، وإلا المقطع"""
    return m.get("full") or m["text"]


def _key(m):
    return m["type"], m["id"]


def retrieve(questions, k=config.TOP_K, rerank_query=None, rerank=True):
    """يرجع [(مقطع, درجة التشابه بالمعنى)] الأنسب للسؤال.
    [questions]: سؤال واحد أو عدة صياغات لنفس السؤال (الأصلي، المفهوم، المعاد صياغته).
    [rerank_query]: السؤال الذي يحكم به المرتّب (الأوضح)، افتراضياً أول صياغة.
    [rerank]: False للبحث السريع (بدون نموذج إعادة الترتيب)."""
    if isinstance(questions, str):
        questions = [questions]
    # نموذج e5 يتطلب البادئة "query: " للأسئلة
    qs = embedder.encode([f"query: {q}" for q in questions], normalize_embeddings=True)
    all_dense = vectors @ np.asarray(qs).T  # (عدد المقاطع، عدد الصياغات)
    dense = all_dense.max(axis=1)  # أحسن درجة لكل مقطع بين كل الصياغات

    # 1) الدمج: المعنى + الكلمات لكل صياغة
    fused = defaultdict(float)
    for col in range(all_dense.shape[1]):
        for rank, i in enumerate(np.argsort(-all_dense[:, col])[:_CANDIDATES]):
            fused[int(i)] += 1 / (_RRF_K + rank + 1)
        lexical = bm25.scores(questions[col])
        for rank, (i, _) in enumerate(sorted(lexical.items(), key=lambda x: -x[1])[:_CANDIDATES]):
            fused[i] += 1 / (_RRF_K + rank + 1)

    # تطابق تام لرمز أو رقم (رقم المادة، G12...) في عنوان المصدر: دفعة إضافية
    codes = {t for q in questions for t in tokenize(q) if any(ch.isdigit() for ch in t)}
    if codes:
        for i in fused:
            m = meta[i]
            own = set(tokenize(m["title"])) | ({_normalize(m["number"])} if m.get("number") else set())
            if codes & own:
                fused[i] += 1 / _RRF_K

    ranked = sorted(fused, key=lambda i: -fused[i])

    # 2) المرشحون: الأعلى عموماً + حصة لكل قسم (المؤسسات، الأسئلة، الأنظمة...)
    #    الحصة تؤخذ من ترتيب كل المحتوى بالمعنى، حتى القسم الذي لم يظهر في الدمج
    candidates = ranked[:_CANDIDATES]
    chosen = set(candidates)
    per_type = Counter(meta[i]["type"] for i in candidates)
    for i in ranked[_CANDIDATES:] + [int(j) for j in np.argsort(-dense)]:
        t = meta[i]["type"]
        if per_type[t] < _PER_TYPE and i not in chosen:
            candidates.append(i)
            chosen.add(i)
            per_type[t] += 1

    # 3) إعادة الترتيب: هل هذا المصدر يجيب فعلاً عن السؤال؟
    if rerank and reranker is not None and candidates:
        query = rerank_query or questions[0]
        pairs = [(query, f"{meta[i]['title']}\n{_content(meta[i])[:1500]}") for i in candidates]
        scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)
        order = [i for _, i in sorted(zip(scores, candidates), key=lambda x: -float(x[0]))]
        rerank_score = {i: float(s) for s, i in zip(scores, candidates)}
    else:
        order, rerank_score = candidates, {}

    # 4) الأفضل، بحد أقصى مقطعين من نفس العنصر (ومقطع واحد إذا كان الجواب كاملاً)
    best, used = [], Counter()
    for i in order:
        m = meta[i]
        limit = 1 if m.get("full") else 2
        if used[_key(m)] >= limit:
            continue
        used[_key(m)] += 1
        best.append((dict(m, rerank=rerank_score.get(i)), float(dense[i])))
        if len(best) >= k:
            break
    return best


def format_sources(results):
    lines = []
    for n, (m, _) in enumerate(results, 1):
        label = m.get("label") or TYPE_LABELS.get(m["type"], "")
        ref = f"{label}: {m['title']}"
        if m["number"]:
            ref = f"المادة {m['number']} — {m['title']}"
        if m["document"]:
            ref += f" ({m['document']})"
        lines.append(f"[{n}] {ref}\n{_content(m)}")
    return "\n\n".join(lines)


def unique_sources(results):
    """نفس المادة قد تظهر في عدة مقاطع؛ نرجعها مرة واحدة"""
    seen, sources = set(), []
    for m, score in results:
        key = _key(m)
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "type": m["type"], "id": m["id"], "title": m["title"],
            "number": m["number"], "document": m["document"],
            "label": m.get("label") or TYPE_LABELS.get(m["type"], ""),
            # نص المصدر: يعرضه التطبيق عند الضغط على المصدر
            "content": _content(m)[:1500],
            "score": round(score, 3),
        })
    return sources
