"""يقرأ محتوى Chafi من قاعدة البيانات ويحوّله إلى vectors للبحث بالمعنى.
أعد تشغيله كلما تغيّر المحتوى:  python build_index.py

المصادر: المواد، الأنظمة الجبائية، المؤسسات، differents (الأسئلة الشائعة، القاموس،
ضرائب ورسوم أخرى...)، المواعيد والالتزامات، الأنشطة، والمقالات.
كل عنصر يُفهرس بالعربية وبالفرنسية/الإنجليزية إن وُجدت، حتى يجد البحث
الأسئلة المكتوبة بأي لغة.
"""
import html
import json
import os
import re

import numpy as np
import pymysql
from sentence_transformers import SentenceTransformer

import config

CHUNK_SIZE = config.CHUNK_SIZE  # حروف لكل مقطع
OVERLAP = config.CHUNK_OVERLAP

# أنواع لها جواب جاهز: إذا طابق السؤالُ عنوانَها يُرجع الجواب مباشرة بدون النموذج
DIRECT_DIFFERENT_TYPES = {1, 4}  # سؤال شائع، القاموس الجبائي

# أنواع جدول differents (type) كما يستعملها التطبيق
DIFFERENT_TYPES = {
    1: "سؤال شائع",
    3: "ضرائب ورسوم أخرى",
    4: "القاموس الجبائي",
}


def clean(text):
    """إزالة وسوم HTML والتطويل "ـ" والمسافات الزائدة"""
    text = html.unescape(re.sub(r"<[^>]+>", " ", str(text or "")))
    text = text.replace("ـ", "")  # "تصريــــح" ← "تصريح"
    return re.sub(r"\s+", " ", text).strip()


def chunks(text):
    if len(text) <= CHUNK_SIZE:
        return [text]
    parts, start = [], 0
    while start < len(text):
        parts.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - OVERLAP
    return parts


def join(*parts):
    return "\n".join(p for p in (clean(x) for x in parts) if p)


def item(kind, id_, title, text, *, lang="ar", number=None, document=None, label=None,
         answer=None):
    return {
        "type": kind, "id": id_, "title": clean(title), "text": text, "lang": lang,
        "number": number, "document": document, "label": label,
        # الجواب الجاهز (للأسئلة الشائعة والقاموس والمواعيد)
        "answer": answer,
    }


def load_sources(cursor):
    """كل مصدر في دالة مستقلة: إذا فشل جدول (عمود ناقص مثلاً) تكمل البقية"""

    def fetch(sql):
        cursor.execute(sql)
        return cursor.fetchall()

    def articles():
        out = []
        for r in fetch("""
            SELECT a.id, a.number, a.label, a.label_en, a.text, a.text_en, a.is_repealed,
                   d.title_ar AS document
            FROM articles a LEFT JOIN documents d ON d.id = a.document_id
        """):
            if r["is_repealed"]:
                continue
            title = r["label"] or f"المادة {r['number'] or ''}"
            out.append(item("article", r["id"], title, join(r["text"]),
                            number=r["number"], document=r["document"], label="مادة"))
            if clean(r["text_en"]):
                out.append(item("article", r["id"], r["label_en"] or title, join(r["text_en"]),
                                lang="en", number=r["number"], document=r["document"], label="Article"))
        return out

    def title_body(table, kind, label, extra_where=""):
        out = []
        for r in fetch(f"SELECT id, title, body, title_fr, body_fr FROM {table} {extra_where}"):
            out.append(item(kind, r["id"], r["title"], join(r["title"], r["body"]), label=label))
            if clean(r["body_fr"]) or clean(r["title_fr"]):
                out.append(item(kind, r["id"], r["title_fr"] or r["title"],
                                join(r["title_fr"], r["body_fr"]), lang="fr", label=label))
        return out

    def taxs_and_apps():
        """جدول taxs_and_apps قسمان حسب categories.type_cat:
        1 = الأنظمة الجبائية، 2 = العقوبات والجزاءات الجبائية"""
        out = []
        for r in fetch("""
            SELECT t.id, t.title, t.body, t.title_fr, t.body_fr,
                   c.type_cat, c.name AS category, c.name_fr AS category_fr
            FROM taxs_and_apps t LEFT JOIN categories c ON c.id = t.cat_id
        """):
            if r["type_cat"] == 2:
                kind, label, label_fr = "penalty", "جزاءات", "Sanctions"
            else:
                kind, label, label_fr = "tax_system", "نظام جبائي", "Régime fiscal"
            out.append(item(kind, r["id"], r["title"],
                            join(f"{label}: {clean(r['category'])}" if r["category"] else "",
                                 r["title"], r["body"]),
                            label=label))
            if clean(r["body_fr"]) or clean(r["title_fr"]):
                out.append(item(kind, r["id"], r["title_fr"] or r["title"],
                                join(f"{label_fr}: {clean(r['category_fr'])}" if r["category_fr"] else "",
                                     r["title_fr"], r["body_fr"]),
                                lang="fr", label=label_fr))
        return out

    def differents():
        out = []
        for r in fetch("SELECT id, type, title, body, title_fr, body_fr FROM differents"):
            label = DIFFERENT_TYPES.get(r["type"], "مواضيع جبائية أخرى")
            direct = r["type"] in DIRECT_DIFFERENT_TYPES
            out.append(item("different", r["id"], r["title"], join(r["title"], r["body"]), label=label,
                            answer=clean(r["body"]) if direct else None))
            if clean(r["body_fr"]) or clean(r["title_fr"]):
                out.append(item("different", r["id"], r["title_fr"] or r["title"],
                                join(r["title_fr"], r["body_fr"]), lang="fr", label=label,
                                answer=clean(r["body_fr"]) if direct else None))
        return out

    def appointments():
        dates = {}
        for r in fetch("SELECT appointment_id, appointment_date FROM appointment_dates ORDER BY appointment_date"):
            dates.setdefault(r["appointment_id"], []).append(str(r["appointment_date"]))
        out = []
        for r in fetch("""SELECT id, declaration, declaration_fr, noticeDate, deadline,
                                 dependencies, dependencies_fr FROM appointments"""):
            when = ", ".join(dates.get(r["id"], []))
            ar = join(
                f"التصريح/الالتزام: {clean(r['declaration'])}",
                f"تاريخ الإشعار: {clean(r['noticeDate'])}" if clean(r["noticeDate"]) else "",
                f"آخر أجل: {clean(r['deadline'])}" if clean(r["deadline"]) else "",
                f"التواريخ: {when}" if when else "",
                f"الوثائق/التبعات: {clean(r['dependencies'])}" if clean(r["dependencies"]) else "",
            )
            out.append(item("appointment", r["id"], r["declaration"], ar, label="موعد والتزام",
                            answer=ar))
            if clean(r["declaration_fr"]):
                fr = join(
                    f"Déclaration / obligation: {clean(r['declaration_fr'])}",
                    f"Échéance: {clean(r['deadline'])}" if clean(r["deadline"]) else "",
                    f"Dates: {when}" if when else "",
                    clean(r["dependencies_fr"]),
                )
                out.append(item("appointment", r["id"], r["declaration_fr"], fr, lang="fr",
                                label="Échéance fiscale", answer=fr))
        return out

    def activities():
        out = []
        for r in fetch("""
            SELECT a.id, a.name, a.name_fr, a.body, a.body_fr, a.code_activity,
                   n.name AS nature, n.name_fr AS nature_fr
            FROM activitys a LEFT JOIN nataire_activitys n ON n.id = a.nataire_activitys_id
            WHERE COALESCE(a.is_delete, 0) = 0
        """):
            code = f" ({r['code_activity']})" if r["code_activity"] else ""
            out.append(item("activity", r["id"], f"{clean(r['name'])}{code}",
                            join(f"النشاط: {clean(r['name'])}{code}",
                                 f"طبيعة النشاط: {clean(r['nature'])}" if r["nature"] else "",
                                 r["body"]),
                            label="نشاط"))
            if clean(r["name_fr"]):
                out.append(item("activity", r["id"], f"{clean(r['name_fr'])}{code}",
                                join(f"Activité: {clean(r['name_fr'])}{code}",
                                     f"Nature: {clean(r['nature_fr'])}" if r["nature_fr"] else "",
                                     r["body_fr"]),
                                lang="fr", label="Activité"))
        return out

    def posts():
        out = []
        for r in fetch("""SELECT id, title, title2, body, chafi_advice, legal_source,
                                 title_fr, title2_fr, body_fr, chafi_advice_fr, legal_source_fr
                          FROM posts WHERE type = 1"""):
            out.append(item("post", r["id"], r["title"],
                            join(r["title"], r["title2"], r["body"],
                                 f"نصيحة شافي: {clean(r['chafi_advice'])}" if clean(r["chafi_advice"]) else "",
                                 f"المصدر القانوني: {clean(r['legal_source'])}" if clean(r["legal_source"]) else ""),
                            label="مقال"))
            if clean(r["body_fr"]):
                out.append(item("post", r["id"], r["title_fr"] or r["title"],
                                join(r["title_fr"], r["title2_fr"], r["body_fr"], r["chafi_advice_fr"],
                                     r["legal_source_fr"]),
                                lang="fr", label="Article"))
        return out

    sources = {
        "المواد": articles,
        "الأنظمة الجبائية والجزاءات": taxs_and_apps,
        "المؤسسات": lambda: title_body("institutions", "institution", "مؤسسة"),
        "differents (أسئلة، قاموس، ضرائب أخرى...)": differents,
        "المواعيد والالتزامات": appointments,
        "الأنشطة": activities,
        "المقالات": posts,
    }
    items = []
    for name, loader in sources.items():
        try:
            rows = loader()
            items += rows
            print(f"  ✔ {name}: {len(rows)}")
        except Exception as e:  # noqa: BLE001
            print(f"  ✘ {name}: {e}")
    return [i for i in items if i["text"]]


def main():
    conn = pymysql.connect(
        host=config.DB_HOST, port=config.DB_PORT, user=config.DB_USER,
        password=config.DB_PASS, database=config.DB_NAME, charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )
    with conn.cursor() as cursor:
        sources = load_sources(cursor)
    conn.close()
    print(f"المصادر: {len(sources)}")

    passages, meta = [], []
    for s in sources:
        # المحتوى غير المواد جواب متكامل: يُعطى للنموذج كاملاً لا مقطعاً منه
        if s["type"] != "article":
            s = {**s, "full": s["text"][:config.FULL_TEXT_MAX]}
        for part in chunks(s["text"]):
            header = s["title"] + (f" — {s['document']}" if s["document"] else "")
            # نموذج e5 يتطلب البادئة "passage: " للنصوص
            passages.append(f"passage: {s['label'] or ''}: {header}\n{part}")
            meta.append({**s, "text": part, "direct": False})
        # العنوان وحده (السؤال الشائع / المصطلح / التصريح) يُقارن بسؤال المستخدم مباشرة:
        # سؤال مقابل سؤال، فتكون الدرجة عالية عند التطابق
        if s["answer"] and s["title"]:
            passages.append(f"query: {s['title']}")
            meta.append({**s, "text": s["answer"], "direct": True})
    print(f"المقاطع: {len(passages)}")

    model = SentenceTransformer(config.EMBED_MODEL)
    vectors = model.encode(passages, batch_size=32, normalize_embeddings=True,
                           show_progress_bar=True)

    os.makedirs(config.INDEX_DIR, exist_ok=True)
    np.save(os.path.join(config.INDEX_DIR, "vectors.npy"), vectors.astype("float32"))
    with open(os.path.join(config.INDEX_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    print("تم بناء الفهرس ✔")


if __name__ == "__main__":
    main()
