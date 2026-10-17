"""الكلام العادي في المحادثة (تحية، شكر، "حسناً"، وداع، "من أنت"...): رد جاهز فوري بدون النموذج.

الكلمات بالفصحى والدارجة الجزائرية والإنجليزية والفرنسية. التطابق يتسامح مع:
التشكيل والتطويل، أشكال الهمزة، ة/ه، ى/ي، الحروف المكررة ("شكراااا")، علامات الترقيم،
والأخطاء الإملائية ("حسننا"، "شكرن"، "السلم عليكم").
"""
import difflib
import itertools
import re

# ---------------- الكلمات (تُكتب بأي شكل؛ تُطبَّع تلقائياً)

PHRASES = {
    # "من أنت؟"
    "identity": [
        # فصحى
        "من انت", "من أنت", "من تكون", "ما اسمك", "ما هو اسمك", "ما اسمك يا", "عرف بنفسك",
        "عرفني بنفسك", "قدم نفسك", "من صنعك", "من برمجك", "من طورك", "من طوّرك", "من صممك", "من انشاك", "من أنشأك",
        "من وراءك", "من مطورك", "ما هو نموذجك", "ما نموذجك", "اي نموذج انت", "على ماذا تعتمد",
        "ما هي التقنية", "كم عمرك", "اين تعيش", "هل تفكر", "هل لديك مشاعر", "هل انت انسان", "هل انت روبوت",
        "هل انت بوت", "هل انت ذكاء اصطناعي", "انت روبوت", "انت انسان", "انت بوت",
        # دارجة
        "مين انت", "شكون انت", "شكون نتا", "شكون نتي", "شكون راك", "شكون انتا", "واش سميتك",
        "واش اسمك", "كيفاش سميتك", "اش سميتك", "شنو سميتك", "شنو اسمك", "شكون صنعك", "شكون دارك", "شكون طورك", "شكون برمجك", "شكون خدمك", "واش من موديل",
        "واش من نموذج", "شحال عمرك", "وين ساكن",
        "انت عبد ولا روبو", "نتا روبو", "نتا بوت", "انت روبو",
        # English
        "who are you", "what is your name", "what s your name", "whats your name", "your name",
        "introduce yourself", "who made you", "who created you", "are you a bot", "are you human",
        "are you a robot", "are you ai", "are you real", "who developed you", "who built you",
        "what model are you", "which model", "how old are you", "are you chatgpt", "are you gpt",
        # Français
        "qui es tu", "qui etes vous", "qui êtes vous", "comment tu t appelles", "comment vous appelez vous",
        "ton nom", "votre nom", "tu es qui", "t es qui", "es tu un robot", "presente toi", "présente toi",
    ],
    # "ماذا تستطيع أن تفعل؟"
    "capabilities": [
        # فصحى
        "ماذا تستطيع", "ماذا يمكنك", "ماذا تستطيع ان تفعل", "ماذا يمكنك ان تفعل",
        "ماذا تستطيع أن تفعل", "ماذا يمكنك أن تفعل", "ما الذي تستطيع فعله", "ما الذي يمكنك فعله",
        "ماذا تقدم", "ماذا تقدم لي", "ما الذي تستطيع", "ما الذي يمكنك", "ماذا تفعل", "ما عملك",
        "ما وظيفتك", "ما هي وظيفتك", "ما خدماتك", "ما هي خدماتك", "بماذا تساعدني", "فيما تساعدني",
        "كيف تساعدني", "كيف يمكنك مساعدتي", "ما الذي تعرفه", "ماذا تعرف", "ما فائدتك", "كيف استعملك",
        "كيف استخدمك", "ما هي قدراتك",
        # دارجة
        "واش تقدر", "واش تقدر دير", "واش تعرف", "واش تعرف دير", "واش تدير", "واش تخدم", "واش خدمتك",
        "فاش تعاوني", "فاش تعاونني", "كيفاش تعاوني", "كيفاش تعاونني", "واش تعاوني", "شنو تقدر دير",
        "اش تقدر دير", "شنو كتدير", "كيفاش نستعملك", "كيفاش نخدم بيك",
        # English
        "what can you do", "how can you help", "what do you do", "what are you able", "what are you for",
        "how do i use you", "what can i ask", "what do you know", "your features", "help me with what",
        # Français
        "que peux tu faire", "que pouvez vous faire", "comment peux tu m aider", "comment pouvez vous m aider",
        "tu sers a quoi", "a quoi tu sers", "que sais tu faire",
    ],
    # "كيف حالك؟"
    "how_are_you": [
        # فصحى
        "كيف حالك", "كيف الحال", "كيف حالكم", "كيفك", "كيف انت", "كيف اصبحت", "كيف امسيت",
        "هل انت بخير", "عساك بخير", "انشالله بخير", "ان شاء الله بخير",
        # دارجة
        "واش راك", "واش راكي", "واش راكم", "لاباس", "لا باس", "لاباس عليك", "واش الاحوال", "واش الحال",
        "كي راك", "كيراك", "كي راكي", "كيداير", "كي داير", "كيف داير", "شخبارك", "شو اخبارك",
        "اش خبارك", "واش خبارك", "واش اخبارك", "لباس", "لاباس عليك", "راك مليح", "راك لاباس",
        "كيداير لاباس", "شلونك", "اشحالك",
        # English
        "how are you", "how r u", "how are u", "how is it going", "how s it going", "how are things",
        "how do you do", "what s up", "whats up", "sup", "are you ok", "are you fine",
        # Français
        "ca va", "ça va", "comment ca va", "comment ça va", "comment allez vous", "comment vas tu",
        "tu vas bien", "vous allez bien", "quoi de neuf", "la forme",
    ],
    # "عندي سؤال" / "ساعدني"
    "help": [
        # فصحى
        "ساعدني", "اريد مساعدة", "أريد مساعدة", "احتاج مساعدة", "ممكن مساعدة", "هل يمكنك مساعدتي",
        "اريد ان اسال", "أريد أن أسأل", "عندي سؤال", "لدي سؤال", "عندي استفسار", "لدي استفسار",
        "سؤال", "استفسار", "عندي مشكلة", "لدي مشكلة", "اريد معلومة", "ابحث عن معلومة",
        # دارجة
        "عاوني", "عاونّي", "عاونني", "ساعفني", "حاب نسقسي", "نحب نسقسي", "بغيت نسقسي", "بغيت نسول",
        "نسقسيك", "نسقسي", "حاب نسقسيك", "عندي مشكل", "حاب نعرف", "نحب نعرف", "بغيت نعرف", "حاب معلومة",
        "تقدر تعاوني", "تقدر تعاونني", "ممكن تعاوني",
        # English
        "i have a question", "help me", "help", "can you help", "can you help me", "i need help",
        "need help", "question", "i want to ask", "can i ask", "i have a problem",
        # Français
        "aide moi", "aidez moi", "j ai une question", "une question", "j ai besoin d aide", "besoin d aide",
        "pouvez vous m aider", "tu peux m aider",
    ],
    # "مع السلامة"
    "farewell": [
        # فصحى
        "مع السلامه", "مع السلامة", "في امان الله", "في أمان الله", "وداعا", "وداعاً", "الى اللقاء",
        "إلى اللقاء", "الى لقاء", "تصبح على خير", "تصبحون على خير", "ليلة سعيدة", "يوما سعيدا",
        "اراك لاحقا", "نلتقي لاحقا", "استودعك الله",
        # دارجة
        "بسلامه", "بالسلامه", "بالسلامة", "باي", "باي باي", "نشوفك", "نشوفكم", "نتلاقاو", "نتلاقاو من بعد",
        "يالله باي", "اسمحلي", "سمحلي", "نروح", "رايح", "تهلا", "تهلاو", "الله يعاونك", "ربي يعاونك",
        "بالتوفيق", "تصبح على خير خويا",
        # English
        "bye", "bye bye", "goodbye", "good bye", "see you", "see you later", "see ya", "good night",
        "take care", "have a nice day", "have a good day", "later", "cya",
        # Français
        "au revoir", "a bientot", "à bientôt", "a plus", "à plus", "bonne journee", "bonne journée",
        "bonne soiree", "bonne soirée", "bonne nuit", "salut a plus", "a demain", "à demain", "ciao", "tchao",
    ],
    # "شكراً"
    "thanks": [
        # فصحى
        "شكرا", "شكراً", "شكرا جزيلا", "شكرا لك", "شكرا لكم", "اشكرك", "أشكرك", "نشكرك", "مشكور",
        "مشكورين", "بارك الله فيك", "بارك الله فيكم", "جزاك الله خيرا", "جزاك الله خير", "جزاكم الله خيرا",
        "احسنت", "أحسنت", "اكرمك الله", "لك جزيل الشكر", "شكرا على المساعدة", "شكرا على الجواب",
        "ممتن", "انا ممتن",
        # دارجة
        "يعطيك الصحه", "يعطيك الصحة", "يعطيكم الصحة", "صحيت", "صحيتي", "صحيتو", "صحا", "صحة", "صحا خويا",
        "الله يحفظك", "ربي يحفظك", "ربي يبارك", "ربي يبارك فيك", "الله يبارك فيك", "ربي يجازيك",
        "الله يجازيك", "يسلمو", "تسلم", "تسلم ايدك", "مرسي", "ميرسي", "شكرا بزاف", "بارك الله",
        "الله يعطيك الصحة", "ربي يخليك",
        # English
        "thanks", "thank you", "thank u", "thx", "ty", "many thanks", "thanks a lot", "thank you so much",
        "appreciate it", "much appreciated",
        # Français
        "merci", "merci beaucoup", "merci bien", "je vous remercie", "je te remercie", "mille merci",
        "mille mercis", "c est gentil",
    ],
    # "السلام عليكم"
    "greeting": [
        # فصحى
        "السلام عليكم", "السلام عليكم ورحمة الله", "السلام", "سلام", "سلام عليكم", "عليكم السلام",
        "وعليكم السلام", "مرحبا", "مرحباً", "مرحبتين", "اهلا", "أهلاً", "اهلين", "اهلا وسهلا",
        "أهلا وسهلا", "هلا", "هلا والله", "صباح الخير", "صباح النور", "مساء الخير", "مساء النور",
        "طاب يومك", "طاب صباحك", "طاب مساؤك", "تحية طيبة", "تحياتي",
        # دارجة
        "صباحو", "صباح الخير خويا", "سلام خويا", "سلام عليكم خويا", "صحة صباحك", "مساء الخير خويا",
        "اهلا خويا", "ازول", "أزول", "ونعم", "يا هلا",
        # English
        "salam", "slm", "assalamu alaikum", "salam alaikum", "hello", "hi", "hey", "hii", "hello there",
        "hey there", "good morning", "good evening", "good afternoon", "greetings", "yo",
        # Français
        "bonjour", "bonsoir", "salut", "coucou", "bonjour a vous", "salut a toi", "bjr", "bsr", "slt",
    ],
    # "حسناً"، "تمام"، "فهمت"، "نعم"...
    "ack": [
        # فصحى
        "حسنا", "حسناً", "حسن", "طيب", "تمام", "تمام التمام", "مفهوم", "فهمت", "فهمتك", "واضح",
        "وصلت", "وصلت الفكرة", "نعم", "اجل", "أجل", "بلى", "اكيد", "أكيد", "طبعا", "بالتأكيد",
        "ممتاز", "رائع", "جيد", "جيد جدا", "بخير", "انا بخير", "لاباس الحمد لله", "جميل", "عظيم", "مذهل", "صح", "صحيح", "صحيح جدا",
        "معك حق", "لا", "لا شكرا", "لا بأس", "لا داعي", "كفى", "انتهى", "حسنا فهمت", "نعم فهمت",
        "موافق", "اوافق", "اتفق", "حاضر", "ان شاء الله", "إن شاء الله", "انشالله", "ماشاء الله",
        "سبحان الله", "الحمد لله", "الحمدلله", "ههه", "هههه", "هاها", "لول",
        # دارجة
        "اوكي", "أوكي", "اوك", "اوكاي", "ايه", "إيه", "اه", "آه", "ايوه", "أيوه", "واه", "إيه نعم",
        "باهي", "باهية", "مليح", "مليحة", "شباب", "صحيح خويا", "معليش", "ماشي مشكل", "ماكاش مشكل",
        "ولا حاجة", "خلاص", "صافي", "صافي خلاص", "زوين", "مزيان", "واخا", "وخا", "يزي", "برك", "ماشي",
        "لالا", "لا لا", "فهمت خويا", "راني فاهم", "فهمتها", "دبا", "هكا", "هكاك",
        # English
        "ok", "okay", "okey", "k", "kk", "yes", "yeah", "yep", "yup", "sure", "of course", "no", "nope",
        "nah", "alright", "all right", "fine", "cool", "great", "nice", "good", "perfect", "awesome",
        "excellent", "got it", "understood", "i see", "i understand", "makes sense", "right", "correct",
        "agreed", "lol", "haha", "wow",
        # Français
        "oui", "non", "d accord", "daccord", "dac", "ok merci", "tres bien", "très bien", "parfait",
        "super", "genial", "génial", "compris", "entendu", "bien", "c est bon", "ça marche", "ca marche",
        "exact", "exactement", "bien sur", "bien sûr", "pas de souci", "pas de probleme",
    ],
}

# أولوية التصنيف عند وجود أكثر من نوع ("السلام عليكم كيف حالك" ← كيف حالك)
ORDER = ["identity", "capabilities", "how_are_you", "help", "farewell", "thanks", "greeting", "ack"]

# ---------------- الردود (بالفصحى)

REPLIES = {
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
    "ack": {
        "ar": "حسناً! إذا كان لديك سؤال آخر عن الضرائب أو التصريحات فأنا في الخدمة. 😊",
        "en": "Alright! If you have another question about taxes or declarations, I'm here. 😊",
        "fr": "D'accord ! Si vous avez une autre question sur la fiscalité, je suis là. 😊",
    },
    "farewell": {
        "ar": "مع السلامة! سعدتُ بمساعدتك، وأنا هنا متى احتجت إليّ. 👋",
        "en": "Goodbye! It was a pleasure helping you. I'm here whenever you need me. 👋",
        "fr": "Au revoir ! Ce fut un plaisir de vous aider. Je suis là quand vous voulez. 👋",
    },
    "how_are_you": {
        "ar": "بخير والحمد لله، شكراً لسؤالك! 😊 كيف يمكنني مساعدتك في الجباية اليوم؟",
        "en": "I'm doing well, thanks for asking! 😊 How can I help you with taxes today?",
        "fr": "Je vais bien, merci ! 😊 Comment puis-je vous aider en fiscalité aujourd'hui ?",
    },
    # أسئلة عن البوت نفسه (من أنت، من طوّرك، ماذا تستطيع...): لا نجيب عنها، نذكّر بالدور فقط
    "identity": {
        "ar": "أنا مساعد جبائي فقط 😊 اطرح سؤالك عن الضرائب أو التصريحات أو الأنظمة الجبائية وسأساعدك.",
        "en": "I'm only a tax assistant 😊 Ask me about taxes, declarations or tax regimes and I'll help.",
        "fr": "Je suis uniquement un assistant fiscal 😊 Posez-moi une question sur les impôts ou les déclarations.",
    },
    "capabilities": {
        "ar": "أنا مساعد جبائي فقط 😊 اطرح سؤالك عن الضرائب أو التصريحات أو الأنظمة الجبائية وسأساعدك.",
        "en": "I'm only a tax assistant 😊 Ask me about taxes, declarations or tax regimes and I'll help.",
        "fr": "Je suis uniquement un assistant fiscal 😊 Posez-moi une question sur les impôts ou les déclarations.",
    },
    "help": {
        "ar": "تفضّل، اطرح سؤالك عن الجباية وسأحاول مساعدتك. 😊",
        "en": "Go ahead, ask your tax question and I'll do my best to help. 😊",
        "fr": "Allez-y, posez votre question fiscale et je ferai de mon mieux. 😊",
    },
}

# كلام ملخبط ("عالعاخهاه"، "uiohihi"): عدة صياغات حتى لا يتكرر نفس الرد
GIBBERISH_REPLIES = {
    "ar": [
        "لم أفهم ما كتبته 🤔 هل يمكنك إعادة كتابة سؤالك بوضوح؟",
        "يبدو أن الرسالة غير واضحة 🙂 اكتب سؤالك عن الضرائب أو التصريحات وسأساعدك.",
        "عذراً، لم أتمكن من فهم رسالتك. حاول صياغة سؤالك بكلمات أخرى.",
        "لم تصلني الرسالة بشكل مفهوم 😅 أعد كتابتها من فضلك، وأنا في الخدمة.",
    ],
    "en": [
        "I didn't understand what you wrote 🤔 Could you rewrite your question clearly?",
        "That message isn't clear to me 🙂 Ask me about taxes or declarations and I'll help.",
        "Sorry, I couldn't understand your message. Try phrasing your question differently.",
    ],
    "fr": [
        "Je n'ai pas compris votre message 🤔 Pouvez-vous reformuler votre question ?",
        "Votre message n'est pas clair 🙂 Posez-moi une question sur la fiscalité et je vous aiderai.",
        "Désolé, je n'ai pas compris. Essayez de formuler votre question autrement.",
    ],
}

# ---------------- التطبيع والتطابق


def normalize(text):
    text = str(text or "").lower()
    text = re.sub(r"[ً-ْـ]", "", text)  # التشكيل والتطويل
    text = re.sub(r"[إأآا]", "ا", text)
    text = text.replace("ة", "ه").replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    text = re.sub(r"(.)\1{2,}", r"\1", text)  # "شكراااا" ← "شكرا"
    text = re.sub(r"[^\w\s]", " ", text)  # علامات الترقيم والرموز التعبيرية
    return re.sub(r"\s+", " ", text).strip()


def _pattern(phrases):
    words = sorted({normalize(p) for p in phrases if normalize(p)}, key=len, reverse=True)
    # (?<!\S) و (?!\S): حدود الكلمة بدون استهلاك المسافة، حتى تُلتقط الكلمات المتتالية ("ok ok")
    return re.compile(r"(?<!\S)(?:" + "|".join(re.escape(w) for w in words) + r")(?!\S)")


_PATTERNS = {kind: _pattern(phrases) for kind, phrases in PHRASES.items()}

# كلمات تُتجاهل عند الحكم بأن الرسالة كلام عادي فقط ("حسنا يا شافي"، "شكرا ChafiBot")
_FILLER = {normalize(w) for w in [
    "يا", "و", "لك", "ليك", "بزاف", "كثيرا", "جزيلا", "خويا", "اختي", "اخي", "صديقي", "عليكم",
    "chafibot", "chafi", "شافي", "bot", "بوت", "very", "much", "so", "a", "lot", "you", "too", "beaucoup",
    "bien", "the", "my", "friend", "اليوم", "الحمد", "لله", "الله", "ورحمه", "وبركاته",
]}


# كل كلمات العبارات، لتصحيح الأخطاء الإملائية ("حسننا" ← "حسنا")
_VOCAB = sorted({w for phrases in PHRASES.values() for p in phrases for w in normalize(p).split()})
_VOCAB_SET = set(_VOCAB)


def _candidates(word):
    """الكلمة نفسها إن كانت معروفة، وإلا أقرب الكلمات إليها (حتى 3) ثم الكلمة كما هي"""
    if word in _VOCAB_SET or word in _FILLER or len(word) < 2 or word.isdigit():
        return [word]
    return difflib.get_close_matches(word, _VOCAB, n=3, cutoff=0.75) + [word]


def kind_of(text, max_words=7):
    """نوع الكلام العادي، أو None إذا كانت الرسالة سؤالاً حقيقياً.
    يجرّب النص كما هو، ثم بتصحيحات الأخطاء الإملائية ("حسننا" ← "حسنا")"""
    norm = normalize(text)
    if not norm or len(norm.split()) > max_words:
        return None
    found = _kind(norm)
    if found is not None:
        return found
    # كل تركيبة من التصحيحات المحتملة (الأقرب أولاً)، بحد أقصى معقول
    options = [_candidates(w) for w in norm.split()]
    for i, combo in enumerate(itertools.product(*options)):
        if i >= 50:
            break
        candidate = " ".join(combo)
        if candidate != norm and (found := _kind(candidate)):
            return found
    return None


def _kind(norm):
    found = None
    for kind in ORDER:
        if _PATTERNS[kind].search(norm):
            found = kind
            break
    if not found:
        return None
    # الرسالة كلام عادي فقط إذا لم يبقَ بعد حذف عباراته إلا كلمات حشو
    rest = norm
    for pattern in _PATTERNS.values():
        rest = pattern.sub(" ", rest)
    leftover = [w for w in rest.split() if w not in _FILLER]
    return found if not leftover else None
