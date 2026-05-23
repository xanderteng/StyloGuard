"""
Stylometric feature extraction — unified with the Kaggle training notebook.

Produces a **52-dimensional** feature vector per text:
  • 24 base features  (counts, ratios, morphological signals)
  • 28 function-word ratios

The feature order, formulas, and stopword set are identical to the notebook so
the fitted ``StandardScaler`` can be applied directly.
"""

from __future__ import annotations

import string

from app.model.preprocess import split_sentences, tokenize_words

# ── Function words (28 items — matches notebook) ────────────────────────────
FUNCTION_WORDS: list[str] = [
    "yang", "dan", "di", "ke", "dari", "dengan", "untuk", "pada",
    "ini", "itu", "tidak", "akan", "juga", "karena", "sebagai",
    "dalam", "adalah", "atau", "oleh", "agar", "bagi", "para",
    "saat", "setelah", "sebelum", "namun", "tetapi", "hingga",
]

# ── Feature name list (24 base + 28 fw = 52 total) ──────────────────────────
BASE_FEATURE_NAMES: list[str] = [
    "word_count",
    "sentence_count",
    "avg_word_length",
    "avg_sentence_length",
    "sentence_length_variance",
    "lexical_diversity",
    "punctuation_density",
    "comma_ratio",
    "period_ratio",
    "question_ratio",
    "exclamation_ratio",
    "semicolon_colon_ratio",
    "dash_ratio",
    "digit_char_ratio",
    "uppercase_ratio",
    "numeric_ratio",
    "stopword_ratio",
    "paragraph_count",
    "avg_paragraph_length",
    "short_word_ratio",
    "long_word_ratio",
    "suffix_nya_ratio",
    "suffix_lah_ratio",
    "suffix_kah_ratio",
]

FEATURE_NAMES: list[str] = BASE_FEATURE_NAMES + [f"fw_{w}" for w in FUNCTION_WORDS]

NUM_STYLOMETRIC_FEATURES: int = len(FEATURE_NAMES)  # 52

# ── Indonesian stopwords (NLTK "indonesian" corpus — Tala 2003) ──────────────
# Hardcoded to avoid an nltk runtime dependency in the API server.
INDONESIAN_STOPWORDS: frozenset[str] = frozenset({
    "ada", "adanya", "adalah", "adapun", "agak", "agaknya", "agar", "akan",
    "akankah", "akhir", "akhiri", "akhirnya", "aku", "akulah", "amat",
    "amatlah", "anda", "andalah", "antar", "antara", "antaranya", "apa",
    "apaan", "apabila", "apakah", "apalagi", "apatah", "artinya", "asal",
    "asalkan", "atas", "atau", "ataukah", "ataupun", "awal", "awalnya",
    "bagai", "bagaikan", "bagaimana", "bagaimanakah", "bagaimanapun", "bagi",
    "bagian", "bahkan", "bahwa", "bahwasanya", "baik", "baiklah", "bakal",
    "bakalan", "balik", "banyak", "bapak", "baru", "bawah", "beberapa",
    "begini", "beginian", "beginikah", "beginilah", "begitu", "begitukah",
    "begitulah", "begitupun", "bekas", "belakang", "belakangan", "belum",
    "belumlah", "benar", "benarkah", "benarlah", "berada", "berakhir",
    "berakhirlah", "berakhirnya", "berapa", "berapakah", "berapalah",
    "berapapun", "berarti", "berawal", "berbagai", "berdatangan", "beri",
    "berikan", "berikut", "berikutnya", "berjumlah", "berkali", "berkenaan",
    "berlainan", "berlalu", "berlangsung", "berlebihan", "bermacam",
    "bermaksud", "bermula", "bersama", "bersiap", "bertanya", "berturut",
    "bertutur", "berupa", "besar", "besok", "betul", "betulkah", "biasa",
    "biasanya", "bila", "bilakah", "bilamana", "bisa", "bisakah", "boleh",
    "bolehkah", "bolehlah", "buat", "bukan", "bukankah", "bukanlah",
    "bukannya", "bulan", "bung", "cara", "caranya", "cukup", "cukupkah",
    "cukuplah", "cuma", "dahulu", "dalam", "dan", "dapat", "dari",
    "daripada", "datang", "dekat", "demi", "demikian", "demikianlah",
    "dengan", "depan", "di", "dia", "diakhiri", "diakhirinya", "dialah",
    "diantara", "diantaranya", "diberi", "diberikan", "diberikannya",
    "dibuat", "dibuatnya", "didapat", "didatangkan", "digunakan",
    "diibaratkan", "diibaratkannya", "diingat", "diingatkan", "diinginkan",
    "dijawab", "dijawabnya", "dijelas", "dijelaskan", "dijelaskannya",
    "dikarenakan", "dikatakan", "dikatakannya", "dikerjakan", "diketahui",
    "diketahuinya", "dikira", "dilakukan", "dilalui", "dilihat", "dimaksud",
    "dimaksudkan", "dimaksudkannya", "dimaksudnya", "diminta", "dimintai",
    "dimisalkan", "dimulai", "dimulailah", "dimulainya", "dimungkinkan",
    "dini", "dipastikan", "diperbuat", "diperbuatnya", "dipergunakan",
    "diperkirakan", "diperlihatkan", "diperlukan", "diperlukannya",
    "dipersoalkan", "dipertanyakan", "dipunyai", "diri", "dirinya",
    "disampaikan", "disebut", "disebutkan", "disebutkannya", "disini",
    "disinilah", "ditambahkan", "ditandaskan", "ditanya", "ditanyai",
    "ditanyakan", "ditegaskan", "ditujukan", "ditunjuk", "ditunjuki",
    "ditunjukkan", "ditunjukkannya", "ditunjuknya", "dituturkan",
    "dituturkannya", "diucapkan", "diucapkannya", "diungkapkan", "dong",
    "dua", "dulu", "empat", "enggak", "enggaknya", "entah", "entahlah",
    "guna", "gunakan", "hal", "hampir", "hanya", "hanyalah", "hari",
    "harus", "haruslah", "harusnya", "hendak", "hendaklah", "hendaknya",
    "hingga", "ia", "ialah", "ibarat", "ibaratkan", "ibaratnya", "ibu",
    "ikut", "ingat", "ingin", "inginkah", "inginkan", "ini", "inikah",
    "inilah", "itu", "itukah", "itulah", "jadi", "jadilah", "jadinya",
    "jangan", "jangankan", "janganlah", "jauh", "jawab", "jawaban",
    "jawabnya", "jelas", "jelaskan", "jelaslah", "jelasnya", "jika",
    "jikalau", "juga", "jumlah", "jumlahnya", "justru", "kala", "kalau",
    "kalaulah", "kalaupun", "kalian", "kami", "kamilah", "kamu", "kamulah",
    "kan", "kapan", "kapankah", "kapanpun", "karena", "karenanya", "kasus",
    "kata", "katakan", "katakanlah", "katanya", "ke", "keadaan", "kebetulan",
    "kecil", "kedua", "keduanya", "keinginan", "kelamaan", "kelihatan",
    "kelihatannya", "kelima", "keluar", "kembali", "kemudian",
    "kemungkinan", "kemungkinannya", "kenapa", "kepada", "kepadanya",
    "kesampaian", "keseluruhan", "keseluruhannya", "keterlaluan", "ketika",
    "khususnya", "kini", "kinilah", "kira", "kiranya", "kita", "kitalah",
    "kok", "kurang", "lagi", "lagian", "lah", "lain", "lainnya", "lalu",
    "lama", "lamanya", "lanjut", "lanjutnya", "lebih", "lewat", "lima",
    "luar", "macam", "maka", "makanya", "makin", "malah", "malahan",
    "mampu", "mampukah", "mana", "manakala", "manalagi", "masa", "masalah",
    "masalahnya", "masih", "masihkah", "masing", "mau", "maupun",
    "melainkan", "melakukan", "melalui", "melihat", "melihatnya", "memang",
    "memastikan", "memberi", "memberikan", "membuat", "memerlukan",
    "memihak", "meminta", "memintakan", "memisalkan", "memperbuat",
    "mempergunakan", "memperkirakan", "memperlihatkan", "mempersiapkan",
    "mempersoalkan", "mempertanyakan", "mempunyai", "memulai",
    "memungkinkan", "menaiki", "menambahkan", "menandaskan", "menanti",
    "menantikan", "menanya", "menanyai", "menanyakan", "mendapat",
    "mendapatkan", "mendatang", "mendatangi", "mendatangkan", "menegaskan",
    "mengakhiri", "mengapa", "mengatakan", "mengatakannya", "mengenai",
    "mengerjakan", "mengetahui", "menggunakan", "menghendaki",
    "mengibaratkan", "mengibaratkannya", "mengingat", "mengingatkan",
    "menginginkan", "mengira", "mengucapkan", "mengucapkannya",
    "mengungkapkan", "menjadi", "menjawab", "menjelaskan", "menuju",
    "menunjuk", "menunjuki", "menunjukkan", "menunjuknya", "menurut",
    "menuturkan", "menyampaikan", "menyangkut", "menyatakan",
    "menyebutkan", "menyeluruh", "menyiapkan", "merasa", "mereka",
    "merekalah", "merupakan", "meski", "meskipun", "minta", "mirip",
    "misal", "misalkan", "misalnya", "mula", "mulai", "mulailah",
    "mulanya", "mungkin", "mungkinkah", "nah", "naik", "namun", "nanti",
    "nantinya", "nyaris", "nyatanya", "oleh", "olehnya", "pada", "padahal",
    "padanya", "pak", "paling", "panjang", "pantas", "para", "pasti",
    "pastilah", "penting", "pentingnya", "per", "percuma", "perlu",
    "perlukah", "perlunya", "pernah", "persoalan", "pertama", "pertanyaan",
    "pertanyakan", "pihak", "pihaknya", "pukul", "pula", "pun", "punya",
    "rasa", "rasanya", "rata", "rupanya", "saat", "saatnya", "saja",
    "sajalah", "saling", "sama", "sambil", "sampai", "sampaikan", "sana",
    "sangat", "sangatlah", "satu", "saya", "sayalah", "se", "sebab",
    "sebabnya", "sebagai", "sebagaimana", "sebagainya", "sebagian",
    "sebaik", "sebaiknya", "sebaliknya", "sebanyak", "sebegini", "sebegitu",
    "sebelum", "sebelumnya", "sebenarnya", "seberapa", "sebesar",
    "sebetulnya", "sebisanya", "sebuah", "sebut", "sebutlah", "sebutnya",
    "secara", "secukupnya", "sedang", "sedangkan", "sedemikian", "sedikit",
    "sedikitnya", "seenaknya", "segala", "segalanya", "segera",
    "seharusnya", "sehingga", "seingat", "sejak", "sejauh", "sejenak",
    "sejumlah", "sekadar", "sekadarnya", "sekali", "sekalian", "sekaligus",
    "sekalipun", "sekarang", "sekaranglah", "sekecil", "seketika",
    "sekiranya", "sekitar", "sekitarnya", "sekurang", "sekurangnya", "sela",
    "selain", "selaku", "selalu", "selama", "selamanya", "selanjutnya",
    "seluruh", "seluruhnya", "semacam", "semakin", "semampu", "semampunya",
    "semasa", "semasih", "semata", "sementara", "semisal", "semisalnya",
    "sempat", "semua", "semuanya", "semula", "sendiri", "sendirian",
    "sendirinya", "seolah", "seperti", "sepertinya", "seperlunya", "sering",
    "seringnya", "serta", "serupa", "sesaat", "sesama", "sesampai",
    "sesegera", "seseorang", "sesuatu", "sesuatunya", "sesudah",
    "sesudahnya", "setelah", "setempat", "setengah", "seterusnya", "setiap",
    "setiba", "setidak", "setidaknya", "setinggi", "seusai", "sewaktu",
    "siap", "siapa", "siapakah", "siapapun", "sini", "sinilah", "soal",
    "soalnya", "suatu", "sudah", "sudahkah", "sudahlah", "supaya", "tadi",
    "tadinya", "tahu", "tahun", "tak", "tampak", "tampaknya", "tandas",
    "tandasnya", "tanpa", "tanya", "tanyakan", "tanyanya", "tapi", "tenang",
    "tengah", "tentang", "tentu", "tentulah", "tentunya", "tepat",
    "terakhir", "terasa", "terbanyak", "terdahulu", "terdapat", "terdiri",
    "terhadap", "terhadapnya", "teringat", "terjadi", "terjadilah",
    "terjadinya", "terkira", "terlalu", "terlebih", "terlihat", "termasuk",
    "ternyata", "tersampaikan", "tersebut", "tersebutlah", "tertentu",
    "tertuju", "terus", "terutama", "tetap", "tetapi", "tiada", "tiadakah",
    "tiadalah", "tidak", "tidakkah", "tidaklah", "tiga", "tinggi", "toh",
    "tunjuk", "turut", "tutur", "tuturnya", "ucap", "ucapnya", "ujar",
    "ujarnya", "umum", "umumnya", "ungkap", "ungkapnya", "untuk", "usah",
    "usai", "waduh", "wah", "wahai", "waktu", "waktunya", "walau",
    "walaupun", "wong", "yaitu", "yakin", "yakni", "yang",
})


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


# ── Main extraction function ────────────────────────────────────────────────

def extract_features(text: str) -> list[float]:
    """Return a 52-element feature vector matching the Kaggle notebook order."""
    raw = str(text)
    words = tokenize_words(raw)
    sentences = split_sentences(raw)
    sentence_lengths = [len(tokenize_words(s)) for s in sentences]
    paragraphs = [p.strip() for p in raw.splitlines() if p.strip()]
    chars = [c for c in raw if not c.isspace()]
    punctuation = [c for c in raw if c in string.punctuation]
    uppercase_chars = [c for c in raw if c.isupper()]
    numeric_tokens = [w for w in words if w.isdigit()]
    stop_count = [w for w in words if w in INDONESIAN_STOPWORDS]

    wc = len(words)
    sc = len(sentences)
    cc = len(chars)
    avg_sent_len = _safe_divide(wc, sc)
    sent_var = _safe_divide(
        sum((x - avg_sent_len) ** 2 for x in sentence_lengths),
        len(sentence_lengths),
    )

    feats: list[float] = [
        float(wc),                                                       # word_count
        float(sc),                                                       # sentence_count
        _safe_divide(sum(len(w) for w in words), wc),                    # avg_word_length
        avg_sent_len,                                                    # avg_sentence_length
        sent_var,                                                        # sentence_length_variance
        _safe_divide(len(set(words)), wc),                               # lexical_diversity
        _safe_divide(len(punctuation), cc),                              # punctuation_density
        _safe_divide(raw.count(","), cc),                                # comma_ratio
        _safe_divide(raw.count("."), cc),                                # period_ratio
        _safe_divide(raw.count("?"), cc),                                # question_ratio
        _safe_divide(raw.count("!"), cc),                                # exclamation_ratio
        _safe_divide(raw.count(";") + raw.count(":"), cc),               # semicolon_colon_ratio
        _safe_divide(raw.count("-"), cc),                                # dash_ratio
        _safe_divide(sum(ch.isdigit() for ch in raw), cc),               # digit_char_ratio
        _safe_divide(len(uppercase_chars), cc),                          # uppercase_ratio
        _safe_divide(len(numeric_tokens), wc),                           # numeric_ratio
        _safe_divide(len(stop_count), wc),                               # stopword_ratio
        float(len(paragraphs) or 1),                                     # paragraph_count
        _safe_divide(
            sum(len(tokenize_words(p)) for p in paragraphs),
            len(paragraphs) or 1,
        ),                                                               # avg_paragraph_length
        _safe_divide(sum(1 for w in words if len(w) <= 3), wc),          # short_word_ratio
        _safe_divide(sum(1 for w in words if len(w) >= 8), wc),          # long_word_ratio
        _safe_divide(sum(1 for w in words if w.endswith("nya")), wc),    # suffix_nya_ratio
        _safe_divide(sum(1 for w in words if w.endswith("lah")), wc),    # suffix_lah_ratio
        _safe_divide(sum(1 for w in words if w.endswith("kah")), wc),    # suffix_kah_ratio
    ]

    # 28 function-word ratios — same order as notebook's FUNCTION_WORDS list
    for fw in FUNCTION_WORDS:
        feats.append(_safe_divide(words.count(fw), wc))

    return feats


def extract_features_dict(text: str) -> dict[str, float]:
    """Return features as a ``{name: value}`` dict (used for the API response)."""
    values = extract_features(text)
    return dict(zip(FEATURE_NAMES, values))
