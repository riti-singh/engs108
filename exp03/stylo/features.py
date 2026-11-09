"""exp03 stylometric feature groups.

Groups (names used in configs and ablations):
  char    char_wb 3-5 TF-IDF (exp02)
  word    word 1-2 TF-IDF (exp02)
  style   27 exp02 style features (copied verbatim from exp02; test_features checks equality)
  fw      relative frequencies of a fixed English function-word list (dense)
  fwng    function-word n-grams: content words masked by their shape, TF-IDF 1-3 grams
  typo    typography / formatting rates (dense)
  sent    sentence / paragraph structure statistics and word-length histogram (dense)
  shape   token-shape n-grams (Xx, x, X, d, punctuation literal, newline), TF-IDF 1-3 grams
Dense groups are standardised; TF-IDF groups are used as is. Every learned step is a sklearn
transformer, so it is refit inside each CV fold.
"""
import re
import string

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# ---------------------------------------------------------------- exp02 style features (verbatim)
PUNCT = set(string.punctuation) | set("—–‘’“”…«»·•")
STYLE_NAMES = ["log_chars", "log_words", "mean_word_len", "n_sentences", "mean_sentence_words", "type_token_ratio",
               "punct_rate", "upper_rate", "digit_rate", "whitespace_rate", "newline_rate",
               "comma_rate", "semicolon_rate", "colon_rate", "em_dash_rate", "en_dash_rate", "hyphen_rate",
               "curly_apos_rate", "straight_apos_rate", "curly_quote_rate", "straight_quote_rate",
               "paren_rate", "exclaim_rate", "question_rate", "ellipsis_rate", "asterisk_rate", "hash_rate"]


def style_features(texts):
    rows = []
    for t in texts:
        n = max(len(t), 1)
        words = re.findall(r"\w+", t)
        sents = [s for s in re.split(r"[.!?]+\s", t) if s.strip()]
        rows.append([
            np.log1p(len(t)), np.log1p(len(words)), np.mean([len(w) for w in words]) if words else 0.0,
            len(sents), len(words) / max(len(sents), 1), len({w.lower() for w in words}) / max(len(words), 1),
            sum(c in PUNCT for c in t) / n, sum(c.isupper() for c in t) / n, sum(c.isdigit() for c in t) / n,
            sum(c.isspace() for c in t) / n, t.count("\n") / n,
            t.count(",") / n, t.count(";") / n, t.count(":") / n, t.count("—") / n, t.count("–") / n, t.count("-") / n,
            t.count("’") / n, t.count("'") / n, (t.count("“") + t.count("”")) / n, t.count('"') / n,
            (t.count("(") + t.count(")")) / n, t.count("!") / n, t.count("?") / n,
            (t.count("…") + t.count("...")) / n, t.count("*") / n, t.count("#") / n,
        ])
    return np.asarray(rows, dtype=np.float64)


# ---------------------------------------------------------------- function words
# Closed-class English words: determiners, pronouns, prepositions, conjunctions, auxiliaries,
# common adverbs/particles and discourse markers. Fixed a priori (not learned from the data).
FUNCTION_WORDS = sorted(set("""
a an the this that these those some any each every either neither no another such what which whose
i me my mine myself we us our ours ourselves you your yours yourself yourselves he him his himself she her hers
herself it its itself they them their theirs themselves one ones oneself who whom whoever whatever whichever
about above across after against along amid among around as at before behind below beneath beside besides
between beyond but by despite down during except for from in inside into like near of off on onto out outside
over past per since than through throughout till to toward towards under underneath unlike until up upon via
with within without
and or nor so yet both whether because although though while whereas if unless once whenever wherever
however therefore thus hence moreover furthermore nevertheless nonetheless meanwhile instead otherwise indeed
also still even just only already again ever never always often sometimes rather quite very too much more most
less least enough almost perhaps maybe merely simply really actually particularly especially
be am is are was were been being have has had having do does did doing will would shall should can could
may might must ought need dare
not here there where when why how then now well all both few many several most other others same own
""".split()))
_WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
_FW_INDEX = {w: i for i, w in enumerate(FUNCTION_WORDS)}


def _words(t):
    return _WORD_RE.findall(t)


def function_word_features(texts):
    """Relative frequency (per word) of each function word; contractions split on the apostrophe."""
    out = np.zeros((len(texts), len(FUNCTION_WORDS)))
    for r, t in enumerate(texts):
        ws = [w.lower().replace("’", "'").split("'")[0] for w in _words(t)]
        for w in ws:
            j = _FW_INDEX.get(w)
            if j is not None:
                out[r, j] += 1
        out[r] /= max(len(ws), 1)
    return out


def _word_shape(w):
    if w.isdigit():
        return "d"
    if w.isupper() and len(w) > 1:
        return "X"
    if w[0].isupper():
        return "Xx"
    return "x"


_TOKEN_RE = re.compile(r"\n|[A-Za-z]+(?:['’][A-Za-z]+)?|\d+(?:[.,]\d+)*|[^\w\s]")


def _tokens(t):
    return _TOKEN_RE.findall(t)


def fw_masked_docs(texts):
    """Keep function words (lower-cased) and punctuation; replace other tokens by their shape."""
    docs = []
    for t in texts:
        out = []
        for tok in _tokens(t):
            if tok == "\n":
                out.append("NL")
            elif tok[0].isalpha():
                base = tok.lower().replace("’", "'").split("'")[0]
                out.append(base if base in _FW_INDEX else "<" + _word_shape(tok) + ">")
            elif tok[0].isdigit():
                out.append("<d>")
            else:
                out.append(tok)
        docs.append(" ".join(out))
    return docs


def shape_docs(texts):
    """Token-shape sequence: word shape + length bucket (s<=3, m<=6, l), digits 'd', punctuation literal."""
    docs = []
    for t in texts:
        out = []
        for tok in _tokens(t):
            if tok == "\n":
                out.append("NL")
            elif tok[0].isalpha():
                L = len(tok)
                out.append(_word_shape(tok) + ("s" if L <= 3 else "m" if L <= 6 else "l")
                           + ("'" if ("'" in tok or "’" in tok) else ""))
            elif tok[0].isdigit():
                out.append("d")
            else:
                out.append(tok)
        docs.append(" ".join(out))
    return docs


# ---------------------------------------------------------------- typography
TYPO_NAMES = ["non_ascii_rate", "curly_share_of_quotes", "curly_share_of_apos", "spaced_em_dash_rate",
              "unspaced_em_dash_rate", "spaced_hyphen_dash_rate", "double_hyphen_rate", "double_space_rate",
              "space_before_punct_rate", "bold_md_rate", "italic_md_rate", "bullet_line_frac", "numbered_line_frac",
              "heading_line_frac", "blank_line_rate", "contraction_rate", "hyphenated_word_rate", "title_case_word_rate",
              "all_caps_word_rate", "oxford_comma_rate", "eg_ie_rate", "percent_rate", "ampersand_rate", "slash_rate",
              "colon_then_capital_rate", "semicolon_per_sentence", "ends_with_period", "ends_with_other",
              "starts_with_quote", "trailing_space_line_frac", "number_token_rate", "ellipsis_char_share"]


def typography_features(texts):
    rows = []
    for t in texts:
        n = max(len(t), 1)
        words = _words(t)
        nw = max(len(words), 1)
        lines = t.split("\n")
        nl = max(len(lines), 1)
        quotes_curly = t.count("“") + t.count("”")
        quotes_all = quotes_curly + t.count('"')
        apos_curly = t.count("’")
        apos_all = apos_curly + t.count("'")
        n_sent = max(len([s for s in re.split(r"[.!?]+\s", t) if s.strip()]), 1)
        ell_char, ell_dots = t.count("…"), t.count("...")
        s = t.rstrip()
        rows.append([
            sum(ord(c) > 127 for c in t) / n,
            quotes_curly / quotes_all if quotes_all else 0.5,
            apos_curly / apos_all if apos_all else 0.5,
            len(re.findall(r" — ", t)) / n, len(re.findall(r"\S—\S", t)) / n, len(re.findall(r" - ", t)) / n,
            t.count("--") / n, t.count("  ") / n, len(re.findall(r" [,.;:!?]", t)) / n,
            len(re.findall(r"\*\*[^*]+\*\*", t)) / n, len(re.findall(r"(?<![*\w])[*_][^*_\n]+[*_](?![*\w])", t)) / n,
            sum(bool(re.match(r"\s*[-*•]\s", l)) for l in lines) / nl,
            sum(bool(re.match(r"\s*\d+[.)]\s", l)) for l in lines) / nl,
            sum(bool(re.match(r"\s*#+\s", l)) for l in lines) / nl,
            t.count("\n\n") / n,
            sum(("'" in w or "’" in w) for w in words) / nw,
            len(re.findall(r"\b[A-Za-z]+-[A-Za-z]+\b", t)) / nw,
            sum(w[0].isupper() and w[1:].islower() for w in words if len(w) > 1) / nw,
            sum(w.isupper() and len(w) > 1 for w in words) / nw,
            len(re.findall(r",\s+(?:and|or)\b", t)) / nw,
            len(re.findall(r"\b(?:e\.g\.|i\.e\.|etc\.)", t)) / nw,
            t.count("%") / n, t.count("&") / n, t.count("/") / n,
            len(re.findall(r":\s+[A-Z]", t)) / n, t.count(";") / n_sent,
            float(s.endswith(".")), float(bool(s) and s[-1] not in ".!?\"”'’)"),
            float(t.lstrip()[:1] in "\"“'‘"),
            sum(l != l.rstrip(" \t") for l in lines) / nl,
            len(re.findall(r"\b\d", t)) / nw,
            ell_char / (ell_char + ell_dots) if (ell_char + ell_dots) else 0.5,
        ])
    return np.asarray(rows, dtype=np.float64)


# ---------------------------------------------------------------- sentence / paragraph structure
_SENT_SPLIT = re.compile(r"(?<=[.!?…])[\"”’)]*\s+(?=[\"“‘(]*[A-Z0-9])")
SENT_NAMES = (["sent_len_mean", "sent_len_std", "sent_len_cv", "sent_len_median", "sent_len_p10", "sent_len_p90",
               "sent_len_min", "sent_len_max", "n_sentences_log", "frac_short_sent", "frac_long_sent",
               "first_sent_len", "last_sent_len", "commas_per_sent", "frac_question_sent", "frac_exclaim_sent",
               "frac_conj_start", "frac_the_start", "frac_pronoun_start", "frac_adverb_ly_start",
               "frac_sent_with_colon", "frac_sent_with_dash", "frac_sent_with_paren",
               "n_paragraphs_log", "sent_per_para_mean", "words_per_para_mean", "words_per_para_std",
               "word_len_std", "long_word_rate", "hapax_rate", "mean_word_freq_rank_proxy"]
              + [f"wordlen_{k}" for k in range(1, 16)])
_CONJ = {"and", "but", "or", "so", "yet", "nor"}
_PRON = {"i", "we", "you", "he", "she", "it", "they", "this", "that", "these", "those"}


def sentence_features(texts):
    rows = []
    for t in texts:
        sents = [s for s in _SENT_SPLIT.split(t.strip()) if s.strip()] or [t]
        sw = [_words(s) for s in sents]
        L = np.array([len(w) for w in sw], dtype=float)
        first = [w[0].lower() if w else "" for w in sw]
        paras = [p for p in re.split(r"\n\s*\n", t.strip()) if p.strip()] or [t]
        pw = np.array([len(_words(p)) for p in paras], dtype=float)
        words = _words(t)
        wl = np.array([len(w) for w in words], dtype=float) if words else np.zeros(1)
        lower = [w.lower() for w in words]
        counts = {}
        for w in lower:
            counts[w] = counts.get(w, 0) + 1
        nw = max(len(words), 1)
        ns = len(sents)
        hist = np.bincount(np.clip(wl.astype(int), 1, 15), minlength=16)[1:16] / nw
        rows.append([
            L.mean(), L.std(), L.std() / max(L.mean(), 1e-9), np.median(L), np.percentile(L, 10), np.percentile(L, 90),
            L.min(), L.max(), np.log1p(ns), (L < 8).mean(), (L > 30).mean(), L[0], L[-1],
            sum(s.count(",") for s in sents) / ns, np.mean([s.rstrip().endswith("?") for s in sents]),
            np.mean([s.rstrip().endswith("!") for s in sents]),
            np.mean([f in _CONJ for f in first]), np.mean([f == "the" for f in first]),
            np.mean([f in _PRON for f in first]), np.mean([f.endswith("ly") and len(f) > 3 for f in first]),
            np.mean([":" in s for s in sents]), np.mean([("—" in s or " - " in s or "–" in s) for s in sents]),
            np.mean(["(" in s for s in sents]),
            np.log1p(len(paras)), ns / len(paras), pw.mean(), pw.std(),
            wl.std(), (wl >= 7).mean(), sum(c == 1 for c in counts.values()) / nw,
            np.mean([w in _FW_INDEX for w in lower]) if lower else 0.0,
            *hist,
        ])
    return np.asarray(rows, dtype=np.float64)


# ---------------------------------------------------------------- group registry
def _dense(fn, clip=None):
    steps = [("f", FunctionTransformer(fn)), ("z", StandardScaler())]
    if clip is not None:
        steps.append(("clip", FunctionTransformer(np.clip, kw_args={"a_min": -clip, "a_max": clip})))
    return Pipeline(steps)


def fw_tokens(text):
    """Function-word tokens of a text (the analyzer of the `fw` group)."""
    return [w for w in (x.lower().replace("’", "'").split("'")[0] for x in _words(text)) if w in _FW_INDEX]


def _tfidf_docs(fn, **kw):
    return Pipeline([("docs", FunctionTransformer(fn)),
                     ("tfidf", TfidfVectorizer(token_pattern=r"\S+", lowercase=False, sublinear_tf=True, **kw))])


def make_group(name):
    if name == "char":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    if name == "word":
        return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    if name == "style":
        return _dense(style_features)
    if name == "fw":
        # sublinear TF-IDF over the fixed function-word vocabulary; standardised per-word rates gave rare
        # function words huge z-scores on held-out rows (first ablation run: OOF log loss ~12)
        return TfidfVectorizer(analyzer=fw_tokens, vocabulary=FUNCTION_WORDS, sublinear_tf=True)
    if name == "fwng":
        return _tfidf_docs(fw_masked_docs, ngram_range=(1, 3), min_df=3)
    if name == "typo":
        return _dense(typography_features, clip=DENSE_CLIP)
    if name == "sent":
        return _dense(sentence_features, clip=DENSE_CLIP)
    if name == "shape":
        return _tfidf_docs(shape_docs, ngram_range=(1, 3), min_df=3)
    raise ValueError(name)


DENSE_CLIP = 5.0
GROUPS = ["char", "word", "style", "fw", "fwng", "typo", "sent", "shape"]
EXP02_GROUPS = ["char", "word", "style"]
NEW_GROUPS = ["fw", "fwng", "typo", "sent", "shape"]


def make_features(groups):
    """ColumnTransformer over a DataFrame with a 'text' column; block order = `groups` order."""
    return ColumnTransformer([(g, make_group(g), "text") for g in groups])
