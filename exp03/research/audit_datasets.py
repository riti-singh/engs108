"""Exp03 data audit of the candidate augmentation datasets (audit only; nothing is merged into training).

Reads local copies only (no network). Inputs:
  COMP_DIR: competition files from `kaggle competitions download -c name-that-llm`
  ZAK_DIR:  `kaggle datasets download zakirkhanaleemi/name-that-llm-benchmarks-proxy-generations --unzip` (version 1)
The maniceet dataset could not be downloaded (HTTP 403 / 404), so it is recorded as inaccessible.

Usage: python exp03/research/audit_datasets.py --comp ~/work/comp --zak ~/work/ds/zak --out exp03/artifacts/data_audit.json
"""
import argparse
import hashlib
import json
import platform
import re
import string
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import scipy.optimize
import sklearn
from scipy import sparse
from scipy.stats import ks_2samp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import normalize

SEED = 42
LABELS = ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'glm-5.3',
          'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
          'kimi-k3', 'qwen3.8-max']
NEAR_JACCARD = 0.5       # word 5-gram shingle Jaccard; >= 0.5 flags a near duplicate
NEAR_CHAR_COS = 0.9      # char 3-5 gram TF-IDF cosine; >= 0.9 flags a near duplicate

# Facts recorded by hand during the audit (Kaggle API / web, 2026-11-08); not recomputed by this script.
MANICEET = {
    'ref': 'maniceet/name-that-llm-openrouter-openai-generations', 'status': 'NOT AUDITED: inaccessible',
    'access_attempts_2026_11_08': {
        'kaggle datasets files / metadata / status / download (CLI 2.2.5, account ritisingh123)':
            '403 Forbidden "You don\'t have access to this resource."',
        'https://www.kaggle.com/datasets/maniceet/name-that-llm-openrouter-openai-generations (unauthenticated GET)': 'HTTP 404',
        'kaggle datasets list --user maniceet': 'only 4 datasets, all from 2018 (histograms, histogram, keyalldays, teamscsv)',
        'kaggle datasets list -s "openrouter-openai-generations" / -s "name that llm"': 'no dataset with this ref',
        'dataset_sources of pulled public notebooks (6 kernels) and crtproduction open pool': 'not referenced',
    },
    'interpretation': 'private, deleted, renamed, or a mistyped ref; contents unknown. No version, hash, class count, '
                      'provenance, duplicate or shift result exists for it, and none is inferred.',
    'kaggle_version': None, 'files': None, 'class_counts_18': None,
}
ZAK_PROVENANCE = {
    'published': '2026-10-19 10:03:15 UTC (kaggle datasets list --user zakirkhanaleemi); generation 03:47-04:09 UTC same day',
    'uploader': 'zakirkhanaleemi',
    'source_model_identity': {
        'value': 'Kaggle Benchmarks Model Proxy slugs (model_slug column), e.g. openai/gpt-5.5-2026-04-23, google/gemini-3.6-flash',
        'verification': 'per-row metadata + README only (CLAIMED). No generation code or proxy logs are published; the proxy '
                        'slug -> competition label mapping is the uploader\'s. Indirect check: an 8-way char TF-IDF '
                        'classifier fit on competition train rows recovers family 87% and exact label at 0.317 balanced '
                        'accuracy (vs 0.403 grouped-CV on train), i.e. labels are mostly consistent with train styles but shifted.'},
    'generation_settings': {
        'value': 'daily: reasoning=low, temperature=1.0, max_tokens=4096 (rlow_t1, 747 rows); ab2: reasoning=low, temperature=0.0 '
                 '(rlow_t0, 80); ab_settings: default / t1 / t1_rlow (7)',
        'verification': 'setting column + README (CLAIMED). Request template verified from the request column: '
                        '"<prompt>\\n\\nWrite a single paragraph of about N words, with no title or headings." Target length N '
                        'tracks n_words (corr 0.957, median ratio 1.00).'},
    'prompts': {'value': 'sy = prompts.csv of seyitkaangunes/ntl-extra-api-passages (821 rows), pv = prompts.csv of '
                         'prvsiyan/prvsiyans-extended-name-that-llm-dataset (13 rows); 213 distinct prompts',
                'verification': 'bank column + README (CLAIMED); README states no prompt was derived from competition data, '
                                'consistent with the topic-overlap check (0/213 prompts above the train cross-prompt 99th pct).'},
    'text_processing': 'README: surrounding whitespace stripped, otherwise unedited, unfiltered (CLAIMED).',
}
RULES = {
    'sources': ['competition Rules page (external data must be publicly and equally accessible at no cost, or meet '
                'the reasonableness criteria)', 'Code Requirements page (external data allowed; offline; 9 h)',
                'discussion 752084: host posts 2026-10-28 and 2026-10-29'],
    'host_quotes': ['"Existing public community datasets on Kaggle are fine to train on." (2026-10-28)',
                    '"It is not the intent of the Benchmarks proxy quota to be used for non-benchmarking work; however, '
                    'we will not penalize participants who have already used it ... Public community datasets on Kaggle '
                    '(including those generated via the proxy) are fine to train on." (2026-10-29)'],
    'zakirkhanaleemi': 'LIKELY COMPLIANT as external training data: public Kaggle dataset, CC BY 4.0, free, published '
                       '2026-10-19 (before the host clarifications); host explicitly allows public proxy-generated '
                       'datasets. Attribution to the dataset and both prompt banks required by CC BY 4.0. We must NOT '
                       'generate more data with the proxy ourselves.',
    'maniceet': 'UNDETERMINED: not publicly accessible to us (403/404), so it currently fails the "publicly and '
                'equally accessible" test; re-check if the ref is corrected.',
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def norm_text(t):
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', str(t).lower())).strip()


def shingles(t, k=5):
    w = norm_text(t).split()
    return {' '.join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


def shingle_matrix(text_lists):
    """Binary CSR shingle matrices over a shared vocabulary, one per input list."""
    vocab, rows = {}, []
    for texts in text_lists:
        r, c = [], []
        for i, t in enumerate(texts):
            for s in shingles(t):
                c.append(vocab.setdefault(s, len(vocab))); r.append(i)
        rows.append((r, c, len(texts)))
    return [sparse.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, len(vocab))) for r, c, n in rows]


def jaccard(A, B):
    inter = (A @ B.T).toarray()
    sa, sb = np.asarray(A.sum(1)).ravel(), np.asarray(B.sum(1)).ravel()
    return inter / np.maximum(sa[:, None] + sb[None, :] - inter, 1)


def dup_report(a_texts, b_texts, char_vec, same=False, a_meta=None):
    if same:
        b_texts = a_texts
    na = np.array([norm_text(t) for t in a_texts]); nb = np.array([norm_text(t) for t in b_texts])
    A, B = shingle_matrix([a_texts, b_texts])
    J = jaccard(A, B)
    Ca, Cb = char_vec.transform(a_texts), char_vec.transform(b_texts)
    C = (Ca @ Cb.T).toarray()
    E = na[:, None] == nb[None, :]
    if same:
        iu = np.triu_indices(len(a_texts), 1)
        pairs = [(int(i), int(j)) for i, j in zip(*iu)]
        jv, cv, ev = J[iu], C[iu], E[iu]
        out = {'pairs_compared': len(pairs), 'exact_dup_pairs': int(ev.sum()),
               'near_dup_pairs_jaccard5>=%.1f' % NEAR_JACCARD: int((jv >= NEAR_JACCARD).sum()),
               'near_dup_pairs_charcos>=%.2f' % NEAR_CHAR_COS: int((cv >= NEAR_CHAR_COS).sum()),
               'max_jaccard5': round(float(jv.max()), 4), 'max_charcos': round(float(cv.max()), 4)}
        flagged = np.where((jv >= NEAR_JACCARD) | (cv >= NEAR_CHAR_COS) | ev)[0]
        if a_meta is not None:
            ex = []
            for k in flagged[:50]:
                i, j = pairs[k]
                ex.append({'row_a': i, 'row_b': j, 'a': a_meta[i], 'b': a_meta[j], 'exact': bool(ev[k]),
                           'jaccard5': round(float(jv[k]), 3), 'charcos': round(float(cv[k]), 3)})
            out['flagged_pairs'] = ex
            out['flagged_pairs_same_model'] = int(sum(a_meta[pairs[k][0]]['model'] == a_meta[pairs[k][1]]['model']
                                                      for k in flagged))
        return out
    return {'pairs_compared': int(J.size), 'exact_dup_pairs': int(E.sum()),
            'rows_with_exact_dup': int(E.any(1).sum()),
            'near_dup_pairs_jaccard5>=%.1f' % NEAR_JACCARD: int((J >= NEAR_JACCARD).sum()),
            'near_dup_pairs_charcos>=%.2f' % NEAR_CHAR_COS: int((C >= NEAR_CHAR_COS).sum()),
            'max_jaccard5': round(float(J.max()), 4), 'max_charcos': round(float(C.max()), 4),
            'charcos_rowmax_quantiles': {q: round(float(np.quantile(C.max(1), q)), 4) for q in (0.5, 0.9, 0.99, 1.0)}}


PUNCT = {'comma': ',', 'semicolon': ';', 'colon': ':', 'em_dash': '\u2014', 'en_dash': '\u2013', 'hyphen': '-',
         'curly_double': '\u201c\u201d', 'curly_single': '\u2018\u2019', 'straight_double': '"',
         'straight_single': "'", 'paren': '(', 'exclaim': '!', 'question': '?', 'asterisk': '*'}


def style_features(t):
    t = str(t); words = t.split(); nc = max(len(t), 1)
    sents = [s for s in re.split(r'(?<=[.!?])\s+', t.strip()) if s]
    f = {'n_words': len(words), 'n_chars': len(t), 'n_sentences': len(sents),
         'words_per_sentence': len(words) / max(len(sents), 1),
         'mean_word_len': float(np.mean([len(w.strip(string.punctuation)) for w in words])) if words else 0.0,
         'type_token_ratio_first100': len({w.lower() for w in words[:100]}) / max(min(len(words), 100), 1),
         'has_newline': float('\n' in t), 'digit_per_1k': 1000 * sum(c.isdigit() for c in t) / nc,
         'upper_per_1k': 1000 * sum(c.isupper() for c in t) / nc}
    for k, chars in PUNCT.items():
        f[k + '_per_1k'] = 1000 * sum(t.count(c) for c in chars) / nc
    f['has_em_dash'] = float('\u2014' in t); f['has_curly_quote'] = float(any(c in t for c in '\u201c\u201d\u2018\u2019'))
    return f


def style_shift(train, ext, classes):
    Ft = pd.DataFrame([style_features(t) for t in train.text]); Ft['model'] = train.model.values
    Fe = pd.DataFrame([style_features(t) for t in ext.text]); Fe['model'] = ext.model.values
    feats = [c for c in Ft.columns if c != 'model']
    pooled, per_class = {}, {}
    tr8 = Ft[Ft.model.isin(classes)]
    for c in feats:
        ks = ks_2samp(tr8[c], Fe[c])
        pooled[c] = {'train_mean': round(float(tr8[c].mean()), 4), 'ext_mean': round(float(Fe[c].mean()), 4),
                     'ks_stat': round(float(ks.statistic), 4), 'ks_p': float(f'{ks.pvalue:.3g}')}
    for m in classes:
        a, b = Ft[Ft.model == m], Fe[Fe.model == m]
        d = {}
        for c in feats:
            ks = ks_2samp(a[c], b[c])
            d[c] = {'train_mean': round(float(a[c].mean()), 3), 'ext_mean': round(float(b[c].mean()), 3),
                    'ks_stat': round(float(ks.statistic), 3)}
        per_class[m] = {'n_train': len(a), 'n_ext': len(b), 'features': d}
    return pooled, per_class


def prompt_overlap(train, ext, ext_groups, ext_requests, stop):
    """Topic overlap via content-word TF-IDF prompt centroids. Train has no prompt text, so train prompts are
    represented by the centroid of their passages; the null distribution is train-vs-train across prompts."""
    vec = TfidfVectorizer(stop_words=stop, sublinear_tf=True, min_df=2, token_pattern=r'(?u)\b[a-zA-Z]{3,}\b')
    vec.fit(pd.concat([train.text, ext.text]))
    rng = np.random.RandomState(SEED)
    tp = sorted(train.prompt_id.unique())
    halves_a, halves_b = [], []
    for p in tp:
        idx = rng.permutation(np.where(train.prompt_id.values == p)[0]); h = len(idx) // 2
        halves_a.append(np.asarray(vec.transform(train.text.values[idx[:h]]).mean(0)).ravel())
        halves_b.append(np.asarray(vec.transform(train.text.values[idx[h:]]).mean(0)).ravel())
    HA, HB = normalize(np.vstack(halves_a)), normalize(np.vstack(halves_b))
    S = HA @ HB.T
    same = np.diag(S); cross = S[~np.eye(len(tp), dtype=bool)]
    full = normalize(np.vstack([np.asarray(vec.transform(train.text.values[train.prompt_id.values == p]).mean(0)).ravel()
                                for p in tp]))
    thr = float(np.quantile(np.max(np.where(np.eye(len(tp), dtype=bool), -1, S), 1), 0.99))
    eg = sorted(set(ext_groups))
    E = normalize(np.vstack([np.asarray(vec.transform(ext.text.values[np.array(ext_groups) == g]).mean(0)).ravel()
                             for g in eg]))
    SE = E @ full.T
    best = SE.max(1); arg = SE.argmax(1)
    flagged = []
    for k in np.argsort(-best):
        if best[k] < thr:
            break
        g = eg[k]; req = ext_requests.get(g, '')
        snippet = train.text.values[train.prompt_id.values == tp[arg[k]]][0][:160]
        flagged.append({'ext_prompt': g, 'nearest_train_prompt': tp[arg[k]], 'cos': round(float(best[k]), 3),
                        'ext_request_head': req[:160], 'train_passage_head': snippet})
    return {'method': 'content-word TF-IDF (english stop words, alpha tokens >=3 chars, sublinear tf, min_df=2) prompt '
                      'centroids; train prompt = mean of its passages; null = max cosine of a train half-prompt '
                      'centroid to any OTHER train prompt half (seed 42)',
            'train_same_prompt_half_cos_quantiles': {q: round(float(np.quantile(same, q)), 4) for q in (0.01, 0.05, 0.5)},
            'train_cross_prompt_rowmax_cos_quantiles': {q: round(float(np.quantile(np.max(np.where(np.eye(len(tp), dtype=bool), -1, S), 1), q)), 4)
                                                       for q in (0.5, 0.9, 0.99, 1.0)},
            'flag_threshold_cos (99th pct of train cross-prompt rowmax)': round(thr, 4),
            'ext_prompts': len(eg),
            'ext_prompt_to_nearest_train_prompt_cos_quantiles': {q: round(float(np.quantile(best, q)), 4) for q in (0.5, 0.9, 0.99, 1.0)},
            'ext_prompts_above_threshold': len(flagged), 'flagged': flagged[:25]}


def domain_and_label_checks(train, tr_idx, ext, ext_groups, classes):
    """(a) train-vs-ext domain classifier AUC (grouped 5-fold); (b) 8-class style classifier fit on the exp01
    training rows only, applied to ext rows, vs grouped-CV on the same rows. The 414 dev rows are never used."""
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), sublinear_tf=True, min_df=3, max_features=100000)
    tr = train.iloc[tr_idx]; tr = tr[tr.model.isin(classes)].reset_index(drop=True)
    X_all = vec.fit_transform(pd.concat([tr.text, ext.text]))
    Xt, Xe = X_all[:len(tr)], X_all[len(tr):]
    y_dom = np.r_[np.zeros(len(tr)), np.ones(len(ext))]
    groups = np.r_[('t_' + tr.prompt_id).values, np.array(['e_' + g for g in ext_groups])]
    oof = np.zeros(len(y_dom))
    for a, b in GroupKFold(5).split(X_all, y_dom, groups):
        clf = LogisticRegression(C=4.0, max_iter=2000, class_weight='balanced').fit(X_all[a], y_dom[a])
        oof[b] = clf.predict_proba(X_all[b])[:, 1]
    auc = roc_auc_score(y_dom, oof)
    yt = tr.model.values
    cv_pred = np.empty(len(tr), dtype=object)
    for a, b in GroupKFold(5).split(Xt, yt, tr.prompt_id.values):
        cv_pred[b] = LogisticRegression(C=4.0, max_iter=2000).fit(Xt[a], yt[a]).predict(Xt[b])
    clf = LogisticRegression(C=4.0, max_iter=2000).fit(Xt, yt)
    ext_pred = clf.predict(Xe)
    fam = lambda m: m.split('-')[0]
    per_class = {m: {'train_cv_recall': round(float((cv_pred[yt == m] == m).mean()), 3),
                     'ext_recall': round(float((ext_pred[ext.model.values == m] == m).mean()), 3)} for m in classes}
    return {'domain_classifier': {'features': 'char_wb 2-4 TF-IDF (sublinear, min_df=3, max 100k)',
                                  'model': 'LogisticRegression(C=4, class_weight=balanced)',
                                  'cv': 'GroupKFold(5) by prompt (train prompt_id / ext bank:prompt_id)',
                                  'rows_train': int(len(tr)), 'rows_ext': int(len(ext)),
                                  'oof_auc_train_vs_ext': round(float(auc), 4)},
            'label_consistency': {'note': f'{len(classes)}-way char-TF-IDF LR fit on exp01 training rows of these '
                                          'classes; ext rows scored as an external test set',
                                  'train_grouped_cv_balanced_acc': round(float(balanced_accuracy_score(yt, cv_pred)), 4),
                                  'ext_balanced_acc': round(float(balanced_accuracy_score(ext.model.values, ext_pred)), 4),
                                  'train_cv_family_acc': round(float(np.mean([fam(a) == fam(b) for a, b in zip(yt, cv_pred)])), 4),
                                  'ext_family_acc': round(float(np.mean([fam(a) == fam(b) for a, b in zip(ext.model.values, ext_pred)])), 4),
                                  'per_class': per_class}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--comp', required=True); ap.add_argument('--zak', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    comp, zdir = Path(a.comp).expanduser(), Path(a.zak).expanduser()
    warnings.filterwarnings('ignore', category=scipy.optimize.OptimizeWarning)
    train = pd.read_csv(comp / 'train.csv'); test = pd.read_csv(comp / 'test.csv')
    assert sorted(train.model.unique()) == LABELS
    tr_idx, _ = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED).split(train, groups=train.prompt_id))
    assert len(tr_idx) == 2285
    z = pd.read_csv(zdir / 'ntl_proxy_generations.csv')
    z_groups = (z.bank + ':' + z.prompt_id.astype(str)).tolist()
    z_req = {}
    for g, r in zip(z_groups, z.request.fillna('')):
        z_req.setdefault(g, re.split(r'\n\nWrite a single paragraph', r)[0])
    meta = [{'model': m, 'prompt': g, 'setting': s, 'run': r} for m, g, s, r in zip(z.model, z_groups, z.setting, z.run)]

    char_vec = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), sublinear_tf=True, min_df=2)
    char_vec.fit(pd.concat([train.text, test.text, z.text]))
    stop = list(ENGLISH_STOP_WORDS)
    classes = [m for m in LABELS if (z.model == m).any()]
    pooled, per_class = style_shift(train, z, classes)

    zak = {
        'ref': 'zakirkhanaleemi/name-that-llm-benchmarks-proxy-generations', 'status': 'audited',
        'kaggle_dataset_id': 12703965, 'kaggle_version': 1, 'license': 'CC BY 4.0 (dataset-metadata.json)',
        'files': {p.name: {'sha256': sha256(p), 'bytes': p.stat().st_size} for p in sorted(zdir.iterdir()) if p.is_file()},
        'rows': int(len(z)), 'columns': list(z.columns),
        'class_counts_18': {m: int((z.model == m).sum()) for m in LABELS},
        'classes_present': len(classes), 'classes_absent': [m for m in LABELS if m not in classes],
        'non_target_labels': sorted(set(z.model) - set(LABELS)),
        'model_slug_by_label': {m: sorted(z.model_slug[z.model == m].unique().tolist()) for m in classes},
        'run_counts': z.run.value_counts().to_dict(), 'setting_counts': z.setting.value_counts().to_dict(),
        'setting_by_run': pd.crosstab(z.setting, z.run).to_dict(), 'bank_counts': z.bank.value_counts().to_dict(),
        'finish_reason_counts': z.finish_reason.value_counts().to_dict(),
        'single_paragraph_counts': {str(k): int(v) for k, v in z.single_paragraph.value_counts().items()},
        'created_at_range': [str(z.created_at.min()), str(z.created_at.max())],
        'n_prompts': len(set(z_groups)),
        'models_per_prompt_quantiles': {q: float(np.quantile(pd.Series(z_groups).groupby(pd.Series(z_groups)).size().index.map(
            lambda g: z.model[np.array(z_groups) == g].nunique()), q)) for q in (0.0, 0.5, 1.0)},
        'n_words_vs_target_words': {'corr': round(float(np.corrcoef(z.target_words, z.n_words)[0, 1]), 4),
                                    'median_ratio': round(float((z.n_words / z.target_words).median()), 4)},
        'request_template_suffix_example': z.request.dropna().iloc[0].split('\n\n', 1)[-1],
        'duplicates': {
            'within': dup_report(z.text.tolist(), None, char_vec, same=True, a_meta=meta),
            'vs_train_text': dup_report(z.text.tolist(), train.text.tolist(), char_vec),
            'vs_test_text': dup_report(z.text.tolist(), test.text.tolist(), char_vec),
            'vs_maniceet': 'not computed: maniceet dataset inaccessible',
        },
        'prompt_overlap_vs_train': prompt_overlap(train, z, z_groups, z_req, stop),
        'prompt_id_namespace_overlap_with_train': sorted(set(z.prompt_id.astype(str)) & set(train.prompt_id)),
        'style_shift_vs_train_same_classes_pooled': pooled,
        'style_shift_vs_train_per_class': per_class,
        'length_by_class': {m: {'train_median_words': float(train.text[train.model == m].str.split().str.len().median()),
                                'ext_median_words': float(z.text[z.model == m].str.split().str.len().median())} for m in classes},
    }
    zak.update(domain_and_label_checks(train, tr_idx, z, z_groups, classes))
    zak['provenance'] = ZAK_PROVENANCE
    zg = pd.Series(z_groups)
    zak['family_prompt_coverage'] = {}
    for fam in ('gpt', 'gemini'):
        fam_labels = [m for m in LABELS if m.startswith(fam)]
        sel = z.model.str.startswith(fam).values
        nuniq = z.model[sel].groupby(zg[sel].values).nunique()
        zak['family_prompt_coverage'][fam] = {
            'target_labels': fam_labels, 'labels_present': sorted(z.model[sel].unique().tolist()),
            'rows': int(sel.sum()), 'prompts': int(len(nuniq)),
            'prompts_by_n_labels': {int(k): int(v) for k, v in nuniq.value_counts().sort_index().items()}}
    zak['rows_by_model_and_setting'] = pd.crosstab(z.model, z.setting).to_dict(orient='index')

    out = {
        'generated_by': 'exp03/research/audit_datasets.py', 'seed': SEED,
        'environment': {'python': sys.version.split()[0], 'platform': platform.platform(), 'numpy': np.__version__,
                        'pandas': pd.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
        'status_note': 'Candidate augmentation data only; NOT approved for training. Nothing was merged into any training set.',
        'competition_files': {p: sha256(comp / p) for p in ('train.csv', 'test.csv', 'sample_submission.parquet')},
        'competition_train': {'rows': len(train), 'prompts': int(train.prompt_id.nunique()),
                              'class_counts': train.model.value_counts().sort_index().to_dict()},
        'competition_test_local_rows': len(test),
        'thresholds': {'near_dup_jaccard_word5': NEAR_JACCARD, 'near_dup_char35_tfidf_cos': NEAR_CHAR_COS},
        'within_train_reference': dup_report(train.text.tolist(), None, char_vec, same=True),
        'datasets': {'maniceet/name-that-llm-openrouter-openai-generations': MANICEET,
                     'zakirkhanaleemi/name-that-llm-benchmarks-proxy-generations': zak},
        'cross_dataset_duplicates': 'not computed: maniceet dataset inaccessible',
        'rule_compliance': RULES,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
