"""Dynamic Creative Optimization (DCO): turns a (product, customer) pair
into ad copy.

Two tiers, matching the blueprint's design:
  1. Curated templates for the common (product, segment) combinations, with
     *dynamic value hooks* computed from the customer's own features (e.g.
     an actual estimated credit limit, not a flat number shown to everyone
     in the segment) — this is what makes it 1:1 rather than segment-level.
  2. A generative fallback for personas that fall outside the curated grid
     (e.g. an unusual age/segment/product combination). It calls a real LLM
     only if ANTHROPIC_API_KEY is set and the `anthropic` package is
     installed; otherwise it degrades to a deterministic template composer
     so the demo never hard-fails on a missing API key.
"""
from __future__ import annotations

import os

from src import config

PRODUCT_CODES = list(config.PRODUCT_CATALOG.keys())


def _cashback_rate(income: float) -> str:
    # income is on the proxy dataset's raw scale (median ~97k), not real
    # Rupiah — see config.INCOME_P* / docs/EVALUATION_NOTES.md. Gate on the
    # 75th percentile so this is a real ~top-quartile tier, not a threshold
    # almost nobody clears.
    return "5%" if income >= config.INCOME_P75 else "3%"


def _credit_limit_estimate(income: float) -> str:
    # Percentile-tiered rather than a raw multiply-and-round: `income` isn't
    # real Rupiah, so doing arithmetic on it as if it were (e.g. income * 3)
    # produces a number that looks precise but is meaningless, and previously
    # floored to the same Rp1,000,000 for nearly every customer.
    if income >= config.INCOME_P90:
        limit = 50_000_000
    elif income >= config.INCOME_P85:
        limit = 30_000_000
    elif income >= config.INCOME_P75:
        limit = 15_000_000
    elif income >= config.INCOME_P50:
        limit = 8_000_000
    elif income >= config.INCOME_P25:
        limit = 5_000_000
    else:
        limit = 3_000_000
    return f"Rp{limit:,.0f}".replace(",", ".")


def _kpr_rate(segment: str) -> str:
    return "5.75% p.a." if segment == "Nasabah Prioritas" else "6.75% p.a."


def _compute_hooks(product_code: str, segment: str, age: float, income: float) -> dict:
    return {
        "cashback_rate": _cashback_rate(income),
        "credit_limit": _credit_limit_estimate(income),
        "kpr_rate": _kpr_rate(segment),
        "age": int(age),
        "segment": segment,
    }


# (product_code, segment) -> (headline template, cta)
CREATIVE_TEMPLATES: dict[tuple[str, str], tuple[str, str]] = {
    ("ind_tjcr_fin_ult1", "Nasabah Reguler"): ("Cashback {cashback_rate} untuk belanja & makan, limit awal hingga {credit_limit}.", "Ajukan Sekarang"),
    ("ind_tjcr_fin_ult1", "Nasabah Prioritas"): ("Kartu Kredit eksklusif dengan cashback {cashback_rate} dan limit hingga {credit_limit}.", "Klaim Kartu"),
    ("ind_hip_fin_ult1", "Nasabah Prioritas"): ("KPR bunga spesial mulai {kpr_rate} untuk rumah impian Anda.", "Hitung Simulasi"),
    ("ind_hip_fin_ult1", "Nasabah Reguler"): ("Wujudkan rumah pertama dengan KPR mulai {kpr_rate}.", "Cek Kelayakan"),
    ("ind_cno_fin_ult1", "Nasabah Muda"): ("Pindahkan gajimu & dapatkan bonus welcome instan.", "Pindah Payroll"),
    ("ind_plan_fin_ult1", "Nasabah Prioritas"): ("Maksimalkan masa pensiun dengan program Dana Pensiun terkelola.", "Mulai Sekarang"),
    ("ind_fond_fin_ult1", "Nasabah Reguler"): ("Mulai investasi Reksa Dana mulai dari Rp100 ribu.", "Mulai Investasi"),
    ("ind_ecue_fin_ult1", "Nasabah Muda"): ("Buka Rekening Digital gratis biaya admin, langsung dari HP.", "Buka Rekening"),
    ("ind_recibo_ult1", "Nasabah Reguler"): ("Aktifkan autodebet tagihan, tidak pernah telat bayar lagi.", "Aktifkan"),
    ("ind_pres_fin_ult1", "Nasabah Reguler"): ("Butuh dana cepat? Kredit Multiguna cair tanpa ribet.", "Ajukan Kredit"),
}

GENERIC_CTA_BY_CATEGORY = {
    "Tabungan": "Buka Rekening", "Deposito": "Buka Deposito", "Investasi": "Mulai Investasi",
    "Pinjaman": "Ajukan Sekarang", "Kartu": "Ajukan Kartu", "Payroll": "Pindah Payroll",
    "Digital": "Buka Rekening", "Layanan": "Aktifkan",
}


def _fallback_copy(product_code: str, hooks: dict) -> tuple[str, str]:
    """Deterministic template composer used when no curated template exists
    and no LLM is configured. Structurally identical output shape to the
    LLM path so callers never need to special-case it."""
    meta = config.PRODUCT_CATALOG[product_code]
    headline = f"Penawaran khusus untuk Anda: {meta['name']}, dirancang untuk {hooks['segment']}."
    cta = GENERIC_CTA_BY_CATEGORY.get(meta["category"], "Pelajari Lebih Lanjut")
    return headline, cta


def _llm_copy(product_code: str, hooks: dict) -> tuple[str, str] | None:
    """Real LLM path — only attempted if an API key is configured. Returns
    None (never raises) on any failure so callers can fall back cleanly."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
    except ImportError:
        return None

    meta = config.PRODUCT_CATALOG[product_code]
    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=60,
            messages=[{
                "role": "user",
                "content": (
                    "You are a senior Indonesian retail-banking copywriter. Generate a concise "
                    "12-word Bahasa Indonesia ad headline and a 3-word CTA (pipe-separated: "
                    f"headline|cta) for customer segment {hooks['segment']}, age {hooks['age']}, "
                    f"pitching the product '{meta['name']}'. No emoji, no quotes."
                ),
            }],
        )
        text = msg.content[0].text.strip()
        headline, _, cta = text.partition("|")
        headline, cta = headline.strip(), cta.strip()
        if headline and cta:
            return headline, cta
    except Exception:
        pass
    return None


def generate_creative(product_code: str, segment: str, age: float, income: float) -> dict:
    hooks = _compute_hooks(product_code, segment, age, income)
    meta = config.PRODUCT_CATALOG[product_code]

    template = CREATIVE_TEMPLATES.get((product_code, segment))
    if template is not None:
        headline_tpl, cta = template
        headline = headline_tpl.format(**hooks)
        source = "curated_template"
    else:
        llm_result = _llm_copy(product_code, hooks)
        if llm_result is not None:
            headline, cta = llm_result
            source = "llm_generated"
        else:
            headline, cta = _fallback_copy(product_code, hooks)
            source = "template_fallback"

    return {
        "product_code": product_code,
        "product_name": meta["name"],
        "headline": headline,
        "cta": cta,
        "deep_link": f"app://ads/open?product={product_code}",
        "creative_source": source,
    }
