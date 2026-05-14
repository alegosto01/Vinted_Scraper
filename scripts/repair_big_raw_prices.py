import argparse
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Optional

import pandas as pd

TWOPLACES = Decimal('0.01')
FEE_RATE = Decimal('0.05')
FIXED_FEE = Decimal('0.70')
ROUNDISH_CENTS = {0, 50, 90, 95, 99}


@dataclass
class RepairDecision:
    final_price: Optional[Decimal]
    reason: str
    confidence: str
    normalized_price: Optional[Decimal]
    separator_recovered_price: Optional[Decimal]
    fee_inverted_normalized_price: Optional[Decimal]
    fee_inverted_separator_price: Optional[Decimal]
    raw_decimal_places: Optional[int]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Repair malformed prices in big_raw.csv while leaving old_df untouched.')
    ap.add_argument('--input', required=True, help='Path to big_raw.csv')
    ap.add_argument('--old_df', default=None, help='Optional path to old_df.csv used only as a validation reference')
    ap.add_argument('--output', default=None, help='Output path for repaired CSV (default: <input stem>_fixed.csv)')
    ap.add_argument('--review', default=None, help='Output path for review CSV (default: <input stem>_price_review.csv)')
    return ap.parse_args()


def quantize_2(value: Decimal) -> Decimal:
    return value.quantize(TWOPLACES, rounding=ROUND_HALF_UP)


def extract_numeric_token(raw_value: object) -> Optional[str]:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text or text.lower() == 'nan':
        return None
    match = re.search(r'-?\d+(?:[.,]\d+)?', text)
    if not match:
        return None
    return match.group(0).replace(',', '.')


def decimal_places(raw_token: Optional[str]) -> Optional[int]:
    if raw_token is None or '.' not in raw_token:
        return 0 if raw_token else None
    return len(raw_token.split('.', 1)[1])


def normalize_two_decimals(raw_token: Optional[str]) -> Optional[Decimal]:
    if raw_token is None:
        return None
    try:
        return quantize_2(Decimal(raw_token))
    except InvalidOperation:
        return None


def recover_separator_collapsed_price(raw_token: Optional[str]) -> Optional[Decimal]:
    if raw_token is None or '.' not in raw_token:
        return None
    integer_part, fractional_part = raw_token.split('.', 1)
    if len(fractional_part) < 3:
        return None

    zeros_to_append = (2 - (len(fractional_part) % 3)) % 3
    adjusted_fractional = fractional_part + ('0' * zeros_to_append)
    euro_digits = f"{integer_part}{adjusted_fractional[:-2]}"
    euro_digits = euro_digits.lstrip('0') or '0'
    cents = adjusted_fractional[-2:]

    try:
        return quantize_2(Decimal(f'{euro_digits}.{cents}'))
    except InvalidOperation:
        return None


def invert_fee(gross_price: Optional[Decimal]) -> Optional[Decimal]:
    if gross_price is None or gross_price <= FIXED_FEE:
        return None
    return quantize_2((gross_price - FIXED_FEE) / (Decimal('1.00') + FEE_RATE))


def forward_fee(listing_price: Optional[Decimal]) -> Optional[Decimal]:
    if listing_price is None or listing_price < Decimal('0.00'):
        return None
    return quantize_2((listing_price * (Decimal('1.00') + FEE_RATE)) + FIXED_FEE)


def matches_fee_formula(gross_price: Optional[Decimal], listing_price: Optional[Decimal]) -> bool:
    if gross_price is None or listing_price is None:
        return False
    return forward_fee(listing_price) == quantize_2(gross_price)


def is_roundish_listing_price(value: Optional[Decimal]) -> bool:
    if value is None:
        return False
    cents = int((value * 100) % 100)
    return cents in ROUNDISH_CENTS


def load_reference_prices(path: Optional[str]) -> dict[str, Decimal]:
    if not path:
        return {}
    ref_path = Path(path)
    if not ref_path.exists():
        return {}

    ref_df = pd.read_csv(ref_path, dtype={'Dataid': str, 'Link': str, 'Price': str})
    refs: dict[str, Decimal] = {}
    for _, row in ref_df.iterrows():
        token = extract_numeric_token(row.get('Price'))
        price = normalize_two_decimals(token)
        if price is None:
            continue
        dataid = str(row.get('Dataid', '')).strip()
        link = str(row.get('Link', '')).strip()
        if dataid and dataid.lower() != 'nan':
            refs[f'dataid:{dataid}'] = price
        if link and link.lower() != 'nan':
            refs[f'link:{link}'] = price
    return refs


def choose_by_reference(candidates: list[tuple[str, Optional[Decimal]]], reference_price: Optional[Decimal]) -> Optional[tuple[str, Decimal]]:
    if reference_price is None:
        return None
    valid = [(name, value) for name, value in candidates if value is not None]
    if not valid:
        return None
    best_name, best_value = min(valid, key=lambda item: abs(item[1] - reference_price))
    if abs(best_value - reference_price) <= Decimal('0.01'):
        return best_name, best_value
    return None


def decide_price_repair(raw_value: object, reference_price: Optional[Decimal] = None) -> RepairDecision:
    raw_token = extract_numeric_token(raw_value)
    places = decimal_places(raw_token)
    normalized = normalize_two_decimals(raw_token)
    separator_fixed = recover_separator_collapsed_price(raw_token)
    fee_from_normalized = invert_fee(normalized)
    fee_from_separator = invert_fee(separator_fixed)

    candidates = [
        ('normalized', normalized),
        ('separator_recovered', separator_fixed),
        ('fee_inverted_normalized', fee_from_normalized if matches_fee_formula(normalized, fee_from_normalized) else None),
        ('fee_inverted_separator', fee_from_separator if matches_fee_formula(separator_fixed, fee_from_separator) else None),
    ]

    ref_choice = choose_by_reference(candidates, reference_price)
    if ref_choice is not None:
        reason, value = ref_choice
        return RepairDecision(value, f'{reason}_matched_old_df', 'high', normalized, separator_fixed, fee_from_normalized, fee_from_separator, places)

    if normalized is None:
        return RepairDecision(None, 'unparsed', 'low', None, None, None, None, places)

    if places is None:
        return RepairDecision(normalized, 'normalize_only', 'low', normalized, separator_fixed, fee_from_normalized, fee_from_separator, places)

    if places <= 2:
        return RepairDecision(normalized, 'normalize_only', 'high', normalized, separator_fixed, fee_from_normalized, fee_from_separator, places)

    if places == 3:
        if fee_from_normalized is not None and matches_fee_formula(normalized, fee_from_normalized) and fee_from_normalized >= Decimal('1.00') and separator_fixed is not None:
            ratio = separator_fixed / normalized if normalized != Decimal('0.00') else Decimal('Infinity')
            if ratio >= Decimal('100'):
                return RepairDecision(fee_from_normalized, 'fee_inverted_from_normalized', 'medium', normalized, separator_fixed, fee_from_normalized, fee_from_separator, places)
        if separator_fixed is not None:
            return RepairDecision(separator_fixed, 'separator_recovered', 'medium', normalized, separator_fixed, fee_from_normalized, fee_from_separator, places)
        return RepairDecision(normalized, 'normalize_only', 'medium', normalized, separator_fixed, fee_from_normalized, fee_from_separator, places)

    if separator_fixed is not None and fee_from_separator is not None and matches_fee_formula(separator_fixed, fee_from_separator) and is_roundish_listing_price(fee_from_separator):
        return RepairDecision(fee_from_separator, 'separator_recovered_then_fee_inverted', 'medium', normalized, separator_fixed, fee_from_normalized, fee_from_separator, places)

    if separator_fixed is not None:
        return RepairDecision(separator_fixed, 'separator_recovered', 'high', normalized, separator_fixed, fee_from_normalized, fee_from_separator, places)

    return RepairDecision(normalized, 'normalize_only', 'medium', normalized, separator_fixed, fee_from_normalized, fee_from_separator, places)


def find_reference_price(row: pd.Series, reference_prices: dict[str, Decimal]) -> Optional[Decimal]:
    dataid = str(row.get('Dataid', '')).strip()
    link = str(row.get('Link', '')).strip()
    if dataid and dataid.lower() != 'nan':
        hit = reference_prices.get(f'dataid:{dataid}')
        if hit is not None:
            return hit
    if link and link.lower() != 'nan':
        return reference_prices.get(f'link:{link}')
    return None


def build_outputs(df: pd.DataFrame, reference_prices: dict[str, Decimal]) -> tuple[pd.DataFrame, pd.DataFrame]:
    repaired_rows = []
    review_rows = []

    for _, row in df.iterrows():
        reference_price = find_reference_price(row, reference_prices)
        decision = decide_price_repair(row.get('Price'), reference_price=reference_price)

        repaired_row = row.copy()
        if decision.final_price is not None:
            repaired_row['Price'] = float(decision.final_price)
        repaired_rows.append(repaired_row)

        review_rows.append({
            'Dataid': row.get('Dataid'),
            'Title': row.get('Title'),
            'Link': row.get('Link'),
            'OriginalPriceRaw': row.get('Price'),
            'NormalizedPrice': float(decision.normalized_price) if decision.normalized_price is not None else None,
            'SeparatorRecoveredPrice': float(decision.separator_recovered_price) if decision.separator_recovered_price is not None else None,
            'FeeInvertedNormalizedPrice': float(decision.fee_inverted_normalized_price) if decision.fee_inverted_normalized_price is not None else None,
            'FeeInvertedSeparatorPrice': float(decision.fee_inverted_separator_price) if decision.fee_inverted_separator_price is not None else None,
            'FinalPrice': float(decision.final_price) if decision.final_price is not None else None,
            'FixReason': decision.reason,
            'Confidence': decision.confidence,
            'RawDecimalPlaces': decision.raw_decimal_places,
            'OldDfReferencePrice': float(reference_price) if reference_price is not None else None,
        })

    return pd.DataFrame(repaired_rows, columns=df.columns), pd.DataFrame(review_rows)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(f'{input_path.stem}_fixed.csv')
    review_path = Path(args.review) if args.review else input_path.with_name(f'{input_path.stem}_price_review.csv')

    df = pd.read_csv(input_path, dtype={'Dataid': str, 'Link': str, 'Price': str})
    reference_prices = load_reference_prices(args.old_df)
    repaired_df, review_df = build_outputs(df, reference_prices)

    repaired_df.to_csv(output_path, index=False)
    review_df.to_csv(review_path, index=False)

    print(f'Wrote repaired CSV to {output_path}')
    print(f'Wrote review CSV to {review_path}')
    print(review_df['FixReason'].value_counts(dropna=False).to_string())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
