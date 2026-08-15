def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def validate_pagination(limit: int, offset: int) -> tuple[int, int]:
    limit = int(clamp(limit, 1, 200))
    offset = max(0, int(offset))
    return limit, offset
