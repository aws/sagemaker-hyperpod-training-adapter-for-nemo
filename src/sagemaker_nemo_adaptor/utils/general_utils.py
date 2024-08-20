def is_power_of_two(n: int) -> bool:
    "Brian Kernighan's Algorithm"
    return n > 0 and n & (n - 1) == 0
