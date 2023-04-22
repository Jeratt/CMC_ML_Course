from typing import List


def hello(x: str) -> str:
    if x is None or x == "":
        return "Hello!"
    else:
        return "Hello, "+x+"!"

def int_to_roman(num: int) -> str:
    out = ""
    while x >= 1:
        if x >= 1000:
            out += (x // 1000) * "M"
            x %= 1000
        elif x >= 500:
            if x >= 900:
                out += "CM"
                x %= 100
            else:
                out += "D"
                x -= 500
        elif x >= 400:
            out += "CD"
            x -= 400
        elif x >= 100:
            out += (x // 100) * "C"
            x %= 100
        elif x >= 50:
          if x >= 90:
              out += "XC"
              x %= 10
          else:
              out += "L"
              x -= 50
        elif x >= 40:
            out += "XL"
            x -= 40
        elif x >= 10:
            out += (x // 10) * "X"
            x %= 10
        else:
            if x == 9:
                out += "IX"
                x = 0
            elif x >= 5:
                out += "V" + "I" * (x - 5)
                x = 0
            else:
                if x == 4:
                    out += "IV"
                    x = 0
                else:
                    out += "I" * x
                    x = 0
    return out


def longest_common_prefix(strs_input: List[str]) -> str:
    pass


def primes() -> int:
    yield


class BankCard:
    def __init__(self, total_sum: int, balance_limit: int):
        pass

