from typing import List


def hello(x: str = None) -> str:
    if x is None or x == "":
        return "Hello!"
    else:
        return f"Hello, {x}!"


def int_to_roman(x: int) -> str:
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


def longest_common_prefix(lst: List[str]) -> str:
    if not len(lst):
        return ""
    for i in range(len(lst)):
        lst[i] = lst[i].lstrip()
    dp = lst[0]
    for i in range(1, len(lst)):
        tmp = ""
        for j in range(min(len(lst[i]), len(dp))):
            if lst[i][j] == dp[j]:
                tmp += dp[j]
            else:
                break
        dp = tmp
    return dp


def primes() -> int:
    prim_lst = set()
    x = 2
    while True:
        key = True
        for i in prim_lst:
            if not x % i:
                key = False
                break
        if key:
            prim_lst.add(x)
            yield x
        x += 1


class BankCard:
    def __init__(self, total_sum: int, balance_limit: int = 10000000):
        self.total_sum = total_sum
        self.balance_limit = balance_limit

    def __call__(self, sum_spent):
        try:
            if sum_spent > self.total_sum:
                raise ValueError
            else:
                self.total_sum -= sum_spent
                print(f"You spent {sum_spent} dollars")
        except ValueError:
            print(f"Can't spend {sum_spent} dollars")

    def __str__(self):
        return "To learn the balance call balance."

    def __getattr__(self, item):
        if item == "balance":
            try:
                if self.balance_limit <= 0:
                    raise ValueError
                self.balance_limit -= 1
                return self.total_sum
            except ValueError:
                print("Balance check limits exceeded.")

    def put(self, sum_put):
        self.total_sum += sum_put
        print(f"You put {sum_put} dollars")

    def __add__(self, other):
        return BankCard(self.total_sum + other.total_sum, max(self.balance_limit, other.balance_limit))