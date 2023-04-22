import re

def find_shortest(l):
    o = [len(x) for x in re.findall(r'[a-zA-Z]+',l)]
    return min(o) if o else 0