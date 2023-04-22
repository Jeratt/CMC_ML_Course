def check(x: str, file: str):
    with open(file, "w") as fout:
        lst = [y.lower() for y in x.split()]
        lst.sort()
        out = {}
        used = set()
        for word in lst:
            if word not in used:
                used.add(word)
                out[word] = 1
            else:
                out[word] += 1
        for word, cnt in out.items():
            fout.write(word+" "+str(cnt)+"\n")
