import random
body = [random.randint(1, 9)] + [random.randint(0, 9) for _ in range(8)]
digits = [int(d) for d in "80840"] + body
total = sum(sum(divmod(d * 2, 10)) if i % 2 == 0 else d for i, d in enumerate(reversed(digits)))
print("".join(map(str, body)) + str((10 - total % 10) % 10))
