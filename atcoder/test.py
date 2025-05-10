import os

for k, v in os.environ.items():
    if k == "PATH":
        continue
    print(f"{k}: {v}")
