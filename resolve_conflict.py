with open("data/attack.txt", "r") as f:
    lines = f.readlines()

resolved_lines = []
skip = False
for line in lines:
    if line.startswith("<<<<<<<"):
        continue
    elif line.startswith("======="):
        continue
    elif line.startswith(">>>>>>>"):
        continue
    else:
        resolved_lines.append(line)

with open("data/attack.txt", "w") as f:
    f.writelines(resolved_lines)
