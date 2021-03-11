file_name = "mocap_wx200.txt"

new_lines = []


with open(file_name, "r") as filestream:
    for lines in filestream:
        new_lines.append(lines)

for i in range(len(new_lines)):
    new_lines[i] = new_lines[i][:-1]
    new_lines[i] = new_lines[i] + ", 0, 0, 0, 0, 0, 0, 0.02, -0.02\n"

print(new_lines[:4])
print(len(new_lines))

with open(file_name, "w") as filestream:
    filestream.writelines(new_lines)
