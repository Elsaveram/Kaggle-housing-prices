file = open('docs/data_description.txt')
lines = file.readlines()
file.close()

first = True
new_lines = "{\n"
for line in lines:
    # TBD: Make this handle variables with a leading number like 1stFlrSF
    if line[0].isalpha():
        if first:
            first = False
        else:
            new_lines += "\t\t}\n\t},\n"

        new_lines += '\t"' + line.replace("\n","").replace(": ", '": {\n\t\t"description": "', 1)
        new_lines += '",\n\t\t"dtype": "",\n\t\t"imputation_method": "",\n\t\t"members": {\n'
    elif line[0].isspace() and len(line.strip()) > 0:
        new_lines += ('\t\t\t"' + line.replace("\t", '": { "ordinal":0, "name":"', 1).lstrip().replace("\n","") + '"},\n')

new_lines += "\t\t}\n\t},\n}"

print(new_lines)
