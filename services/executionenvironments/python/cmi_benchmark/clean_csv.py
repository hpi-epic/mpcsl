
input_file = open("cmi_test.csv", "r")
output_file = open("cleaned_cmi_test.csv", "w")

for line in input_file:
    new_line = line.replace(" ", "")
    new_line = new_line.replace(",",".")
    output_file.write(new_line)


