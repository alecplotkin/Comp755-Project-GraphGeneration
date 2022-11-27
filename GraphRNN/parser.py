def read_parameters(fname):
    f = open(fname, 'r')
    return f.readlines()
parameters = read_parameters('test_template.txt')

default_parameters = ['enzymes_small', 'None', 'None', '16', '8', '32', '16', '32', '32', '1000', '4', '4', '32', '3000', '100', '100', '100', '100', '0.003', '[400, 1000]', '0.3', '2', './', 'False', '3000', 'True', 'BA', 'clustering']

if (len(parameters) != 29):
    print("ERROR: Malformed input file! Please retry!")
for active_params in parameters:
    if ']' not in active_params and '[' not in active_params:
        print(active_params)
    else:
        print("No value provided, ignoring...       using default parameter instead!")

# Splices all items out of brackets
# for s in parameters:
#     print(s[1:-2])
# print(parameters)
