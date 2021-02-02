import csv


def load_original_strings(paths):
    strings = {}

    for key, val in paths.items():
        with open(val) as f:
            reader = csv.reader(f)
            strings[key] = list(reader)

    return strings


def get_all_chars(strings):
    all_chars = []
    for key, val in strings.items():
        for val_ in val:
            for c_ in val_:
                for c in c_:
                    all_chars.append(c)

    return all_chars


def get_unique_chars(all_chars):
    return sorted(list(set(all_chars)))


paths = {'train': './DEBUG/MELD.Raw/train_sent_emo.csv',
         'dev': './DEBUG/MELD.Raw/dev_sent_emo.csv',
         'test': './DEBUG/MELD.Raw/test_sent_emo.csv'}

strings = load_original_strings(paths)
all_chars = get_all_chars(strings)
unique_chars = get_unique_chars(all_chars)

print(f'there are in total of {len(unique_chars)} characters used: ')
print(unique_chars)
print()

"""
85	…	2026	HORIZONTAL ELLIPSIS
91	‘	2018	LEFT SINGLE QUOTATION MARK
92	’	2019	RIGHT SINGLE QUOTATION MARK
93	“	201C	LEFT DOUBLE QUOTATION MARK
94	”	201D	RIGHT DOUBLE QUOTATION MARK
96	–	2013	EN DASH
97	—	2014	EM DASH
A0	 	00A0	NO-BREAK SPACE
"""
cp1252_to_utf8 = {'\x85': "…",
                  '\x91': "‘",
                  '\x92': "’",
                  '\x93': "“",
                  '\x94': "”",
                  '\x96': "–",
                  '\x97': "—",
                  '\xa0': " "}

strings_converted = {}

for key, val in strings.items():

    strings_converted[key] = []
    for i, val_ in enumerate(val):

        strings_converted[key].append([])
        for j, c_ in enumerate(val_):

            strings_converted[key][i].append([])
            for k, c in enumerate(c_):
                # strings_converted[key][i][j].append([])
                try:
                    target = cp1252_to_utf8[c]
                    # print(f"{strings[key][i][j]} will be")
                    strings_converted[key][i][j] += target
                    # print(f"{strings[key][i][j]}")

                except KeyError:
                    strings_converted[key][i][j] += c

            strings_converted[key][i][j] = ''.join(
                strings_converted[key][i][j])


for (key, val), (keyc, valc) in zip(strings.items(), strings_converted.items()):
    assert key == keyc

    for (i, val_), (ic, val_c) in zip(enumerate(val), enumerate(valc)):
        assert i == ic

        for (j, c_), (jc, c_c) in zip(enumerate(val_), enumerate(val_c)):
            assert j == jc
            assert len(c_) == len(c_c)

all_chars = get_all_chars(strings_converted)
unique_chars = get_unique_chars(all_chars)

print("conversion finished.")
print()

print(f'there are in total of {len(unique_chars)} characters used: ')
print(unique_chars)
print()


paths = {'train': './DEBUG/MELD.Raw/train_sent_emo_converted.csv',
         'dev': './DEBUG/MELD.Raw/dev_sent_emo_converted.csv',
         'test': './DEBUG/MELD.Raw/test_sent_emo_converted.csv'}


for key, val in strings_converted.items():
    with open(paths[key], 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(val)
