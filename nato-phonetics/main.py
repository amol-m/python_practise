import pandas as pd

nato_phonetic_alphabet = pd.read_csv('nato_phonetic_alphabet.csv')

in_word = input('Enter your word')

# lst_in_word = [ n for n in in_word]
# print(lst_in_word)
# output_dict = { in_word.upper():row.code for (index, row) in nato_phonetic_alphabet.iterrows() for in_word in lst_in_word if in_word.lower() == row.letter.lower()
# }
#
# print(output_dict)

phonetic_dict ={ row.letter : row.code for (index, row) in nato_phonetic_alphabet.iterrows() }

lst_in_word = [ phonetic_dict[letter] for letter in in_word]
print(lst_in_word)
