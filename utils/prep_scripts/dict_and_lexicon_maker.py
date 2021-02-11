# Usage: python dict_and_lexicon_maker.py --wrd <path to train.wrd>train.wrd --lexicon <destnation path>/lexicon.lst --dict <destnation path>/dict.ltr.txt

import numpy as np
import sys
import argparse

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--wrd", type = str, help = "path to wrd file")
	parser.add_argument("--lexicon", default = 'lexicon.lst', type = str, help = "path to wrd file")
	parser.add_argument("--dict", default = 'dict.ltr.txt', type = str, help = "path to wrd file")

	args = parser.parse_args()

	text_lines=[]

	with open(args.wrd, mode = 'r', encoding='utf-8') as file_local:
		text_lines = file_local.readlines()

	total_words = " ".join(text_lines).split(" ")
	total_words = [local_word.strip() for local_word in total_words]
	unique_words = np.unique(total_words)

	unique_characters_dict = []

	with open(args.lexicon, mode='w+', encoding='utf-8') as file_lexicon:
		for local_word in unique_words:
			if local_word != "":
				unique_characters = list(local_word)
				unique_characters_dict.extend(unique_characters)
				print(local_word + "\t" +  " ".join( unique_characters ) + " |", file=file_lexicon) 
	
	print("** Lexicon File Created")


	unique_character_set =['|']
	unique_character_set.extend( np.unique(unique_characters_dict) )
	print("** Dictionary of length ", len(unique_character_set)," created as:")
	print(unique_character_set)


	total_character_set = []
	for word in total_words:
		characters = list(word)
		total_character_set.extend(characters)

	with open(args.dict, mode='w+', encoding='utf-8') as file_dict:
		for i in unique_character_set:
			print(i +" "+ str(total_character_set.count(i)), file=file_dict)




if __name__ == "__main__":
	main()