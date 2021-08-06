import sys
print(sys.path)

import two_afc

embedding = two_afc.get_glove_100d()
target_occupations = ['accountant supervisor','accountant','supervisor']
female_male_pairs = [['woman', 'man'],
                     ['female', 'male'],
                     ['she', 'he'],
                     ['her', 'him'],
                     ['hers', 'his'],
                     ['daughter', 'son'],
                     ['girl', 'boy'],
                     ['sister', 'brother']]
                
# print(two_afc.pse(embedding, target_occupations, female_male_pairs))
print(two_afc.pse(embedding, target_occupations, female_male_pairs))
