data1 = {'Name':['Smith, John', 'Jennifer Tal', 'Gates, Bill', 'Alan Fitch', 'Jacob Alan'],
         'Phone':['445-881-4468', '+1-189-456-4513', '(876)546-8165', '5493156648', '(205)1564896'],
         'Date_of_Birth':['August 12,1989', '11/12/1965', 'June 15,1972', '2-6-1985', '1985 January 3'],
         'State':['Maine', 'Tx', 'Kansas', 'Oh', 'Alabama']
}

data2 ={'Name':['John Smith', 'Jennifer Tal', 'Bill Gates', 'Alan Fitch', 'Jacob Alan'],
        'Phone':['445-881-4468', '189-456-4513', '876-546-8165', '549-315-6648', '205-156-4896'],
        'Date_of_Birth':['08/12/1989', '11/12/1965', '06/15/1972', '02/06/1985', '01/03/1985'],
        'State':['Maine', 'Tx', 'Kansas', 'Oh', 'Alabama']
}

from difflib import SequenceMatcher

def trigram_similarity(str1, str2):
    def get_trigrams(s):
        return [s[i:i+3] for i in range(len(s) - 2)]
    
    trigrams1 = get_trigrams(str1)
    trigrams2 = get_trigrams(str2)
    
    matches = sum(1 for trigram in trigrams1 if trigram in trigrams2)
    total = max(len(trigrams1), len(trigrams2))
    
    return matches / total if total > 0 else 0

def bigram_similarity(str1, str2):
    def get_bigrams(s):
        return [s[i:i+2] for i in range(len(s) - 1)]
    
    trigrams1 = get_bigrams(str1)
    trigrams2 = get_bigrams(str2)
    
    matches = sum(1 for trigram in trigrams1 if trigram in trigrams2)
    total = max(len(trigrams1), len(trigrams2))
    
    return matches / total if total > 0 else 0

matches = []
for i, phone1 in enumerate(data1['Phone']):
    for j, phone2 in enumerate(data2['Phone']):
        similarity = trigram_similarity(phone1, phone2)
        if similarity > 0:  # Assuming a threshold of 0.8 for a match
            matches.append((i, j, similarity))

print("Matches based on trigram similarity of Phone numbers:")
for match in matches:
    print(f"data1 index: {match[0]}, data2 index: {match[1]}, similarity: {match[2]:.2f}")

def jaccard_similarity(str1, str2):
    set1 = set(str1.replace('/', ' ').replace('-', ' ').replace(',', ' ').split())
    set2 = set(str2.replace('/', ' ').replace('-', ' ').replace(',', ' ').split())
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union != 0 else 0

def date_of_birth_similarity(dob1, dob2):
    return jaccard_similarity(dob1, dob2)

dob_matches = []
for i, dob1 in enumerate(data1['Date_of_Birth']):
    for j, dob2 in enumerate(data2['Date_of_Birth']):
        similarity = date_of_birth_similarity(dob1, dob2)
        if similarity > 0.1:  # Assuming a threshold of 0.3 for a match
            dob_matches.append((i, j, similarity))

print("Matches based on date of birth similarity using word embedding:")
for match in dob_matches:
    print(f"data1 index: {match[0]}, data2 index: {match[1]}, similarity: {match[2]:.2f}")
