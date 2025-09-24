import csv

word_sample = set({})
with open('sample_classes.txt') as sample_txt:
    content = sample_txt.readlines()
    for line in content:
        word_sample.add(line[:-1])

with open('random_sample.csv', 'w') as output:
    writer = csv.writer(output)
    writer.writerow(['split', 'file', 'gloss'])
    for split in ['train', 'val', 'test']:
        with open(f'../ASL_Citizen/splits/{split}.csv', 'r') as input:
            reader = csv.reader(input)
            next(reader)

            for line in reader:
                if line[2] in word_sample:
                    writer.writerow([split, line[1][:-4], line[2]])