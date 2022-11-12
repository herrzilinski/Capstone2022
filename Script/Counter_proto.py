
import os
import pandas as pd
import zipfile
import time
import re
import xml.etree.ElementTree as ET
from joblib import Parallel, delayed
from collections import Counter, defaultdict
from pattern.en import pluralize, singularize
import numpy as np

OR_PATH = os.getcwd()
os.chdir("..")
PATH = os.getcwd()
DATA_DIR = PATH + os.path.sep + 'Data' + os.path.sep
OUT_DIR = PATH + os.path.sep + 'Output' + os.path.sep

os.chdir(OR_PATH)

# Input Folder (path): Burning Glass text data
# text_dir_bg = 'C:\\Users\\wb560770\\OneDrive - WBG\\Documents\\DECTI\\2_Raw_data\\BGT\\US\\Raw\\'

# Working directory
# os.chdir("C:\\Users\\wb560770\\OneDrive - WBG\\Documents\\DECTI")

# Input Folder (path): keywords excel file
# Folder in working directory
# in_dir = '2_Raw_data\\LCT\\'

# Output Folder (path): save counts
# Folder in working directory
# out_dir_counts = '6_Constructed_datasets\\BGT\\US\\lct\\'

# Define for which years to count keywords in Burning Glass?
# (below, counts for 2007 and 2010-2019; jumps 2008 and 2009)
start_year = 2010
end_year = 2010
jump_year = []

# ###########################################################
# #------------------------Functions------------------------#
# ###########################################################
"""
    Define auxiliary functions to later count keywords.
"""

def clean_string(mystring):
    """
    keep only recognized characters
    """
    mystring = r' ' + mystring.lower() + r' '
    mystring = re.sub('[^a-z ]', ' ', mystring)
    grams = mystring.split(' ')
    grams = [x for x in grams if x != ""]
    if len(grams) > 100:
        del grams[:50 + 1]
        del grams[(len(grams) - 50 - 1): (len(grams) + 1)]
    else:
        return []
    # return bigrams and unigrams
    bigrams = [' '.join(b) for b in zip(grams[:-1], grams[1:])]
    all_grams = bigrams + grams
    return all_grams

def process_parsed_object(parsed_object, keywords):
    """
    take an xml parsed object and return the required elements
    """
    jobtext = clean_string(parsed_object.find('JobText').text)
    found = [x for x in jobtext if x in keywords]
    if not found:
        return {}

    job_id = '0'
    for child in parsed_object:
        # get id
        if child.tag == "JobID":
            job_id = child.text
        # get text
        if child.tag == "JobText":
            mystring = child.text
            mystring = mystring.lower()
            mystring = re.sub('[^a-zA-Z ]', ' ', mystring)
            jobtext = jobtext
        if child.tag == "BGTOcc":
            occ = child.text
            if not occ:
                occ = ''
        if child.tag == "CanonJobTitle":
            job_title = child.text
            if not job_title:
                job_title = ''

    # output
    return {'job_text': jobtext, 'job_id': job_id, 'job_title': job_title, 'occ': occ, 'text': mystring}


def write_output(file_write_name, found_tech_counter, parsed_values, out_dir, basiswords, techs_to_write):
    for tech in techs_to_write:
        with open(out_dir + 'counts' + os.path.sep + tech + os.path.sep + file_write_name, 'a') as ofile:
            ofile.write(parsed_values['job_id'])
            for word in basiswords[tech]:
                if found_tech_counter[tech][word] > 0:
                    ofile.write('|' + str(found_tech_counter[tech][word]))
                    # write snippets
                    if np.random.choice([0, 1], 1, p=[0.999, 0.001]):
                        outfile = open(out_dir + 'snippets' + os.path.sep + tech + os.path.sep + file_write_name, 'a')
                        outfile.write(parsed_values['job_id'] + ',' + word + ',' +
                                      str(parsed_values['job_title']) +
                                      ',' + str(parsed_values['occ']) + ',' +
                                      parsed_values['text'] + '\n')
                        outfile.close()
                else:
                    ofile.write('|')
            ofile.write('\n')


def count_words(text_dir, year, file, out_dir, keywords, basiswords):
    """
    count words and write into a directory
    """
    # pre requirements
    # build list dictionary, from keywords to technologies
    word2tech = defaultdict(list)
    for tech in basiswords:
        for word in basiswords[tech]:
            word2tech[word].append(tech)

    # identifying xml tags for jobs
    id_tag = '<Job>'

    finish_tag = r'</Job>'

    p = re.compile(id_tag)

    zip_file = zipfile.ZipFile(text_dir + year + os.path.sep + file)
    zip_files = zip_file.namelist()

    # check if zipfile has multiple files
    if len(zip_files) > 1:
        raise Exception("zipfile has multiple files")

    i = 0
    # write week to week in this file
    file_write_name = re.sub('.zip', '.csv', file)

    # initialize files
    for tech in basiswords:
        # count file
        with open(out_dir + 'counts' + os.path.sep + tech + os.path.sep + file_write_name, 'w+') as ofile:
            ofile.write("job_id|" + '|'.join(basiswords[tech]) + '\n')
        # snippet file
        outfile = open(out_dir + 'snippets' + os.path.sep + tech + os.path.sep + file_write_name, 'w+')
        outfile.write('job_id,keyword,job_title,occupation,jobtext\n')
        outfile.close()

    # open file and count keywords
    with zip_file.open(zip_file.namelist()[0], 'r') as infile:
        for line_nr, line in enumerate(infile, start=1):

            line_string = str(line.decode('latin'))

            # does the line have job id?
            try:
                # start collecting xml string...
                m = p.search(line_string)
                pid = m.group(0)
                final_string = line_string
            except AttributeError:
                try:
                    final_string = final_string + line_string
                    # check if job text is finished
                    if re.search(finish_tag, line_string):
                        # printing progress
                        i = i + 1
                        if not i % 100_000:
                            print("done with job postings: " + str(i))

                        try:
                            # parsing job text
                            parsed_object = ET.fromstring(final_string)

                            # get values
                            parsed_values = process_parsed_object(parsed_object, keywords)

                            if len(parsed_values) == 0:
                                continue

                            found = [x for x in parsed_values['job_text'] if x in keywords]

                            # take job id and store the count
                            if found:
                                # write total counts
                                found_counter = Counter(found)
                                # transform counter
                                found_tech_counter = defaultdict(Counter)
                                techs_to_write = set()
                                for word in found_counter:
                                    mytechs = word2tech[keywords[word]]
                                    for mytech in mytechs:
                                        found_tech_counter[mytech].update({keywords[word]: found_counter[word]})
                                        techs_to_write.add(mytech)

                                # write output
                                write_output(file_write_name, found_tech_counter,
                                             parsed_values, out_dir, basiswords,
                                             techs_to_write)

                        # if a job cannot be processed
                        except (ET.ParseError, UnicodeEncodeError):
                            print('not processed')
                except (UnboundLocalError, AttributeError, UnicodeEncodeError):
                    pass

def count_words_mp(myinput):
    """
    runs count function in multiprocessing
    """
    text_dir_1 = myinput[0]
    year_1 = myinput[1]
    file_1 = myinput[2]
    out_dir_1 = myinput[3]
    keywords_1 = myinput[4]
    basiswords_1 = myinput[5]
    count_words(text_dir_1, year_1, file_1, out_dir_1, keywords_1, basiswords_1)

def get_all_ps(bigram):
    if len(bigram.split()) == 1:
        if singularize(bigram) == bigram:
            bigram_mod = pluralize(bigram)
        else:
            bigram_mod = singularize(bigram)
        return [bigram, bigram_mod]
    part1 = bigram.split(' ')[0]
    part2 = bigram.split(' ')[1]
    if singularize(part1) == part1:
        part1_mod = pluralize(part1)
    else:
        part1_mod = singularize(part1)
    if singularize(part2) == part2:
        part2_mod = pluralize(part2)
    else:
        part2_mod = singularize(part2)

    all_forms = []
    for a in part1, part1_mod:
        for b in part2, part2_mod:
            all_forms.append(a + ' ' + b)
    return all_forms

# ###########################################################
# #-----------------Dictionary of Keywords------------------#
# ###########################################################
"""
    Get list of keywords and topics from input Excel file,
    and create a dictionary of keywords to topics to be
    counted later.
"""
keywords = {}
basiswords = {}
# collecting word list from audits
# keywords to topic - in order to search for them together
df = pd.ExcelFile(OR_PATH + os.path.sep + 'keywords.xlsx')
techlist = df.sheet_names
for basis in techlist:
    mydf = pd.read_excel(OR_PATH + os.path.sep + 'keywords.xlsx',
                         sheet_name=basis)
    # dictionary: technologies to bigrams
    basiswords[basis] = list(set([(re.sub('[0-9]', '', re.sub('_', ' ', x)))
                             for x in mydf.bigrams]))
    print("tech: " + basis + ', found: ' + str(len(basiswords[basis])))
    for word in basiswords[basis]:
        if len(word) <= 5:
            keywords = {**keywords, **{word: word}}
            continue
        # dictionary: word in plural/singular forms to word
        keywords = {**keywords, **{x: word for x in get_all_ps(word)}}

# ###########################################################
# #------------------------Counting-------------------------#
# ###########################################################
"""
    Count keywords in Burning Glass data.
"""

start = time.time()

years = [str(i) for i in range(start_year, end_year + 1) if i not in jump_year]

# initialize output folder names
for tech in techlist:
    if not os.path.exists(out_dir_counts + 'counts\\' + tech):
        os.makedirs(out_dir_counts + 'counts\\' + tech)

for tech in techlist:
    if not os.path.exists(out_dir_counts + 'snippets\\' + tech):
        os.makedirs(out_dir_counts + 'snippets\\' + tech)

# Count keywords in BG data
for year in years:
    files = os.listdir(text_dir_bg + year)
    print("doing year: " + str(year))
    print("files in folder: " + str(len(files)))
    chunk_size = 60
    n_chunks = int(len(files)/chunk_size)
    print(n_chunks)
    for chunk in range(0, (n_chunks+1)):
        if chunk == n_chunks-1:
            select_files = files[(chunk*chunk_size):len(files)]
        else:
            select_files = files[(chunk*chunk_size):((chunk+1)*chunk_size)]
        print("doing chunk: " + str(chunk))
        start = time.time()
        Parallel(n_jobs=1, verbose=10, backend='loky')(delayed(
            count_words_mp)([text_dir_bg, year, file, out_dir_counts, keywords, basiswords]) for file in select_files)
        print('Total time: ' + '{:.2f}'.format((time.time() - start) / 60) + 'min')

