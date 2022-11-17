import sys
import re
import os
# from bs4 import BeautifulSoup
import pandas as pd
from transformers import pipeline
import time

pd.set_option('max_columns', 19)
# %%
OR_PATH = os.getcwd()
os.chdir("..")  # Change to the parent directory
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
sep = os.path.sep
os.chdir(OR_PATH)  # Come back to the folder where the code resides , all files will be left on this directory

# %%
# with open(f'{DATA_DIR}/US_XML_AddFeed_20100101_20100107.xml', 'rb') as f:
#     raw = f.read()
# data = BeautifulSoup(raw, "xml")

# %%
text = "From: Company:  Providence Health &amp; Services ( ) Job Reference ID: 21228810  Category: other  Duration:  City, ST:  Missoula, Montana  Country:  United States Description: Missoula, Montana St. Patrick Hospital and Health Sciences Center and a highly regarded, physician-owned multispecialty clinic (50 providers) are partnering to bring a BC/BE Pulmonary/Critical Care/Intensivist to serve the community. Flexible practice options: Can be 100 percent Pulm outpatient joining another Pulmonologist in the clinic, or combined Pulm/Critical Care splitting time between the clinic and the ICU at St. Patrick (out of groups office down the hall from the ICU). Allergy consultation also available if interested. Nighttime hospitalist program in place. Not H-1B or J-1 visa eligible. Competitive guarantee plus excellent benefits package, including relocation. St. Patrick Hospital and Health Sciences Center, part of Providence Health and Services, is a pillar of the medical and business community in western Montana. Our 247-bed tertiary center has been named a Top 100 Hospital by Thomson (formerly Solucient). We house the world-renowned International Heart Institute of Montana and collaborate with the University of Montana and others to leverage scientific expertise for the benefit of our patients.  Missoula is a sophisticated university town of 65,000 located about halfway between Glacier and Yellowstone National Parks. The Clark Fork River runs through it, granite peaks surround it, and that famous Big Sky hangs over it all. The area offers unlimited recreational opportunities (ski, paddle, fish or hike out your back door) and numerous cultural amenities, including a symphony and an active theatre and arts scene. Missoula is a family-oriented community with great schools (elementary through university) and social activities year round. The climate is relatively mild; winters, for example, are warmer here than in much of the Midwest and Northeast. Western Montana summers are absolutely glorious, with long, adventure-inspiring days, starry nights and fresh mountain air. Providence Health and Services, a not-for-profit network of hospitals, clinics and physician partners in Alaska, California, Montana, Oregon and Washington. Providence has a proud 150-year history in the West, and we continue to grow with the communities we serve. With more than 300 physician opportunities in virtually all specialties, we offer physicians diverse lifestyle choices, flexible work arrangements and robust practice support. Learn more at www.providence.org/physicianopportunities.  Requirements: See Job Description.  Job Created: Mon Dec 28 2009 11:02:52 PM  Last Modified: Mon Dec 28 2009 11:02:52 PM  Resume Writing Get our professional resume writers to write your Pulmonologist/Critical Care Physician Needed! resume and you're 100% guaranteed to get  more interviews and job offers.  PayScale  Salary Calculator copyright 2003 - 2009 resumes2work |  Privacy Policy"

with open(f'{OR_PATH}/candidates.txt', 'r') as f:
    cand_txt = f.read()
candidates = cand_txt.split('\n')
# %%
classifier = pipeline("zero-shot-classification")

# print(classifier(text, candidate_labels=candidates,))
# %%
start = time.time()
result = classifier(text, candidate_labels=candidates,)
end = time.time()
print(f'Elapsed time for zero-shooting one job ads is {end - start:.2f}s')
res_df = pd.DataFrame(result)
print(res_df)
