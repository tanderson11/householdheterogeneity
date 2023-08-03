import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os

stem = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

to_munge = "bneibrak"

paths = {
	"geneva":"empirical/geneva/empirical_df.parquet",
	"bneibrak":"empirical/BneiBrak/empirical_df.parquet",
	"ontario":"empirical/Ontario/empirical_df.parquet",
}

path = paths[to_munge]

if to_munge == "jing":
	only_household = True
	drop_rare_households = True


	# Munging for Jing Guangzhou Oct 2020 retrospective
	df = pd.read_excel("~/covidhouseholds/empirical/ShahReview/Jing Household secondary attack rate of COVID-19 and associated determinants in Guangzhou/Data/Data_For_Statistical_Analysis_New.xlsx")
	# selecting down to households + close relatives
	df = df[df["relation_with_primary_case"] != "non-household member"]
	if only_household: # remove close relatives if we want only same address
		df = df[df["relation_with_primary_case"] != "close relative"]

	# counting total size of each cluster (when not counting non-household members)
	sizes = df.groupby("clusterid").size()

	print(sizes.value_counts().to_dict())

	# APPROXIMATE : drop clusters with count == 1 (in practice, clusters with size > 11); controlled by drop_rare_households
	too_few_samples = sizes.value_counts().index[sizes.value_counts() == 1]

	if drop_rare_households:
		sizes = sizes[~ sizes.isin(too_few_samples)]
	sizes = sizes[sizes != 1] # drop size = 1, as that has no relevance for us

	# choosing only those who were infected 
	infections = df[df["case_type"] != "non-case"].groupby("clusterid").size()

	empirical_df = pd.concat([infections, sizes], axis=1).dropna()
	empirical_df.columns = ["infections", "size"]

	# dummy entries for true parameters
	empirical_df["inf_var"] = 0
	empirical_df["sus_var"] = 0
	empirical_df["hsar"] = 0
	empirical_df["model"] = "baseline model"
	empirical_df["trialnum"] = "empirical"

elif to_munge == "geneva":

	size_col = [2]*(52+30) + [3]*(29+8+2) + [4]*(38+4+5+4) + [5]*(10+2+1+0+1)
	infections_col = [1]*52 + [2]*30 + [1]*29 + [2]*8 + [3]*2 + [1]*38 + [2]*4 + [3]*5 + [4]*4 + [1]*10 + [2]*2 + [3]*1 + [5]*1

	empirical_df = pd.DataFrame({"size":size_col, "infections":infections_col})

	empirical_df["inf_var"] = 0
	empirical_df["sus_var"] = 0
	empirical_df["hsar"] = 0
	empirical_df["model"] = "baseline model"
	empirical_df["trialnum"] = "empirical"

elif to_munge == "bneibrak":
	df = pd.read_csv(os.path.join(stem, "empirical/BneiBrak/households_dat.csv"))
	# building the columns (and forcing them to lists so the indices don't mess up the concat)
	size_col = df.groupby("household").size().to_list()
	infections_in_household = df.groupby("household").apply(lambda group: group.apply(lambda row: (row == "POS").any(), axis=1).value_counts()).reset_index()
	infections_col = infections_in_household[infections_in_household["level_1"] == True][0].to_list() # the number of True infected individuals in each household: 0 = column name that now contains infections

	empirical_df = pd.DataFrame({"size":size_col, "infections":infections_col})
	empirical_df["inf_var"] = 0
	empirical_df["sus_var"] = 0
	empirical_df["hsar"] = 0
	empirical_df["model"] = "baseline model"
	empirical_df["trialnum"] = "empirical"

elif to_munge == "ontario":
	df = pd.read_csv(os.path.join(stem, "empirical/Ontario/sizes-2-7-ontario.csv"))
	chunks = []
	for index, row in df.iterrows():
		count = int(row['count'])
		sizes = [row['household contacts']+1 for i in range(count)]
		infections = [row['infections'] for i in range(count)]
		chunk = pd.DataFrame({'size':sizes, 'infections':infections})
		chunks.append(chunk)
		#import pdb; pdb.set_trace()
	empirical_df = pd.concat(chunks)
	empirical_df["inf_var"] = 0
	empirical_df["sus_var"] = 0
	empirical_df["hsar"] = 0
	empirical_df["model"] = "baseline model"
	empirical_df["trialnum"] = "empirical"

	# building the columns (and forcing them to lists so the indices don't mess up the concat)
#import pdb; pdb.set_trace()
pq.write_table(pa.Table.from_pandas(empirical_df), path)


