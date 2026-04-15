"""
Generate CSV metadata based on the DBF file sourced from the MuST-C repo. 
See https://github.com/PRBonn/MuST-C/blob/8dc67a5b05d7bc9da34ee9aca405f8371d6cd962/download_scripts/LAI_biomass_and_metadata.txt.
"""

import pandas as pd
from dbfread import DBF

dbf = pd.DataFrame(iter(DBF("metadata/md_FieldSHP.dbf")))
dbf = dbf[dbf['plot_ID'] != 0]

def make_variety(row):
    var = str(row['var']).strip()
    if var and var != 'nan':
        return f"{row['Genotype']}_{var}"
    return row['Genotype']

dbf['variety'] = dbf.apply(make_variety, axis=1)

df = dbf[['plot_ID', 'crop', 'variety']].rename(columns={
    'plot_ID': 'plot_id',
    'crop': 'species'
})

df.to_csv("metadata/plot_metadata.csv", index=False)
print("plot_metadata.csv generated.")