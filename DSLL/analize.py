import pandas as pd
# import numpy as np
print("Start Program")
# df_leda = pd.read_excel(r'LEDAexport_20200224_verrijkt_voorPepijn.xlsx')
xls = pd.ExcelFile('LEDAexport_20200224_verrijkt_voorPepijn.xlsx')
df_leda = pd.read_excel(xls, 'LEDA_20200224')

print(type(df_leda),len(df_leda))
print(df_leda.columns)
# length of 100186
# 1038 J for "VITB als ingredient"
# 755 J for "ijzer als toegevoegd nutrient"
# 425 J for "FOLAC toegevoegd"
print(df_leda["VITB6 als ingredient"].value_counts())
print("=" * 30)
print(df_leda["ijzer als toegevoegd nutrient"].value_counts())
print("=" * 30)
print(df_leda["FOLAC als ingredient SW"].value_counts())
print("=" * 30)
# print(df_leda["CPV_INGREDIENTTEKST"])
print("=" * 30)
print(df_leda["CPV_BEREIDINGSINSTRUCTIE"].value_counts())

# s = s.replace(',', '')
# s = s.replace('.', '')
# a['CPV_INGREDIENTENTEKST'].str.contains('vitaminea')

mindict = {}
mindict["ingredienten"] = ""
mindict['retinol'] = 'vitaminea'
mindict['retinylacetaat'] = 'vitaminea'
mindict['retinylpalmitaat'] = 'vitaminea'
mindict['beta-caroteen'] = 'vitaminea'
mindict['cholecalciferol'] = 'vitamined'
mindict['ergocalciferol'] = 'vitamined'
mindict['ergocalciferol'] = 'vitaminee'
# B6
mindict['pyridoxinehydrochloride'] = 'vitamineb6'
mindict['pyridoxine-5\'-fosfaat'] = 'vitamineb6'
mindict['pyridoxinedipalmitaat'] = 'vitamineb6'
# IJZER - contains ijzer
mindict['ijzer(II)carbonaat'] = 'ijzer'
mindict['ijzer(II)citraat'] = 'ijzer'
mindict['ijzer(III)ammoniumcitraat'] = 'ijzer'
mindict['ijzer(II)gluconaat'] = 'ijzer'
mindict['E579'] = 'ijzer'
mindict['ijzer(II)fumaraat'] = 'ijzer'
mindict['natriumijzer(III)difosfaat'] = 'ijzer'
mindict['ijzer(II)lactaat'] = 'ijzer'
mindict['E585'] = 'ijzer'
mindict['ijzer(II)sulfaat'] = 'ijzer'
mindict['ijzer(III)difosfaat'] = 'ijzer'
mindict['ijzer(III)pyrofosfaat'] = 'ijzer'
mindict['ijzer(III)saccharaat'] = 'ijzer'
# FOLAC
mindict['pteroylmonoglutaminezuur'] = 'foliumzuur'
mindict['vitamineb11'] = 'foliumzuur'
