import pandas as pd

#Leukemia data set
#data=pd.read_table('LEUKEMIA\leukemia_test_34x7129.txt',delimiter=",",header=None)
#data=pd.read_table('LEUKEMIA\leukemia_train_38x7129.txt',delimiter=",",header=None)
#data=pd.read_table('LEUKEMIA\dataset_60x7129.txt',delimiter=",",header=None)

#Colon cancer
#data=pd.read_table('COLONCANCER\colon.txt',delimiter=",",header=None)

#LYMPHOMA
#data=pd.read_table('LYMPHOMA\Lymphoma45x4026_2classes.txt',delimiter=",",header=None)
#data=pd.read_table('LYMPHOMA\Lymphoma96x4026_9classes.txt',delimiter=",",header=None)
#data=pd.read_table('LYMPHOMA\Lymphoma96x4026_10classes.txt',delimiter=",",header=None)

#ECML2004
#data=pd.read_table('ECML2004\ECML90x27679.txt',delimiter=",",header=None)

#GLOBALCANCERMAP
#data=pd.read_table('GLOBALCANCERMAP\GCM_Test.txt',delimiter=",",header=None)
data=pd.read_table('GLOBALCANCERMAP\GCM_Training.txt',delimiter=",",header=None)

print(data.shape)