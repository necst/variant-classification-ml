import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

thr = 80
path = 'datasets/dbnsfp5_full.csv'
pseudo_id_thresh = 0.1
print(f"Loading {path}")
df = pd.read_csv(path, low_memory=False)

print(f"Dataset length: {len(df)}")

print("Removing population-related features")
pop_cols = df.filter(regex='^(1000Gp3|TOPMed|gnomAD|ALFA_)').columns
df_filt = df.drop(columns=pop_cols)

print("Dropping bad features")

allowed_clin_sig = set(["Uncertain significance", "Likely benign", "Pathogenic", "Likely pathogenic", "Benign"])
df_clean = df_filt[df_filt['ClinicalSignificance'].isin(allowed_clin_sig)]
real_ids = ['clinvar_id', 'Ensembl_geneid', 'Ensembl_transcriptid', 'Ensembl_proteinid', 'MutPred_protID', 'clinvar_MedGen_id', 'clinvar_OMIM_id', 'clinvar_Orphanet_id', '#AlleleID', 'GeneID', 'HGNC_ID', 'PhenotypeIDS', 'OtherIDs', 'VariationID']
df_clean = df_clean.drop(real_ids, axis=1)
useless = ['clinvar_clnsig', 'clinvar_clnsig_clean', 'ClinSigSimple', 'RCVaccession', 'clinvar_trait', 'PhenotypeList', 'Name']
df_clean = df_clean.drop(useless, axis=1)

print(f"Dropping features with less than {thr}% of full values")

full_val = []
full_lab = []
full_lv = []
total = len(df_clean)
for c in df_clean.columns.tolist():
    full_lab.append(c)
    empty = df_clean[df_clean[c] == '.']
    v = 1 - len(empty) / total
    full_val.append(v)
    full_lv.append((c, v))

sorted_lv = sorted(full_lv, key=lambda item: item[1])

i = 0
while True:
    if sorted_lv[i][1] >= thr/100:
        break
    df_clean = df_clean.drop([sorted_lv[i][0]], axis=1)
    print(f"Dropping {sorted_lv[i][0]}")
    i += 1
print(f"Final: {len(df_clean.columns.tolist())}, dropped {i}")

print("Dropping other bad features")

df_clean = df_clean.drop(columns=['hg19_chr', 'hg18_chr'], axis=1)
df_clean = df_clean.drop(columns=['aaalt_3', 'aaref_3'], axis=1)
df_clean = df_clean.replace('.', np.nan)

os.makedirs("datasets_new", exist_ok=True)
df_clean.to_csv("./datasets_new/temp.csv", index=False)
df_pc = pd.read_csv("./datasets_new/temp.csv", low_memory=False)

pseudo_id_max = pseudo_id_thresh * len(df_pc)
print(f"Dropping pseudo-ids (nunique() >= {pseudo_id_max})")

categ_dfpc = df_pc.select_dtypes(include=['object', 'category'])

pseudo_ids = []
for c in categ_dfpc.columns.tolist():
    if categ_dfpc[c].nunique() >= pseudo_id_max:
        pseudo_ids.append(c)
print(f"Pseudo-ids: {len(pseudo_ids)}")
print(pseudo_ids)
categ_noid = categ_dfpc.drop(columns=pseudo_ids)

print("Encoding categorical features")

df_lab = df_pc.drop(columns=categ_dfpc.columns.tolist())
for c in categ_noid.columns.tolist():
    le = LabelEncoder()
    c_lab = le.fit_transform(categ_noid[c])
    c_name = f"{c}_lab"
    df_lab[c_name] = c_lab
df_lab = df_lab.drop('ClinicalSignificance_lab', axis=1)
df_lab['ClinicalSignificance'] = categ_noid['ClinicalSignificance']

def balance_equal(df_in):
    vc = df_in['ClinicalSignificance'].value_counts()
    target = vc.min()
    parts = []
    for cls in vc.index.tolist():
        parts.append(df_in[df_in['ClinicalSignificance'] == cls].sample(n=target, random_state=42))
    return pd.concat(parts).sample(frac=1.0, random_state=42)

if "Unnamed: 0" in df_lab.columns:
    print("Dropping 'Unnamed: 0'")
    df_lab = df_lab.drop("Unnamed: 0", axis=1)

df_uncertain = df_lab[df_lab['ClinicalSignificance'] == 'Uncertain significance']
df_uncertain.to_csv("datasets_new/dbnfs_uncertain.csv", index=False)

def sample_test_class(df, n, clinsig):
    rows = df_lab[df_lab['ClinicalSignificance'] == clinsig].sample(n=n, random_state=42)
    df = df.drop(rows.index)
    return (rows, df)

p_rows, df_lab = sample_test_class(df_lab, 2000, "Pathogenic")
lp_rows, df_lab = sample_test_class(df_lab, 2000, 'Likely pathogenic')
unc_rows, df_lab = sample_test_class(df_lab, 2000, "Uncertain significance")
lb_rows, df_lab = sample_test_class(df_lab, 2000, "Likely benign")
b_rows, df_lab = sample_test_class(df_lab, 2000, "Benign")

df_5c_test = pd.concat([p_rows, lp_rows, unc_rows, lb_rows, b_rows], ignore_index=True)
df_5c_test.to_csv("./datasets_new/dbnfs_5c_test.csv", index=False)
df_4c_test = df_5c_test[df_5c_test["ClinicalSignificance"] != "Uncertain significance"]
df_4c_test.to_csv("./datasets_new/dbnfs_4c_test.csv", index=False)
df_3c_test = df_5c_test
df_3c_test["ClinicalSignificance"] = df_3c_test["ClinicalSignificance"].replace("Likely pathogenic", "Pathogenic")
df_3c_test["ClinicalSignificance"] = df_3c_test["ClinicalSignificance"].replace("Likely benign", "Benign")
df_3c_test.to_csv("./datasets_new/dbnfs_3c_test.csv", index=False)
df_2c_test = df_3c_test[df_3c_test["ClinicalSignificance"] != "Uncertain significance"]
df_2c_test.to_csv("./datasets_new/dbnfs_2c_test.csv", index=False)

df_5c_clean = df_lab
df_5c_bal = balance_equal(df_5c_clean)

df_4c_clean = df_lab[df_lab['ClinicalSignificance'] != 'Uncertain significance']
df_4c_bal = balance_equal(df_4c_clean)

clinsig_3 = df_lab['ClinicalSignificance']
clinsig_3 = clinsig_3.replace("Likely pathogenic", "Pathogenic")
clinsig_3 = clinsig_3.replace("Likely benign", "Benign")
df_3c_clean = df_lab.drop('ClinicalSignificance', axis=1)
df_3c_clean['ClinicalSignificance'] = clinsig_3
df_3c_bal = balance_equal(df_3c_clean)

df_2c_clean = df_3c_clean[df_3c_clean['ClinicalSignificance'] != 'Uncertain significance']
df_2c_bal = balance_equal(df_2c_clean)

def sample_balanced(df_in, target_n):
    k = df_in['ClinicalSignificance'].nunique()
    if len(df_in) <= target_n:
        return df_in
    n_per = max(target_n // k, 1)
    parts = []
    for cls, grp in df_in.groupby('ClinicalSignificance'):
        take = min(len(grp), n_per)
        parts.append(grp.sample(n=take, random_state=42))
    out = pd.concat(parts)
    if len(out) < target_n:
        rem = target_n - len(out)
        leftover = df_in.drop(out.index)
        if len(leftover) > 0 and rem > 0:
            extra = leftover.sample(n=min(rem, len(leftover)), random_state=42)
            out = pd.concat([out, extra])
    return out.sample(frac=1.0, random_state=42)

def save_variants(prefix, df_bal):
    sizes = [(10_000, '10k'), (100_000, '100k'), (len(df_bal), 'full')]
    for n, tag in sizes:
        df_s = sample_balanced(df_bal, n)
        print(f"{prefix} {tag}: {len(df_s)}")
        df_s.to_csv(f'./datasets_new/{prefix}_{tag}_{len(df_s)}.csv', index=False)

save_variants('dbnfs_5c_bal', df_5c_bal)
save_variants('dbnfs_4c_bal', df_4c_bal)
save_variants('dbnfs_3c_bal', df_3c_bal)
save_variants('dbnfs_2c_bal', df_2c_bal)