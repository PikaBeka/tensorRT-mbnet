import pandas as pd

df = pd.read_csv("CHWKRS.csv")

print(df["C"])

C = df["C"]
HW = df["HW"]
K = df["K"]
RS = df["RS"]

C_string = "C=("
for c in C:
    C_string += str(c) + " "
print(C_string+")")

HW_string = "HW=("
for c in HW:
    HW_string += str(c) + " "
print(HW_string+")")

K_string = "K=("
for c in K:
    K_string += str(c) + " "
print(K_string+")")

RS_string = "RS=("
for c in RS:
    RS_string += str(c) + " "
print(RS_string+")")
