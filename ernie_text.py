import os
import pandas as pd
from more_itertools import locate

files = os.listdir("/home/muhammed-saeed/Downloads/summary_without_subject_t5_large_tf/")
data = ""
emails = []
summary = []
for file in files:
    path = "/home/muhammed-saeed/Downloads/summary_without_subject_t5_large_tf/" +file
    to_append  = open(path).read() 
    a = to_append.split("################ Text After Summary ######################### ")
    emails.append(a[0].split("################ Text Before Summary ######################### ")[1])
    summary.append(a[1].strip())
    data += to_append

print(f"{len(emails)}")
print(f"{len(summary)}")
print(summary[:3])
print(emails[:3])
d = {"Email":emails, "Summary":summary}
df = pd.DataFrame(d)
print(df.head())
df.to_csv("/home/muhammed-saeed/Downloads/emails_summary.csv")


