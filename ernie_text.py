import os
files = os.listdir("/home/muhammed-saeed/Downloads/summary_without_subject_t5_large_tf/")
data = ""
for file in files:
    path = "/home/muhammed-saeed/Downloads/summary_without_subject_t5_large_tf/" +file
    to_append  = open(path).read() +"\n"
    data += to_append

with open("/home/muhammed-saeed/Downloads/merged_text_2.txt", "w") as fb:
    fb.write(data)
data = open("/home/muhammed-saeed/Downloads/merged_text_2.txt").readlines()
emails = []
summary = []
# print(data.split("################ Text After Summary ######################### "))

email_counter = False
# summary_coutner = False
# for line in data:
#     if "################ Text Before Summary #########################" in line:
#         email_counter = True
#         while email_counter:
#             emails.append(line)
#             if "################ Text After Summary #########################" in line:
#                 email_counter = False
#                 summary_coutner = True
#         while summary_coutner:
#             summary.append(line)
#             if "################ Text Before Summary #########################" in line:
#                 summary_coutner =False
#                 email_counter=False

            


# print(summary)