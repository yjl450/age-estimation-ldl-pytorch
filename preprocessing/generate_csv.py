import csv

path = "list/"

train_align = open("megaage_asian_train_align.csv", "w", newline='')
valid_align = open("megaage_asian_valid_align.csv", "w", newline='')
header = ["photo", "age", "deg", "box1", "box2", "box3", "box4"]
writer1 = csv.writer(train_align)
writer2 = csv.writer(valid_align)
writer1.writerow(header)
writer2.writerow(header)

# train csv
mode = "train"

f_name = open(path+ mode +"_name.txt", "r")
f_age = open(path+ mode+ "_age.txt", "r")
name_data = f_name.readlines()
age_data = f_age.readlines()
lines = len(name_data)
print(lines)
for i in range(lines):
    to_write = []
    to_write.append(mode+"/"+name_data[i].rstrip())
    to_write.append(age_data[i].rstrip())
    writer1.writerow(to_write)
f_name.close()
f_age.close()

# test csv
mode = "test"

f_name = open(path+ mode +"_name.txt", "r")
f_age = open(path+ mode+ "_age.txt", "r")
name_data = f_name.readlines()
age_data = f_age.readlines()
lines = len(name_data)
for i in range(lines):
    to_write = []
    to_write.append(mode+"/"+name_data[i].rstrip())
    to_write.append(age_data[i].rstrip())
    writer2.writerow(to_write)
f_name.close()
f_age.close()

train_align.close()
valid_align.close()