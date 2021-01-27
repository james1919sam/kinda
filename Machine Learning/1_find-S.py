import csv
hypo=['%','%','%','%','%','%']
csv_file=open('finds.csv','r')
readcsv=csv.reader(csv_file,delimiter=',')
print(readcsv)
data=[]
print("\n The given training examples are:")
for row in readcsv:
    print(row)
    if row[len(row)-1].upper()=="YES": 
           data.append(row)
print("The positive examples are:")
for x in data:
    print(x)
    print("\n")
TotalExamples=len(data)
i=0
j=0
k=0
print("the steps of Find-S algorithm are\n",hypo)
list=[]
p=0
d=len(data[p])-1
for j in range(d):
    list.append(data[i][j])
    hypo=list
i=1
for i in range(TotalExamples):
    for k in range(d):
        if hypo[k]!=data[i][k]:
           hypo[k]='?'
           k=k+1
        else:
           hypo[k]
    print(hypo)
print("\n The maximally specific Find-S hypothesis:")
list=[]
for i in range(d):
    list.append(hypo[i])
print(list)
           
