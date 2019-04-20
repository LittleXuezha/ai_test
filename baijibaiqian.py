import time
t = time.clock()
count = 0
for g in range(0, 21):
    for m in range(0, 34):
        for x in range(0, 101):
            count += 1
            if 5*g+3*m+1/3*x==100 and g+m+x==100:
                print("g:{},m:{},x:{}".format(g, m, x))
print("run time:{}s".format(time.clock()-t))
print("count:{}".format(count))
