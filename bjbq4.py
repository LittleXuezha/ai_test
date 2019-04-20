import time
t = time.clock()
count = 0
for g in range(0, 13, 4):
    x = 3 / 4 * g + 75
    m = -7 / 4 * g + 25
    count = count + 1
    if 5*g+3*m+1/3*x == 100:
        print("g:{},m:{},x:{}".format(g, m, x))
print("run time:{}s".format(time.clock()-t))
print("count:{}".format(count))
