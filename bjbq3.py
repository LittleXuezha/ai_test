import time
def check(*args):
    g1, m1, x1 = args
    if g >= 0 and m >= 0 and x >= 0 \
        and g1 == int(g1)\
        and m1 == int(m1)\
        and x1 == int(x1):
        return True
t = time.clock()
count = 0
for g in range(0, 21):
    x = 3/4*g+75
    m = -7/4*g + 25
    count = count+1
    if 5*g+3*m+1/3*x == 100 and g+m+x == 100 and check(g, m, x):
        print("g:{},m:{},x:{}".format(g, m, x))
print("run time:{}s".format(time.clock()-t))
print("count:{}".format(count))

