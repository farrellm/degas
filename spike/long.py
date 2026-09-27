import time
for i in range(12):
    open('/content/tick','w').write(str(i))
    print('tick', i, flush=True); time.sleep(1)
