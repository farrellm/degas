import os, time
for i in range(20):
    if os.path.exists('/content/cancel_flag'):
        print('@@degas {"t":"cancelled","i":%d}' % i, flush=True); break
    print('@@degas {"t":"progress","i":%d}' % i, flush=True)
    time.sleep(1)
