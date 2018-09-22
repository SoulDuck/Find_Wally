import sys
def show_progress(i,max_iter):
    msg='\r progress {}/{}'.format(i, max_iter)
    sys.stdout.write(msg)
    sys.stdout.flush()