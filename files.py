import os
dirname = './resources/'
i = 0
for fn in os.listdir(dirname):
    extension = fn.split('.')[-1]
    os.rename(dirname + fn, dirname + 'image_test_' + str(i) + '.' + extension)
    i += 1
