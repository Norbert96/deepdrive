import data_generator as dg
import pandas as pd
import cv2
input_file = './deepdrivedb/deepdrive/linux_recordings/2018-01-18__05-14-48PM'
output_file = '.'


files = dg.get_hdf5_file_names(input_file)


gen = dg.generator(files)
print(gen())

df = pd.DataFrame(columns=('name', 'steering'))

print('Starting time')
import time
t = time.time()
index = 0
for tpl in gen():

    name = str(index) + '.png'
    cv2.imwrite('images/' + name, tpl[0])
    df.loc[index] = [name, tpl[1]]
    print(tpl[1])
    index += 1

print('Process time: ' + str(time.time() - t))
df.to_csv('a.cvs', index=False)
