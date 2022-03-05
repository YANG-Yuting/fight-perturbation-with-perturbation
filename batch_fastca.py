import os
fastca_in_dir = 'fastca_ins'
fastca_in_files = os.listdir(fastca_in_dir)
fastca_in_files.sort()
fastca_out_dir = 'fastca_outs'
if not os.path.exists(fastca_out_dir):
    os.makedirs(fastca_out_dir)
finished_out_files = os.listdir('fastca_outs')
for f in fastca_in_files:
    print(f)
    if 'fastca' not in f:
        continue
    if f.replace('in', 'out') in finished_out_files:  # calculated
        continue
    in_file = fastca_in_dir + '/' + f
    os.system('./FastCA %s 100 1' % in_file)
    out_file = in_file.replace('in', 'out')
    print(f)
    if not os.path.exists('CA_array.txt'):  # error: no result of ca
        continue
    os.rename('CA_array.txt', out_file)


