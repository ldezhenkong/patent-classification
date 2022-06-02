import argparse
import io

parser = argparse.ArgumentParser()
parser.add_argument('-input_fname', action='store', dest='input_fname')
parser.add_argument('-subset_size', type=int, default=20)
parser.add_argument('-output_fname')
args = parser.parse_args()

fin = io.open(args.input_fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
n, d = map(int, fin.readline().split())
with open(args.output_fname, 'w') as f:
    f.write(' '.join([str(args.subset_size), str(d)]) + '\n')
    num_lines = 0
    for line in fin:
        if num_lines == args.subset_size:
            break
        f.write(line)
        num_lines += 1

