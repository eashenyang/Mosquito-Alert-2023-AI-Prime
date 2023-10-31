import argparse
import time

parser = argparse.ArgumentParser(description='test')
parser.add_argument('second', default=0, type=int)

args = parser.parse_args()

print(f'input {args.second} waiting {args.second}')
time.sleep(args.second)
