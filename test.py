import argparse

parser = argparse.ArgumentParser(description='Human activity recognition method evaluation')
parser.add_argument('--method', metavar='METHOD', nargs=1, help='method to evaluate (rf, cnn, lstm or fe)')
parser.add_argument('--dataset', metavar='DATASET', nargs=1, help='dataset id (1, 2 or 3)')

res = parser.parse_args()
    


