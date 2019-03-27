import sys
import os

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(os.path.basename(sys.argv[0]) + " <path> <N>")
        exit(1)

    data_path = os.path.abspath(sys.argv[1])
    N = int(sys.argv[2])
    
    for n in range(N):
        print(3)    
