import sys

from fitzpatrick_optimizer.evaluate import main

if __name__ == "__main__":
    main(["--model", "residual-filter", *sys.argv[1:]])
