import sys

from fitzpatrick_optimizer.evaluate import main

if __name__ == "__main__":
    main(["--model", "illumination-unet", *sys.argv[1:]])
