import sys

from fitzpatrick_optimizer.infer import main


if __name__ == "__main__":
    main(["--model", "illumination-unet", *sys.argv[1:]])
